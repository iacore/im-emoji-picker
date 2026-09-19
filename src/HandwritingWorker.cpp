#include "HandwritingWorker.hpp"

#include <condition_variable>
#include <mutex>
#include <utility>

struct HandwritingWorker::State {
  struct Pending {
    quint64 generation = 0;
    hanzi::Strokes strokes;
    bool valid = false;
  };

  std::mutex mutex;
  std::condition_variable wake;
  Pending pending;
  bool quit = false;
  Deliver deliver;
  Report report;
};

void HandwritingWorker::run(std::shared_ptr<State> state, std::shared_ptr<hanzi::HanziRecognizer> recognizer, int candidates) {
  for (;;) {
    State::Pending work;
    {
      std::unique_lock<std::mutex> lock{state->mutex};
      state->wake.wait(lock, [&]() {
        return state->quit || state->pending.valid;
      });
      if (state->quit) {
        return;
      }
      work = std::move(state->pending);
      state->pending = State::Pending{};
    }

    std::vector<hanzi::HanziSource> sources = recognizer->rankStrokes(work.strokes, candidates, hanzi::HanziRecognizer::Depth::Detailed, [&](const QString& status) {
      std::lock_guard<std::mutex> lock{state->mutex};
      if (state->report) {
        state->report(work.generation, status);
      }
    });

    std::lock_guard<std::mutex> lock{state->mutex};
    if (state->quit) {
      return;
    }
    if (state->deliver) {
      state->deliver(work.generation, std::move(sources));
    }
  }
}

HandwritingWorker::HandwritingWorker(std::shared_ptr<hanzi::HanziRecognizer> recognizer, int candidates, Deliver deliver, Report report) : _state{std::make_shared<State>()}, _recognizer{std::move(recognizer)} {
  _state->deliver = std::move(deliver);
  _state->report = std::move(report);

  const std::shared_ptr<State> state = _state;
  const std::shared_ptr<hanzi::HanziRecognizer> shared = _recognizer;
  _thread = std::thread{[state, shared, candidates]() {
    run(state, shared, candidates);
  }};
}

HandwritingWorker::~HandwritingWorker() {
  {
    std::lock_guard<std::mutex> lock{_state->mutex};
    _state->quit = true;
    _state->pending = State::Pending{};
    _state->deliver = nullptr;
    _state->report = nullptr;
  }
  _state->wake.notify_all();

  if (_thread.joinable()) {
    _thread.detach();
  }
}

void HandwritingWorker::request(quint64 generation, hanzi::Strokes strokes) {
  {
    std::lock_guard<std::mutex> lock{_state->mutex};
    if (_state->quit) {
      return;
    }
    _state->pending = State::Pending{generation, std::move(strokes), true};
  }
  _state->wake.notify_all();
}
