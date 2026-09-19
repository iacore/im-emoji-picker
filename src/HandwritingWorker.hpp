#pragma once

#include "hanzi/HanziRecognizer.hpp"

#include <QString>

#include <functional>
#include <memory>
#include <thread>
#include <vector>

// Runs the heavy Hanzi routes - the charset-wide field templates and the
// component route, seconds each and minutes for the first template build - off
// the GUI thread.
//
// Requests are a one-slot mailbox: a newer drawing replaces a queued one, so a
// fast hand never queues work, and every result is tagged with the generation of
// the drawing it came from, so a stale one can be dropped. The worker shares
// ownership of the recognizer, which is what makes it safe to walk away from it:
// on destruction the callbacks stop and the thread finishes its pass on its own,
// still holding the recognizer alive. Joining instead would make closing the
// picker wait for a first-run build of a whole charset, and killing the thread
// mid-build is exactly what the templates' atomic rename exists to survive.
class HandwritingWorker {
public:
  // Both callbacks run on the worker thread.
  using Deliver = std::function<void(quint64 generation, std::vector<hanzi::HanziSource> sources)>;
  using Report = std::function<void(quint64 generation, const QString& status)>;

  HandwritingWorker(std::shared_ptr<hanzi::HanziRecognizer> recognizer, int candidates, Deliver deliver, Report report);
  ~HandwritingWorker();
  HandwritingWorker(const HandwritingWorker&) = delete;
  HandwritingWorker& operator=(const HandwritingWorker&) = delete;

  // Replaces a request that has not started yet. Does nothing once the worker
  // is being torn down.
  void request(quint64 generation, hanzi::Strokes strokes);

private:
  struct State;

  // The thread body. A static member so it can outlive the object that started
  // it: it touches only what it is handed.
  static void run(std::shared_ptr<State> state, std::shared_ptr<hanzi::HanziRecognizer> recognizer, int candidates);

  std::shared_ptr<State> _state;
  std::shared_ptr<hanzi::HanziRecognizer> _recognizer;
  std::thread _thread;
};
