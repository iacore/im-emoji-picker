#include "HandwritingPanel.hpp"

#include "EmojiLabel.hpp"
#include "HandwritingWorker.hpp"

#include <QCoreApplication>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPushButton>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>

namespace {

constexpr int kPadSize = 320;
constexpr int kCandidateGlyphSize = 44;

// Long enough that a stroke in progress does not start a pass, short enough
// that the row is filled in while the user is still looking at it.
constexpr int kDetailedDelayMilliseconds = 350;

hanzi::Strokes toHanziStrokes(const QVector<QVector<QPointF>>& strokes) {
  hanzi::Strokes converted;
  converted.reserve(strokes.size());
  for (const QVector<QPointF>& stroke : strokes) {
    hanzi::Stroke points;
    points.reserve(stroke.size());
    for (const QPointF& point : stroke) {
      points.push_back(hanzi::Point{point.x(), point.y()});
    }
    converted.push_back(std::move(points));
  }
  return converted;
}

// Where the rare-character data lives: the setting, then the environment, then
// the installed location - the same order the classifier's model file uses.
QString dataDirectory(const EmojiPickerSettings& settings) {
  const QString configured = QString::fromStdString(settings.handwritingDataPath());
  if (!configured.isEmpty()) {
    return configured;
  }
  const QByteArray fromEnvironment = qgetenv("IM_EMOJI_PICKER_HANZI_DATA");
  if (!fromEnvironment.isEmpty()) {
    return QString::fromLocal8Bit(fromEnvironment);
  }
  return QStandardPaths::locate(QStandardPaths::GenericDataLocation, "im-emoji-picker/hanzi", QStandardPaths::LocateDirectory);
}

// The templates are a cache, not data: they are rebuilt from the system's fonts
// and weigh a few hundred megabytes, so they belong where caches belong.
QString templateCacheDirectory() {
  const QByteArray fromEnvironment = qgetenv("IM_EMOJI_PICKER_HANZI_CACHE");
  if (!fromEnvironment.isEmpty()) {
    return QString::fromLocal8Bit(fromEnvironment);
  }
  return QStandardPaths::writableLocation(QStandardPaths::CacheLocation) + "/hanzi-templates";
}

} // namespace

HandwritingPanel::HandwritingPanel(const EmojiPickerSettings& settings, QWidget* parent) : QWidget(parent), _settings{settings} {
  setFocusPolicy(Qt::NoFocus);
  setAttribute(Qt::WA_ShowWithoutActivating);

  auto* layout = new QVBoxLayout{this};
  layout->setContentsMargins(8, 4, 8, 4);
  layout->setSpacing(4);

  _pad = new HandwritingPad{this};
  _pad->setFixedSize(kPadSize, kPadSize);
  layout->addWidget(_pad, 0, Qt::AlignHCenter);

  _candidateRow = new QHBoxLayout{};
  _candidateRow->setSpacing(2);
  _candidateRow->setContentsMargins(0, 0, 0, 0);

  _candidateRow->addStretch(1);
  for (int index = 0; index < kCandidateCount; ++index) {
    auto* label = new EmojiLabel{this, _settings, Emoji{"", ""}};
    label->setEmoji(Emoji{"", ""}, kCandidateGlyphSize, kCandidateGlyphSize);
    label->hide();
    QObject::connect(label, &EmojiLabel::mousePressed, [this, index]() {
      commitCandidate(index);
    });
    _candidateLabels.push_back(label);
    _candidateRow->addWidget(label);
  }
  _candidateRow->addStretch(1);
  layout->addLayout(_candidateRow);

  auto* bottomRow = new QHBoxLayout{};
  bottomRow->setSpacing(8);

  _clearButton = new QPushButton{tr("Clear"), this};
  _clearButton->setFocusPolicy(Qt::NoFocus);
  _clearButton->setToolTip(tr("Wipe the pad (or right click it, or press Ctrl+Backspace)"));
  QObject::connect(_clearButton, &QPushButton::clicked, [this]() {
    clear();
  });
  bottomRow->addWidget(_clearButton);

  _hint = new QLabel{this};
  _hint->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
  _hint->setWordWrap(true);
  bottomRow->addWidget(_hint, 1);

  layout->addLayout(bottomRow);
  layout->addStretch(1);

  // The heavy routes are asked for once the drawing stops changing, so a pass
  // never starts on a character that is still being written.
  _detailedTimer = new QTimer{this};
  _detailedTimer->setSingleShot(true);
  _detailedTimer->setInterval(kDetailedDelayMilliseconds);
  QObject::connect(_detailedTimer, &QTimer::timeout, [this]() {
    requestDetailed();
  });

  QObject::connect(_pad, &HandwritingPad::strokesChanged, [this]() {
    recognize();
  });

  setHint(tr("Write a character in the box"));
}

HandwritingPanel::~HandwritingPanel() = default;

HccrRecognizer* HandwritingPanel::recognizer() {
  if (_recognizer || _modelLoadAttempted) {
    return _recognizer.get();
  }
  _modelLoadAttempted = true;

  QString path = QString::fromStdString(_settings.handwritingModelPath());
  if (path.isEmpty()) {
    const QByteArray fromEnvironment = qgetenv("IM_EMOJI_PICKER_HCCR_MODEL");
    if (!fromEnvironment.isEmpty()) {
      path = QString::fromLocal8Bit(fromEnvironment);
    } else {
      path = QStandardPaths::locate(QStandardPaths::GenericDataLocation, "im-emoji-picker/hccr-mobilenetv2.gguf");
    }
  }

  if (path.isEmpty()) {
    _modelError = tr("Handwriting model not found. Install hccr-mobilenetv2.gguf or set handwritingModelPath.");
    return nullptr;
  }

  std::string error;
  _recognizer = HccrRecognizer::load(path.toStdString(), &error);
  if (!_recognizer) {
    _modelError = tr("Cannot load %1: %2").arg(path, QString::fromStdString(error));
  }
  return _recognizer.get();
}

hanzi::HanziRecognizer* HandwritingPanel::hanzi() {
  if (_hanzi || _hanziLoadAttempted) {
    return _hanzi.get();
  }
  _hanziLoadAttempted = true;

  const QString directory = dataDirectory(_settings);
  if (directory.isEmpty()) {
    _hanziError = tr("Rare-character data not found. Install hanzi-dictionary.bin and hanzi-ids.bin, or set handwritingDataPath.");
    return nullptr;
  }

  hanzi::HanziRecognizer::Paths paths;
  paths.dictionary = directory + "/hanzi-dictionary.bin";
  paths.ids = directory + "/hanzi-ids.bin";
  paths.templateCache = templateCacheDirectory();

  QString error;
  std::unique_ptr<hanzi::HanziRecognizer> loaded = hanzi::HanziRecognizer::create(paths, &error);
  if (!loaded) {
    _hanziError = error;
    return nullptr;
  }
  _hanzi = std::move(loaded);
  return _hanzi.get();
}

void HandwritingPanel::activate() {
  if (recognizer() == nullptr && hanzi() == nullptr) {
    setHint(_modelError.isEmpty() ? _hanziError : _modelError);
    return;
  }
  if (_pad->isEmpty()) {
    setHint(tr("Write a character in the box"));
  }
}

void HandwritingPanel::clear() {
  const bool hadContent = !_candidates.empty() || !_pad->isEmpty();

  // Drop whatever is in flight: it belongs to the drawing being wiped.
  ++_generation;
  _detailedTimer->stop();
  _candidates.clear();
  _modelCandidates.clear();
  _sources.clear();
  refreshCandidates();
  _pad->clear();

  if (hadContent) {
    setHint(tr("Write a character in the box"));
  }
}

void HandwritingPanel::undoStroke() {
  _pad->undoStroke();
}

int HandwritingPanel::candidateCount() const {
  return static_cast<int>(_candidates.size());
}

bool HandwritingPanel::hasCandidates() const {
  return !_candidates.empty();
}

void HandwritingPanel::recognize() {
  ++_generation;

  if (_pad->isEmpty()) {
    _candidates.clear();
    _modelCandidates.clear();
    _sources.clear();
    refreshCandidates();
    activate();
    return;
  }

  _selected = 0;
  // The detailed answer belonged to the previous drawing; the quick pass puts
  // the trajectory route back in the row for this one.
  _sources.clear();
  _detailedTimer->stop();

  if (HccrRecognizer* model = recognizer()) {
    _modelCandidates = model->recognize(_pad->toBitmap(), kCandidateCount);
  } else {
    _modelCandidates.clear();
  }

  if (hanzi::HanziRecognizer* matcher = hanzi()) {
    _sources = matcher->rankStrokes(toHanziStrokes(_pad->strokes()), kCandidateCount, hanzi::HanziRecognizer::Depth::Quick);
  }

  rebuildRow();

  if (_candidates.empty()) {
    setHint(_modelError.isEmpty() ? _hanziError : _modelError);
  } else {
    setHint(QString{});
  }

  if (_hanzi) {
    _detailedTimer->start();
  }
}

void HandwritingPanel::requestDetailed() {
  if (!_hanzi || _pad->isEmpty()) {
    return;
  }

  if (!_worker) {
    _worker = std::make_unique<HandwritingWorker>(
        _hanzi, kCandidateCount,
        [this](quint64 generation, std::vector<hanzi::HanziSource> sources) {
          // Off the worker thread: hand the answer to the GUI thread, which is
          // also the only place the panel may be touched.
          auto payload = std::make_shared<std::vector<hanzi::HanziSource>>(std::move(sources));
          QMetaObject::invokeMethod(
              this,
              [this, generation, payload]() {
                applyDetailed(generation, *payload);
              },
              Qt::QueuedConnection);
        },
        [this](quint64 generation, const QString& status) {
          Q_UNUSED(generation)
          const auto text = std::make_shared<QString>(status);
          QMetaObject::invokeMethod(
              this,
              [this, text]() {
                applyStatus(*text);
              },
              Qt::QueuedConnection);
        });
  }

  _worker->request(_generation, toHanziStrokes(_pad->strokes()));
}

void HandwritingPanel::applyDetailed(quint64 generation, const std::vector<hanzi::HanziSource>& sources) {
  if (generation != _generation) {
    return;
  }
  _sources = sources;
  rebuildRow();
  if (!_candidates.empty()) {
    setHint(QString{});
  }
}

void HandwritingPanel::applyStatus(const QString& text) {
  setHint(text);
}

void HandwritingPanel::rebuildRow() {
  std::vector<hanzi::HanziSource> sources;
  if (!_modelCandidates.empty()) {
    hanzi::HanziSource model;
    model.origin = QStringLiteral("model");
    model.weight = hanzi::kModelWeight;
    for (const HccrCandidate& candidate : _modelCandidates) {
      model.ranked.push_back(QString::fromStdString(candidate.character));
    }
    sources.push_back(std::move(model));
  }
  sources.insert(sources.end(), _sources.begin(), _sources.end());

  const std::vector<hanzi::HanziCandidate> merged = hanzi::mergeSources(sources, kCandidateCount);
  _candidates.clear();
  _candidates.reserve(merged.size());
  for (const hanzi::HanziCandidate& candidate : merged) {
    RowCandidate row;
    row.character = candidate.character;
    row.origin = candidate.origin;
    for (const HccrCandidate& model : _modelCandidates) {
      if (QString::fromStdString(model.character) == candidate.character) {
        row.probability = model.probability;
        break;
      }
    }
    _candidates.push_back(std::move(row));
  }

  if (_selected >= static_cast<int>(_candidates.size())) {
    _selected = std::max(0, static_cast<int>(_candidates.size()) - 1);
  }
  refreshCandidates();
}

void HandwritingPanel::refreshCandidates() {
  for (int index = 0; index < kCandidateCount; ++index) {
    EmojiLabel* label = _candidateLabels[index];
    if (index < static_cast<int>(_candidates.size())) {
      const RowCandidate& candidate = _candidates[index];
      const Emoji emoji{candidate.character.toStdString(), candidate.character.toStdString()};
      label->setEmoji(emoji, kCandidateGlyphSize, kCandidateGlyphSize);
      if (candidate.probability >= 0.0) {
        label->setToolTip(tr("%1 - %2 (%3%)").arg(candidate.character, candidate.origin).arg(candidate.probability * 100.0, 0, 'f', 1));
      } else {
        label->setToolTip(tr("%1 - %2").arg(candidate.character, candidate.origin));
      }
      label->setHighlighted(index == _selected);
      label->show();
    } else {
      label->hide();
    }
  }
}

void HandwritingPanel::selectCandidate(int index) {
  if (_candidates.empty()) {
    return;
  }
  _selected = std::clamp(index, 0, static_cast<int>(_candidates.size()) - 1);
  refreshCandidates();
}

void HandwritingPanel::selectNext(int delta) {
  if (_candidates.empty()) {
    return;
  }
  const int count = static_cast<int>(_candidates.size());
  _selected = ((_selected + delta) % count + count) % count;
  refreshCandidates();
}

void HandwritingPanel::commitSelected() {
  commitCandidate(_selected);
}

void HandwritingPanel::commitCandidate(int index) {
  if (index < 0 || index >= static_cast<int>(_candidates.size())) {
    return;
  }
  Q_EMIT commitRequested(_candidates[index].character);
}

void HandwritingPanel::setHint(const QString& text) {
  // Stays visible even when empty: it holds the stretch that keeps the Clear
  // button at its natural width.
  _hint->setText(text);
}
