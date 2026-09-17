#include "HandwritingPanel.hpp"

#include "EmojiLabel.hpp"

#include <QCoreApplication>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPushButton>
#include <QStandardPaths>
#include <QVBoxLayout>

#include <algorithm>

namespace {

constexpr int kPadSize = 320;
constexpr int kCandidateGlyphSize = 44;

}  // namespace

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

  QObject::connect(_pad, &HandwritingPad::strokesChanged, [this]() {
    recognize();
  });

  setHint(tr("Write a character in the box"));
}

HccrRecognizer* HandwritingPanel::recognizer() {
  if (_recognizer || _loadAttempted) {
    return _recognizer.get();
  }
  _loadAttempted = true;

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

void HandwritingPanel::activate() {
  if (recognizer() == nullptr) {
    setHint(_modelError);
    return;
  }
  if (_pad->isEmpty()) {
    setHint(tr("Write a character in the box"));
  }
}

void HandwritingPanel::clear() {
  const bool hadContent = !_candidates.empty() || !_pad->isEmpty();

  _candidates.clear();
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
  if (_pad->isEmpty()) {
    _candidates.clear();
    refreshCandidates();
    activate();
    return;
  }

  HccrRecognizer* model = recognizer();
  if (model == nullptr) {
    _candidates.clear();
    refreshCandidates();
    setHint(_modelError);
    return;
  }

  _selected = 0;
  _candidates = model->recognize(_pad->toBitmap(), kCandidateCount);
  refreshCandidates();

  if (_candidates.empty()) {
    setHint(tr("Nothing recognized"));
  } else {
    setHint(QString{});
  }
}

void HandwritingPanel::refreshCandidates() {
  for (int index = 0; index < kCandidateCount; ++index) {
    EmojiLabel* label = _candidateLabels[index];
    if (index < static_cast<int>(_candidates.size())) {
      const HccrCandidate& candidate = _candidates[index];
      const Emoji emoji{candidate.character, candidate.character};
      label->setEmoji(emoji, kCandidateGlyphSize, kCandidateGlyphSize);
      label->setToolTip(tr("%1 (%2%)").arg(QString::fromStdString(candidate.character)).arg(candidate.probability * 100.0f, 0, 'f', 1));
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
  Q_EMIT commitRequested(QString::fromStdString(_candidates[index].character));
}

void HandwritingPanel::setHint(const QString& text) {
  // Stays visible even when empty: it holds the stretch that keeps the Clear
  // button at its natural width.
  _hint->setText(text);
}
