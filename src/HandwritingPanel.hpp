#pragma once

#include "EmojiPickerSettings.hpp"
#include "HandwritingPad.hpp"
#include "hccr/HccrRecognizer.hpp"

#include <QLabel>
#include <QString>
#include <QWidget>
#include <memory>
#include <vector>

class EmojiLabel;
class QHBoxLayout;

// The handwriting view: a square ink pad, a row of recognized characters, and a
// status line. Recognition runs on the pad's own thread of control (the GUI
// thread) and takes ~15 ms, well under a frame budget.
class HandwritingPanel : public QWidget {
  Q_OBJECT

public:
  static constexpr int kCandidateCount = 10;

  explicit HandwritingPanel(const EmojiPickerSettings& settings, QWidget* parent = nullptr);

  // Loads the model on first use and restores the idle hint.
  void activate();

  void clear();
  void undoStroke();

  bool hasCandidates() const;
  int candidateCount() const;
  void selectCandidate(int index);
  void selectNext(int delta);
  void commitSelected();
  void commitCandidate(int index);

Q_SIGNALS:
  void commitRequested(const QString& text);

private:
  void recognize();
  void refreshCandidates();
  void setHint(const QString& text);
  HccrRecognizer* recognizer();

  const EmojiPickerSettings& _settings;
  HandwritingPad* _pad = nullptr;
  QHBoxLayout* _candidateRow = nullptr;
  QLabel* _hint = nullptr;
  std::vector<EmojiLabel*> _candidateLabels;
  std::vector<HccrCandidate> _candidates;

  std::unique_ptr<HccrRecognizer> _recognizer;
  QString _modelError;
  bool _loadAttempted = false;
  int _selected = 0;
};
