#pragma once

#include "EmojiPickerSettings.hpp"
#include "HandwritingPad.hpp"
#include "hanzi/HanziRecognizer.hpp"
#include "hccr/HccrRecognizer.hpp"

#include <QLabel>
#include <QString>
#include <QWidget>
#include <memory>
#include <vector>

class EmojiLabel;
class HandwritingWorker;
class QHBoxLayout;
class QPushButton;
class QTimer;

// The handwriting view: a square ink pad, a row of recognized characters, and a
// status line.
//
// Two recognizers fill the row. The classifier (a 3755-class MobileNetV2 on
// ggml) and the trajectory route of the Hanzi matcher answer in milliseconds and
// run on the GUI thread, right where the stroke happened; the field and
// component routes walk a whole charset, so a worker thread runs them and the
// row is updated when they are done. That is what makes rare characters
// reachable at all: the classifier's alphabet is GB2312 level 1, and a
// character outside it - 谞 for instance - can only come from the component
// route (⿰讠胥) or from the charset-wide templates.
class HandwritingPanel : public QWidget {
  Q_OBJECT

public:
  static constexpr int kCandidateCount = 10;

  explicit HandwritingPanel(const EmojiPickerSettings& settings, QWidget* parent = nullptr);
  ~HandwritingPanel() override;

  // Loads what is needed on first use and restores the idle hint.
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
  struct RowCandidate {
    QString character;
    QString origin;            // "model", "trajectory", "field", "composed ⿰讠胥"
    double probability = -1.0; // the classifier's, when it produced this one
  };

  void recognize();
  void requestDetailed();
  void applyDetailed(quint64 generation, const std::vector<hanzi::HanziSource>& sources);
  void applyStatus(const QString& text);
  void rebuildRow();
  void refreshCandidates();
  void setHint(const QString& text);
  HccrRecognizer* recognizer();
  hanzi::HanziRecognizer* hanzi();

  const EmojiPickerSettings& _settings;
  HandwritingPad* _pad = nullptr;
  QHBoxLayout* _candidateRow = nullptr;
  QPushButton* _clearButton = nullptr;
  QLabel* _hint = nullptr;
  QTimer* _detailedTimer = nullptr;
  std::vector<EmojiLabel*> _candidateLabels;
  std::vector<RowCandidate> _candidates;

  std::unique_ptr<HccrRecognizer> _recognizer;
  QString _modelError;
  bool _modelLoadAttempted = false;
  std::vector<HccrCandidate> _modelCandidates;

  std::shared_ptr<hanzi::HanziRecognizer> _hanzi;
  QString _hanziError;
  bool _hanziLoadAttempted = false;
  std::vector<hanzi::HanziSource> _sources;
  std::unique_ptr<HandwritingWorker> _worker;

  // Tags the drawing a result belongs to: strokes change, older answers are
  // dropped instead of replacing a newer row.
  quint64 _generation = 0;

  int _selected = 0;
};
