#pragma once

#include <QImage>
#include <QPointF>
#include <QVector>
#include <QWidget>

class QPainter;

// Square ink pad. Collects pointer strokes (mouse or stylus) and renders them
// the way the recognizer was trained: black ink on white paper.
class HandwritingPad : public QWidget {
  Q_OBJECT

public:
  explicit HandwritingPad(QWidget* parent = nullptr);

  QImage toBitmap() const;
  bool isEmpty() const;
  int strokeCount() const;

public Q_SLOTS:
  void clear();
  void undoStroke();

Q_SIGNALS:
  void strokesChanged();

protected:
  void paintEvent(QPaintEvent* event) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void tabletEvent(QTabletEvent* event) override;

private:
  int penWidth() const;
  void addPoint(const QPointF& point);
  void endStroke();
  void renderInto(QPainter& painter) const;

  QVector<QVector<QPointF>> _strokes;
  bool _drawing = false;
  QPointF _lastPoint;
};
