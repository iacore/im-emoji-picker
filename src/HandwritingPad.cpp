#include "HandwritingPad.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QTabletEvent>

#include <algorithm>
#include <cmath>

namespace {

// Ink width as a fraction of the pad. The recognizer was measured to be
// insensitive between roughly 3px and 16px of ink on a 300px canvas, which this
// ratio reproduces at any pad size.
constexpr double kPenWidthRatio = 0.035;
constexpr int kMinimumPenWidth = 3;

}  // namespace

HandwritingPad::HandwritingPad(QWidget* parent) : QWidget(parent) {
  setFocusPolicy(Qt::NoFocus);
  setAttribute(Qt::WA_ShowWithoutActivating);
  setCursor(Qt::CrossCursor);
  setMinimumSize(120, 120);
  setToolTip(tr("Draw a character here"));
}

int HandwritingPad::penWidth() const {
  return std::max(kMinimumPenWidth, static_cast<int>(std::lround(std::min(width(), height()) * kPenWidthRatio)));
}

bool HandwritingPad::isEmpty() const {
  return _strokes.isEmpty();
}

int HandwritingPad::strokeCount() const {
  return _strokes.size();
}

void HandwritingPad::clear() {
  if (_strokes.isEmpty()) {
    return;
  }
  _strokes.clear();
  update();
  Q_EMIT strokesChanged();
}

void HandwritingPad::undoStroke() {
  if (_strokes.isEmpty()) {
    return;
  }
  _strokes.removeLast();
  update();
  Q_EMIT strokesChanged();
}

void HandwritingPad::addPoint(const QPointF& point) {
  _strokes.last().append(point);
  _lastPoint = point;
  update();
}

void HandwritingPad::endStroke() {
  if (!_drawing) {
    return;
  }
  _drawing = false;
  update();
  Q_EMIT strokesChanged();
}

void HandwritingPad::mousePressEvent(QMouseEvent* event) {
  if (event->button() != Qt::LeftButton) {
    QWidget::mousePressEvent(event);
    return;
  }

  _drawing = true;
  _strokes.append(QVector<QPointF>{});
  addPoint(event->pos());
  event->accept();
}

void HandwritingPad::mouseMoveEvent(QMouseEvent* event) {
  if (!_drawing) {
    event->ignore();
    return;
  }

  addPoint(event->pos());
  event->accept();
}

void HandwritingPad::mouseReleaseEvent(QMouseEvent* event) {
  if (event->button() != Qt::LeftButton) {
    QWidget::mouseReleaseEvent(event);
    return;
  }

  addPoint(event->pos());
  endStroke();
  event->accept();
}

void HandwritingPad::tabletEvent(QTabletEvent* event) {
  switch (event->type()) {
  case QEvent::TabletPress:
    _drawing = true;
    _strokes.append(QVector<QPointF>{});
    addPoint(event->posF());
    break;

  case QEvent::TabletMove:
    if (_drawing) {
      addPoint(event->posF());
    }
    break;

  case QEvent::TabletRelease:
    if (_drawing) {
      addPoint(event->posF());
      endStroke();
    }
    break;

  default:
    QWidget::tabletEvent(event);
    return;
  }

  event->accept();
}

void HandwritingPad::renderInto(QPainter& painter) const {
  QPen pen{Qt::black, static_cast<double>(penWidth()), Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin};
  painter.setPen(pen);

  for (const QVector<QPointF>& stroke : _strokes) {
    if (stroke.size() == 1) {
      painter.drawPoint(stroke.first());
      continue;
    }
    painter.drawPolyline(stroke.data(), stroke.size());
  }
}

void HandwritingPad::paintEvent(QPaintEvent* event) {
  Q_UNUSED(event)

  QPainter painter{this};
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.fillRect(rect(), Qt::white);
  renderInto(painter);

  painter.setPen(QPen{QColor{0, 0, 0, 60}});
  painter.setBrush(Qt::NoBrush);
  painter.drawRect(rect().adjusted(0, 0, -1, -1));
}

QImage HandwritingPad::toBitmap() const {
  QImage bitmap{size(), QImage::Format_Grayscale8};
  bitmap.fill(255);

  QPainter painter{&bitmap};
  painter.setRenderHint(QPainter::Antialiasing, true);
  renderInto(painter);
  painter.end();

  return bitmap;
}
