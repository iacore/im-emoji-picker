#include "hanzi/Ink.hpp"

#include "ImageResample.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace hanzi {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Rasteriser. render_strokes() draws through PIL's ImageDraw, which has no
// antialiasing, so the reference ink is a hard-edged bitmap and the whole
// normalization pipeline is measured against those pixels. Everything below is
// a port of PIL 12.3.0's libImaging/Draw.c plus the _draw_lines, _draw_ellipse
// and _draw_pieslice entry points of _imaging.c, float truncations included:
// the goal is the same pixels, not a nicer line.
//
// Only the 8-bit image paths are ported (mode "L", blend 0, no mask), which is
// what ink.py uses; the alpha paths of polygon_generic are not reachable here.

// libImaging/Draw.c: ROUND_UP. Round to nearest, half away from zero.
int roundUp(double v) {
  return static_cast<int>(v >= 0.0 ? std::floor(v + 0.5) : -std::floor(std::fabs(v) + 0.5));
}

// libImaging/Draw.c: ROUND_DOWN. Round the interval inwards, half towards zero.
int roundDown(double v) {
  return static_cast<int>(v >= 0.0 ? std::ceil(v - 0.5) : -std::ceil(std::fabs(v) - 0.5));
}

// _imaging.c: point8().
void drawPoint(QImage& image, int x, int y, uint8_t ink) {
  if (x >= 0 && x < image.width() && y >= 0 && y < image.height()) {
    image.scanLine(y)[x] = ink;
  }
}

// _imaging.c: hline8().
void drawHLine(QImage& image, int x0, int y, int x1, uint8_t ink) {
  if (y < 0 || y >= image.height()) {
    return;
  }
  if (x0 < 0) {
    x0 = 0;
  } else if (x0 >= image.width()) {
    return;
  }
  if (x1 < 0) {
    return;
  } else if (x1 >= image.width()) {
    x1 = image.width() - 1;
  }
  if (x0 <= x1) {
    std::memset(image.scanLine(y) + x0, ink, static_cast<size_t>(x1 - x0 + 1));
  }
}

// _imaging.c: line8(). Bresenham, and it stops one pixel short of the end: for
// a one-pixel pen _draw_lines draws that last pixel separately.
void drawThinLine(QImage& image, int x0, int y0, int x1, int y1, uint8_t ink) {
  int dx = x1 - x0;
  const int xs = dx < 0 ? (dx = -dx, -1) : 1;
  int dy = y1 - y0;
  const int ys = dy < 0 ? (dy = -dy, -1) : 1;
  int n = dx > dy ? dx : dy;

  if (dx == 0) {
    for (int i = 0; i < dy; ++i) {
      drawPoint(image, x0, y0, ink);
      y0 += ys;
    }
  } else if (dy == 0) {
    for (int i = 0; i < dx; ++i) {
      drawPoint(image, x0, y0, ink);
      x0 += xs;
    }
  } else if (dx > dy) {
    n = dx;
    dy += dy;
    int e = dy - dx;
    dx += dx;
    for (int i = 0; i < n; ++i) {
      drawPoint(image, x0, y0, ink);
      if (e >= 0) {
        y0 += ys;
        e -= dx;
      }
      e += dy;
      x0 += xs;
    }
  } else {
    n = dy;
    dx += dx;
    int e = dx - dy;
    dy += dy;
    for (int i = 0; i < n; ++i) {
      drawPoint(image, x0, y0, ink);
      if (e >= 0) {
        x0 += xs;
        e -= dy;
      }
      e += dx;
      y0 += ys;
    }
  }
}

// libImaging/Draw.c: Edge, as far as polygon_generic() reads it (the `d` field
// of the original struct is only used by the outline code).
struct Edge {
  int x0 = 0;
  int y0 = 0;
  int xmin = 0;
  int ymin = 0;
  int xmax = 0;
  int ymax = 0;
  float dx = 0.0f;
};

// libImaging/Draw.c: add_edge().
void addEdge(Edge* edge, int x0, int y0, int x1, int y1) {
  if (x0 <= x1) {
    edge->xmin = x0;
    edge->xmax = x1;
  } else {
    edge->xmin = x1;
    edge->xmax = x0;
  }
  if (y0 <= y1) {
    edge->ymin = y0;
    edge->ymax = y1;
  } else {
    edge->ymin = y1;
    edge->ymax = y0;
  }
  if (y0 == y1) {
    edge->dx = 0.0f;
  } else {
    edge->dx = static_cast<float>(x1 - x0) / static_cast<float>(y1 - y0);
  }
  edge->x0 = x0;
  edge->y0 = y0;
}

// libImaging/Draw.c: polygon_generic(), the 8-bit hline path. Edges are filled
// scanline by scanline from per-edge x intersections rounded into the interval,
// and the corner adjustment keeps a vertex from dropping a pixel when two
// edges meet at a scanline end. Only ever called with the four edges of a wide
// line, so the working sets are fixed size.
void fillPolygon(QImage& image, int count, const Edge* edges, uint8_t ink) {
  if (count <= 0) {
    return;
  }

  std::array<const Edge*, 4> table{};
  int tableCount = 0;
  int ymin = image.height() - 1;
  int ymax = 0;
  for (int i = 0; i < count; ++i) {
    ymin = std::min(ymin, edges[i].ymin);
    ymax = std::max(ymax, edges[i].ymax);
    if (edges[i].ymin == edges[i].ymax) {
      drawHLine(image, edges[i].xmin, edges[i].ymin, edges[i].xmax, ink);
      continue;
    }
    table[tableCount++] = &edges[i];
  }
  ymin = std::max(ymin, 0);
  ymax = std::min(ymax, image.height());

  // Four edges at most, each contributing a crossing plus, at most once, a
  // repeated one for the scanline through its far end: never more than eight.
  std::array<float, 8> crossings{};
  for (; ymin <= ymax; ++ymin) {
    int total = 0;
    for (int i = 0; i < tableCount && i < 4; ++i) {
      const Edge* current = table[i];
      if (ymin < current->ymin || ymin > current->ymax) {
        continue;
      }
      crossings[total] = (ymin - current->y0) * current->dx + current->x0;
      ++total;
      if (ymin == current->ymax && ymin < ymax) {
        crossings[total] = crossings[total - 1];
        ++total;
      } else if ((ymin == current->ymin || ymin == current->ymax) && current->dx != 0.0f) {
        for (int k = 0; k < i; ++k) {
          const Edge* other = table[k];
          if ((ymin != other->ymin && ymin != other->ymax) || other->dx == 0.0f) {
            continue;
          }
          const float otherX = (ymin - other->y0) * other->dx + other->x0;
          if (std::round(crossings[total - 1]) != std::round(otherX)) {
            continue;
          }
          // The two edges meet in a corner: line up this crossing with the
          // row below (or above, on the last row) so the corner stays filled.
          const int offset = ymin == current->ymax ? -1 : 1;
          const float adjacent = (ymin + offset - current->y0) * current->dx + current->x0;
          if (ymin + offset >= other->ymin && ymin + offset <= other->ymax) {
            const float adjacentOther = (ymin + offset - other->y0) * other->dx + other->x0;
            if (crossings[total - 1] > adjacent + 1 && crossings[total - 1] > adjacentOther + 1) {
              crossings[total - 1] = std::round(std::fmax(adjacent, adjacentOther)) + 1;
            } else if (crossings[total - 1] < adjacent - 1 && crossings[total - 1] < adjacentOther - 1) {
              crossings[total - 1] = std::round(std::fmin(adjacent, adjacentOther)) - 1;
            }
            break;
          }
        }
      }
    }
    // PIL sorts the crossings with qsort before pairing them; with at most
    // eight values an insertion sort reaches the same order, and equal values
    // are interchangeable because the pairing only reads their coordinates.
    for (int i = 1; i < total; ++i) {
      const float value = crossings[static_cast<size_t>(i)];
      int slot = i;
      while (slot > 0 && crossings[static_cast<size_t>(slot - 1)] > value) {
        crossings[static_cast<size_t>(slot)] = crossings[static_cast<size_t>(slot - 1)];
        --slot;
      }
      crossings[static_cast<size_t>(slot)] = value;
    }
    for (int i = 1; i < total; i += 2) {
      drawHLine(image, roundUp(crossings[static_cast<size_t>(i - 1)]), ymin, roundDown(crossings[static_cast<size_t>(i)]), ink);
    }
  }
}

// libImaging/Draw.c: ImagingDrawWideLine(). A pen wider than one pixel is the
// slanted quad spanned by the segment, offset by the integer approximations of
// half the pen, filled by the scanline code above. The ends are not round.
void drawWideLine(QImage& image, int x0, int y0, int x1, int y1, int width, uint8_t ink) {
  const int dx = x1 - x0;
  const int dy = y1 - y0;
  if (dx == 0 && dy == 0) {
    drawPoint(image, x0, y0, ink);
    return;
  }

  const double bigHypotenuse = std::hypot(static_cast<double>(dx), static_cast<double>(dy));
  const double smallHypotenuse = (width - 1) / 2.0;
  const double ratioMax = roundUp(smallHypotenuse) / bigHypotenuse;
  const double ratioMin = roundDown(smallHypotenuse) / bigHypotenuse;

  const int dxmin = roundDown(ratioMin * dy);
  const int dxmax = roundDown(ratioMax * dy);
  const int dymin = roundDown(ratioMin * dx);
  const int dymax = roundDown(ratioMax * dx);

  const int vertices[4][2] = {
      {x0 - dxmin, y0 + dymax},
      {x1 - dxmin, y1 + dymax},
      {x1 + dxmax, y1 - dymin},
      {x0 + dxmax, y0 - dymin},
  };
  std::array<Edge, 4> edges{};
  for (int i = 0; i < 4; ++i) {
    const int next = (i + 1) % 4;
    addEdge(&edges[i], vertices[i][0], vertices[i][1], vertices[next][0], vertices[next][1]);
  }
  fillPolygon(image, 4, edges.data(), ink);
}

// libImaging/Draw.c: quarter_init()/quarter_delta()/quarter_next(). One quarter
// of an integer-grid ellipse, walked by Bresenham on the ellipse equation.
struct QuarterState {
  int32_t a = 0;
  int32_t b = 0;
  int32_t cx = 0;
  int32_t cy = 0;
  int32_t ex = 0;
  int32_t ey = 0;
  int64_t a2 = 0;
  int64_t b2 = 0;
  int64_t a2b2 = 0;
  bool finished = false;
};

void quarterInit(QuarterState* state, int32_t a, int32_t b) {
  if (a < 0 || b < 0) {
    state->finished = true;
    return;
  }
  state->a = a;
  state->b = b;
  state->cx = a;
  state->cy = b % 2;
  state->ex = a % 2;
  state->ey = b;
  state->a2 = static_cast<int64_t>(a) * a;
  state->b2 = static_cast<int64_t>(b) * b;
  state->a2b2 = state->a2 * state->b2;
  state->finished = false;
}

int64_t quarterDelta(const QuarterState& state, int64_t x, int64_t y) {
  return std::llabs(state.a2 * y * y + state.b2 * x * x - state.a2b2);
}

bool quarterNext(QuarterState* state, int32_t* outX, int32_t* outY) {
  if (state->finished) {
    return false;
  }
  *outX = state->cx;
  *outY = state->cy;
  if (state->cx == state->ex && state->cy == state->ey) {
    state->finished = true;
    return true;
  }
  int32_t nx = state->cx;
  int32_t ny = state->cy + 2;
  int64_t delta = quarterDelta(*state, nx, ny);
  if (nx > 1) {
    int64_t candidate = quarterDelta(*state, state->cx - 2, state->cy + 2);
    if (delta > candidate) {
      nx = state->cx - 2;
      ny = state->cy + 2;
      delta = candidate;
    }
    candidate = quarterDelta(*state, state->cx - 2, state->cy);
    if (delta > candidate) {
      nx = state->cx - 2;
      ny = state->cy;
    }
  }
  state->cx = nx;
  state->cy = ny;
  return true;
}

// libImaging/Draw.c: ellipse_state, ellipse_init() and ellipse_next(). The
// outer quarter gives the right edge of each scanline, the inner one the left
// edge of any hollow ring; with `width = a + b` (the fill case the reference
// uses) the inner quarter is always empty and every row is solid.
struct EllipseState {
  QuarterState outer;
  QuarterState inner;
  int32_t py = 0;
  int32_t pl = 0;
  int32_t pr = 0;
  std::array<int32_t, 4> cl{};
  std::array<int32_t, 4> cy{};
  std::array<int32_t, 4> cr{};
  int bufferCount = 0;
  bool finished = false;
  int32_t leftmost = 0;
};

void ellipseInit(EllipseState* state, int32_t a, int32_t b, int32_t width) {
  state->bufferCount = 0;
  state->leftmost = a % 2;
  quarterInit(&state->outer, a, b);
  if (width < 1 || !quarterNext(&state->outer, &state->pr, &state->py)) {
    state->finished = true;
    return;
  }
  state->finished = false;
  quarterInit(&state->inner, a - 2 * (width - 1), b - 2 * (width - 1));
  state->pl = state->leftmost;
}

bool ellipseNext(EllipseState* state, int32_t* outX0, int32_t* outY, int32_t* outX1) {
  if (state->bufferCount == 0) {
    if (state->finished) {
      return false;
    }
    const int32_t y = state->py;
    int32_t left = state->pl;
    const int32_t right = state->pr;
    int32_t cx = 0;
    int32_t cy = 0;
    bool alive = true;
    while ((alive = quarterNext(&state->outer, &cx, &cy)) && cy <= y) {
    }
    if (!alive) {
      state->finished = true;
    } else {
      state->pr = cx;
      state->py = cy;
    }
    while ((alive = quarterNext(&state->inner, &cx, &cy)) && cy <= y) {
      left = cx;
    }
    state->pl = alive ? cx : state->leftmost;

    if ((left > 0 || left < right) && y > 0) {
      state->cl[state->bufferCount] = left == 0 ? 2 : left;
      state->cy[state->bufferCount] = y;
      state->cr[state->bufferCount] = right;
      ++state->bufferCount;
    }
    if (y > 0) {
      state->cl[state->bufferCount] = -right;
      state->cy[state->bufferCount] = y;
      state->cr[state->bufferCount] = -left;
      ++state->bufferCount;
    }
    if (left > 0 || left < right) {
      state->cl[state->bufferCount] = left == 0 ? 2 : left;
      state->cy[state->bufferCount] = -y;
      state->cr[state->bufferCount] = right;
      ++state->bufferCount;
    }
    state->cl[state->bufferCount] = -right;
    state->cy[state->bufferCount] = -y;
    state->cr[state->bufferCount] = -left;
    ++state->bufferCount;
  }
  --state->bufferCount;
  *outX0 = state->cl[state->bufferCount];
  *outY = state->cy[state->bufferCount];
  *outX1 = state->cr[state->bufferCount];
  return true;
}

// libImaging/Draw.c: ellipseNew() with fill set, which is what ImageDraw uses
// for a filled ellipse (including the disc a one-point stroke becomes).
void fillEllipse(QImage& image, int x0, int y0, int x1, int y1, uint8_t ink) {
  const int a = x1 - x0;
  const int b = y1 - y0;
  if (a < 0 || b < 0) {
    return;
  }
  EllipseState state;
  ellipseInit(&state, a, b, a + b);
  int32_t px0 = 0;
  int32_t py = 0;
  int32_t px1 = 0;
  while (ellipseNext(&state, &px0, &py, &px1)) {
    drawHLine(image, x0 + (px0 + a) / 2, y0 + (py + b) / 2, x0 + (px1 + a) / 2, ink);
  }
}

// libImaging/Draw.c: clip_node and clip_tree_do_clip(). A pie slice is the
// filled ellipse with every scanline segment clipped by a wedge of half-planes;
// the tree combines the two planes with AND (narrow wedge) or OR (wide one).
enum class ClipType { And, Or, Clip };

struct ClipNode {
  ClipType type = ClipType::Clip;
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  int left = -1;
  int right = -1;
};

struct ClipEvent {
  int32_t x = 0;
  int8_t type = 0;
};

std::vector<ClipEvent> clipTreeDoClip(const std::array<ClipNode, 7>& nodes, int root, int32_t x0, int32_t y, int32_t x1) {
  std::vector<ClipEvent> result;
  if (root < 0) {
    result.push_back({x0, 1});
    result.push_back({x1, -1});
    return result;
  }

  const ClipNode& node = nodes[static_cast<size_t>(root)];
  if (node.type == ClipType::Clip) {
    constexpr double kEps = 1e-9;
    const double a = node.a;
    const double b = node.b;
    const double c = node.c;
    if (std::fabs(a) < kEps) {
      if (b * y + c < -kEps) {
        x0 = 1;
        x1 = 0;
      }
    } else {
      const double ix = -(b * y + c) / a;
      if (a * x0 + b * y + c < kEps) {
        x0 = static_cast<int32_t>(std::lround(std::fmax(static_cast<double>(x0), ix)));
      }
      if (a * x1 + b * y + c < kEps) {
        x1 = static_cast<int32_t>(std::lround(std::fmin(static_cast<double>(x1), ix)));
      }
    }
    if (x0 <= x1) {
      result.push_back({x0, 1});
      result.push_back({x1, -1});
    }
    return result;
  }

  const std::vector<ClipEvent> left = clipTreeDoClip(nodes, node.left, x0, y, x1);
  const std::vector<ClipEvent> right = clipTreeDoClip(nodes, node.right, x0, y, x1);
  size_t li = 0;
  size_t ri = 0;
  int32_t openLeft = 0;
  int32_t openRight = 0;
  bool hasTail = false;
  int8_t tailType = 0;
  while (li < left.size() || ri < right.size()) {
    ClipEvent event;
    if (ri >= right.size() || (li < left.size() && (left[li].x < right[ri].x || (left[li].x == right[ri].x && left[li].type > right[ri].type)))) {
      event = left[li++];
      openLeft += event.type;
    } else {
      event = right[ri++];
      openRight += event.type;
    }
    // A start event opens an interval, an end event closes one: OR keeps the
    // union, AND keeps only what both sides cover at the same time.
    bool keep = false;
    if (event.type == 1) {
      const bool opens = !hasTail || tailType == -1;
      if (node.type == ClipType::Or) {
        keep = opens;
      } else {
        keep = opens && openLeft > 0 && openRight > 0;
      }
    } else if (node.type == ClipType::Or) {
      keep = openLeft == 0 && openRight == 0;
    } else {
      keep = hasTail && tailType == 1 && (openLeft == 0 || openRight == 0);
    }
    if (keep) {
      result.push_back(event);
      hasTail = true;
      tailType = event.type;
    }
  }
  return result;
}

// libImaging/Draw.c: normalize_angles().
void normalizeAngles(float* from, float* to) {
  if (*to - *from >= 360.0f) {
    *from = 0.0f;
    *to = 360.0f;
    return;
  }
  *from = static_cast<float>(std::fmod(*from < 0.0f ? 360.0 - std::fmod(-static_cast<double>(*from), 360.0) : static_cast<double>(*from), 360.0));
  *to = static_cast<float>(static_cast<double>(*from) + std::fmod(*to < *from ? 360.0 - std::fmod(static_cast<double>(*from) - *to, 360.0) : static_cast<double>(*to) - *from, 360.0));
}

// libImaging/Draw.c: pie_init() and clip_ellipse_next(), for a filled pie
// slice. ImageDraw.line draws one of these at every interior vertex of a
// polyline to fill the wedge the segment quads leave open.
void drawPieSlice(QImage& image, int x0, int y0, int x1, int y1, float from, float to, uint8_t ink) {
  normalizeAngles(&from, &to);
  if (from + 360.0f == to) {
    fillEllipse(image, x0, y0, x1, y1, ink);
    return;
  }
  if (from == to) {
    return;
  }

  const int a = x1 - x0;
  const int b = y1 - y0;
  if (a < 0 || b < 0) {
    return;
  }

  EllipseState state;
  ellipseInit(&state, a, b, a + b);

  const double xl = a * std::cos(from * kPi / 180.0);
  const double xr = a * std::cos(to * kPi / 180.0);
  const double yl = b * std::sin(from * kPi / 180.0);
  const double yr = b * std::sin(to * kPi / 180.0);

  std::array<ClipNode, 7> nodes{};
  nodes[0].type = ClipType::Clip;
  nodes[0].a = -yl;
  nodes[0].b = xl;
  nodes[1].type = ClipType::Clip;
  nodes[1].a = yr;
  nodes[1].b = -xr;
  nodes[2].type = to - from < 180.0f ? ClipType::And : ClipType::Or;
  nodes[2].left = 0;
  nodes[2].right = 1;
  int root = 2;
  // A wedge narrower than 90 degrees would otherwise be closed off behind the
  // origin, where the two clipped scanline halves still overlap: the spike
  // clipper is a third half-plane along the wedge's bisector that keeps that
  // side out. ink.py only draws pies for joints, and a joint that bends by
  // less than 90 degrees is the common case. Source: pie_init()'s "add one
  // more semiplane to avoid spikes" branch.
  if (to - from < 90.0f) {
    nodes[3].type = ClipType::Clip;
    nodes[3].a = (xl + xr) / 2.0;
    nodes[3].b = (yl + yr) / 2.0;
    nodes[4].type = ClipType::And;
    nodes[4].left = 2;
    nodes[4].right = 3;
    root = 4;
  }

  std::vector<ClipEvent> head;
  size_t cursor = 0;
  int32_t headY = 0;
  for (;;) {
    while (cursor >= head.size()) {
      int32_t sx0 = 0;
      int32_t sy = 0;
      int32_t sx1 = 0;
      if (!ellipseNext(&state, &sx0, &sy, &sx1)) {
        return;
      }
      head = clipTreeDoClip(nodes, root, sx0, sy, sx1);
      cursor = 0;
      headY = sy;
    }
    const int32_t left = head[cursor].x;
    const int32_t right = head[cursor + 1].x;
    cursor += 2;
    drawHLine(image, x0 + (left + a) / 2, y0 + (headY + b) / 2, x0 + (right + a) / 2, ink);
  }
}

// ImageDraw.py: coord_at_angle(), the helper line() uses to place the ends of
// the short polyline that covers the gap between a joint's wedge and the
// segments. The offset is truncated towards zero, not rounded.
Point jointGapEnd(const Point& point, double angle, int width) {
  constexpr double kDegToRad = kPi / 180.0;
  const double distance = width / 2.0 - 1.0;
  const double radians = (angle - 90.0) * kDegToRad;
  const double dx = distance * std::cos(radians);
  const double dy = distance * std::sin(radians);
  return {point.x + (dx > 0.0 ? std::floor(dx) : std::ceil(dx)), point.y + (dy > 0.0 ? std::floor(dy) : std::ceil(dy))};
}

// ImageDraw.py: line(). PIL draws every segment, then fills the wedge at each
// interior vertex with a pie slice when the pen is wide enough, and covers the
// remaining gap with a short three-point polyline of width 3. The thresholds
// (4 and 8) and the pie geometry are the installed Pillow's (12.3.0).
void drawLine(QImage& image, const std::vector<Point>& points, int width, uint8_t ink) {
  if (width == 1) {
    for (size_t i = 0; i + 1 < points.size(); ++i) {
      drawThinLine(image, static_cast<int>(points[i].x), static_cast<int>(points[i].y), static_cast<int>(points[i + 1].x), static_cast<int>(points[i + 1].y), ink);
    }
    if (!points.empty()) {
      drawPoint(image, static_cast<int>(points.back().x), static_cast<int>(points.back().y), ink);
    }
  } else {
    for (size_t i = 0; i + 1 < points.size(); ++i) {
      drawWideLine(image, static_cast<int>(points[i].x), static_cast<int>(points[i].y), static_cast<int>(points[i + 1].x), static_cast<int>(points[i + 1].y), width, ink);
    }
  }

  if (width <= 4) {
    return;
  }
  constexpr double kRadToDeg = 180.0 / kPi;
  for (size_t i = 1; i + 1 < points.size(); ++i) {
    const Point& previous = points[i - 1];
    const Point& point = points[i];
    const Point& next = points[i + 1];
    double angles[2] = {std::atan2(point.x - previous.x, previous.y - point.y) * kRadToDeg, std::atan2(next.x - point.x, point.y - next.y) * kRadToDeg};
    for (double& angle : angles) {
      angle = std::fmod(angle, 360.0);
      if (angle < 0.0) {
        angle += 360.0;
      }
    }
    if (angles[0] == angles[1]) {
      continue;
    }
    const bool flipped = (angles[1] > angles[0] && angles[1] - 180.0 > angles[0]) || (angles[1] < angles[0] && angles[1] + 180.0 > angles[0]);
    const double start = flipped ? angles[1] + 90.0 : angles[0] - 90.0;
    const double end = flipped ? angles[0] + 90.0 : angles[1] - 90.0;
    drawPieSlice(image, static_cast<int>(point.x - width / 2.0 + 1), static_cast<int>(point.y - width / 2.0 + 1), static_cast<int>(point.x + width / 2.0 - 1), static_cast<int>(point.y + width / 2.0 - 1), static_cast<float>(start - 90.0), static_cast<float>(end - 90.0), ink);

    if (width > 8) {
      const std::vector<Point> gap{jointGapEnd(point, flipped ? angles[0] + 90.0 : angles[0] - 90.0, width), point, jointGapEnd(point, flipped ? angles[1] + 90.0 : angles[1] - 90.0, width)};
      drawLine(image, gap, 3, ink);
    }
  }
}

// ink.py: render_strokes(). Strokes arrive in whatever box the pad drew them
// in; the drawing is centred on a size x size canvas and scaled so its longest
// side fills it. A width of 0 asks for the reference's pen, 5.5% of the span.
QImage renderStrokes(const Strokes& strokes, int size, int width) {
  if (size < 1) {
    return {};
  }
  QImage image{size, size, QImage::Format_Grayscale8};
  image.fill(255);

  double minX = 0.0;
  double maxX = 0.0;
  double minY = 0.0;
  double maxY = 0.0;
  bool any = false;
  for (const Stroke& stroke : strokes) {
    for (const Point& point : stroke) {
      if (!any) {
        minX = maxX = point.x;
        minY = maxY = point.y;
        any = true;
        continue;
      }
      minX = std::min(minX, point.x);
      maxX = std::max(maxX, point.x);
      minY = std::min(minY, point.y);
      maxY = std::max(maxY, point.y);
    }
  }
  if (!any) {
    return image;
  }

  const double span = std::max(std::max(maxX - minX, maxY - minY), 1e-6);
  const double scale = size / span;
  if (width <= 0) {
    // Python's round() is round-half-to-even, like std::nearbyint.
    width = std::max(2, static_cast<int>(std::nearbyint(span * scale * 0.055)));
  }
  const double centerX = (minX + maxX) / 2.0;
  const double centerY = (minY + maxY) / 2.0;

  for (const Stroke& stroke : strokes) {
    if (stroke.empty()) {
      // ImageDraw.line would raise IndexError on an empty coordinate list;
      // the reference never passes one.
      continue;
    }
    std::vector<Point> points;
    points.reserve(stroke.size());
    for (const Point& point : stroke) {
      points.push_back({(point.x - centerX) * scale + size / 2.0, (point.y - centerY) * scale + size / 2.0});
    }
    if (points.size() == 1) {
      // ink.py draws a one-point stroke as a filled disc of the pen width.
      const Point& point = points.front();
      fillEllipse(image, static_cast<int>(point.x - width / 2.0), static_cast<int>(point.y - width / 2.0), static_cast<int>(point.x + width / 2.0), static_cast<int>(point.y + width / 2.0), 0);
      continue;
    }
    drawLine(image, points, width, 0);
  }
  return image;
}

// ink.py: normalize(). The ink bounding box is cropped, scaled so its longest
// side reaches kCanvas - 8, pasted centred on a 255 canvas and inverted, so
// the result is 1 where the ink is and the antialiased edges survive as
// fractional values. The crop is not binarized: only the box is.
Ink normalizeGray(const QImage& gray, int threshold) {
  QImage source = gray.format() == QImage::Format_Grayscale8 ? gray : gray.convertToFormat(QImage::Format_Grayscale8);
  if (source.isNull()) {
    return Ink{};
  }

  int minX = source.width();
  int maxX = -1;
  int minY = source.height();
  int maxY = -1;
  for (int y = 0; y < source.height(); ++y) {
    const uint8_t* row = source.constScanLine(y);
    for (int x = 0; x < source.width(); ++x) {
      if (static_cast<int>(row[x]) < threshold) {
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
      }
    }
  }
  if (maxX < 0) {
    return Ink{};
  }

  const int cropWidth = maxX - minX + 1;
  const int cropHeight = maxY - minY + 1;
  const double scale = static_cast<double>(kCanvas - 8) / std::max(cropHeight, cropWidth);
  const int targetHeight = std::max(1, static_cast<int>(std::nearbyint(cropHeight * scale)));
  const int targetWidth = std::max(1, static_cast<int>(std::nearbyint(cropWidth * scale)));
  const QImage small = resampleTriangle(source.copy(minX, minY, cropWidth, cropHeight), targetWidth, targetHeight);

  QImage canvas{kCanvas, kCanvas, QImage::Format_Grayscale8};
  canvas.fill(255);
  const int offsetX = (kCanvas - targetWidth) / 2;
  const int offsetY = (kCanvas - targetHeight) / 2;
  for (int y = 0; y < targetHeight; ++y) {
    std::memcpy(canvas.scanLine(offsetY + y) + offsetX, small.constScanLine(y), static_cast<size_t>(targetWidth));
  }

  Ink ink;
  for (int i = 0; i < kCanvasPixels; ++i) {
    ink.value[static_cast<size_t>(i)] = (255.0f - static_cast<float>(canvas.constScanLine(i / kCanvas)[i % kCanvas])) / 255.0f;
  }
  return ink;
}

// ink.py: features() blurs the ink with ImageFilter.GaussianBlur(1.0), which
// is Pillow's fixed-point box cascade, not a real Gaussian: Image.gaussian_blur
// -> _imaging.c's gaussian_blur -> BoxBlur.c's ImagingGaussianBlur with three
// passes and _gaussian_blur_radius(1.0, 3) = floor((sqrt(12 / 3 + 1) - 1) / 2)
// plus a fractional part of exactly 0.25. The integer part is zero, so a pass
// is a weighted mix of the pixel and its two neighbours, with the image edge
// replicated, in 24-bit fixed point. The weights and the wrap-around are part
// of the reference kernel: the float division that produces the weight rounds
// 11184810.67 up to 11184811, and truncating it instead would shift the blurred
// ink by up to a level. A true separable Gaussian of the same sigma would come
// within 5e-3 per feature component, but not within the round-off the stored
// templates are compared at, so the cascade is reproduced exactly.
constexpr float kBoxBlurRadius = 0.25f; // _gaussian_blur_radius(1.0f, 3)
constexpr int kBoxBlurPasses = 3;
constexpr int kBoxBlurIntegerRadius = static_cast<int>(kBoxBlurRadius);
constexpr uint32_t kBoxBlurWeight = static_cast<uint32_t>(static_cast<float>(1 << 24) / (kBoxBlurRadius * 2.0f + 1.0f));
constexpr uint32_t kBoxBlurEdgeWeight = ((1 << 24) - (kBoxBlurIntegerRadius * 2 + 1) * kBoxBlurWeight) / 2;

// libImaging/BoxBlur.c: ImagingLineBoxBlur8, for the zero-radius case the
// sigma 1.0 filter lands on. The sliding window is then the pixel itself, and
// BoxBlur.c's edge handling reduces to clamping the two neighbour reads.
void boxBlurLine(const uint8_t* in, uint8_t* out) {
  for (int x = 0; x < kCanvas; ++x) {
    const uint8_t left = in[x > 0 ? x - 1 : 0];
    const uint8_t right = in[x + 1 < kCanvas ? x + 1 : kCanvas - 1];
    const uint32_t window = in[x];
    const uint32_t bulk = window * kBoxBlurWeight + static_cast<uint32_t>(left + right) * kBoxBlurEdgeWeight;
    out[x] = static_cast<uint8_t>((bulk + (1u << 23)) >> 24);
  }
}

// libImaging/BoxBlur.c: ImagingBoxBlur - three passes along the rows, then
// three down the columns, which is what the transpose in that function amounts
// to. Every pass reads whole lines of the previous pass's output, because the
// in-place case is written through a line buffer.
std::array<uint8_t, kCanvasPixels> boxBlurImage(const std::array<uint8_t, kCanvasPixels>& source) {
  std::array<uint8_t, kCanvasPixels> blurred = source;
  std::array<uint8_t, kCanvasPixels> scratch{};
  for (int pass = 0; pass < kBoxBlurPasses; ++pass) {
    for (int y = 0; y < kCanvas; ++y) {
      boxBlurLine(blurred.data() + static_cast<size_t>(y) * kCanvas, scratch.data() + static_cast<size_t>(y) * kCanvas);
    }
    blurred = scratch;
  }
  std::array<uint8_t, kCanvas> columnIn{};
  std::array<uint8_t, kCanvas> columnOut{};
  for (int pass = 0; pass < kBoxBlurPasses; ++pass) {
    for (int x = 0; x < kCanvas; ++x) {
      for (int y = 0; y < kCanvas; ++y) {
        columnIn[static_cast<size_t>(y)] = blurred[static_cast<size_t>(y) * kCanvas + x];
      }
      boxBlurLine(columnIn.data(), columnOut.data());
      for (int y = 0; y < kCanvas; ++y) {
        blurred[static_cast<size_t>(y) * kCanvas + x] = columnOut[static_cast<size_t>(y)];
      }
    }
  }
  return blurred;
}

// thin.py: skeletonize(). Zhang-Suen thinning, both sub-iterations per round,
// each round removing every pixel its sub-iteration marks so the marks are
// computed against the image as it was at the start of that sub-iteration.
// Padding by one pixel lets the neighbour views be plain offsets.
void zhangSuen(const Mask& mask, Mask& out) {
  constexpr int kWidth = kCanvas + 2;
  std::array<uint8_t, kWidth * kWidth> padded{};
  for (int y = 0; y < kCanvas; ++y) {
    for (int x = 0; x < kCanvas; ++x) {
      padded[static_cast<size_t>(y + 1) * kWidth + (x + 1)] = mask.at(y, x) ? 1 : 0;
    }
  }

  std::array<uint8_t, kCanvasPixels> remove{};
  bool changed = true;
  while (changed) {
    changed = false;
    for (int step = 0; step < 2; ++step) {
      remove.fill(0);
      bool marked = false;
      for (int y = 1; y <= kCanvas; ++y) {
        for (int x = 1; x <= kCanvas; ++x) {
          const size_t centre = static_cast<size_t>(y) * kWidth + x;
          if (padded[centre] != 1) {
            continue;
          }
          const uint8_t neighbours[9] = {
              padded[centre - kWidth],     // p2, north
              padded[centre - kWidth + 1], // p3
              padded[centre + 1],          // p4, east
              padded[centre + kWidth + 1], // p5
              padded[centre + kWidth],     // p6, south
              padded[centre + kWidth - 1], // p7
              padded[centre - 1],          // p8, west
              padded[centre - kWidth - 1], // p9
              padded[centre - kWidth],     // p2 again, closing the ring
          };
          int count = 0;
          for (int i = 0; i < 8; ++i) {
            count += neighbours[i];
          }
          int transitions = 0;
          for (int i = 0; i < 8; ++i) {
            transitions += (neighbours[i] == 0 && neighbours[i + 1] == 1) ? 1 : 0;
          }
          const bool ends = step == 0 ? (neighbours[0] * neighbours[2] * neighbours[4] == 0 && neighbours[2] * neighbours[4] * neighbours[6] == 0) : (neighbours[0] * neighbours[2] * neighbours[6] == 0 && neighbours[0] * neighbours[4] * neighbours[6] == 0);
          if (count >= 2 && count <= 6 && transitions == 1 && ends) {
            remove[static_cast<size_t>(y - 1) * kCanvas + (x - 1)] = 1;
            marked = true;
          }
        }
      }
      if (marked) {
        for (int y = 1; y <= kCanvas; ++y) {
          for (int x = 1; x <= kCanvas; ++x) {
            if (remove[static_cast<size_t>(y - 1) * kCanvas + (x - 1)]) {
              padded[static_cast<size_t>(y) * kWidth + x] = 0;
            }
          }
        }
        changed = true;
      }
    }
  }

  out = Mask{};
  for (int y = 0; y < kCanvas; ++y) {
    for (int x = 0; x < kCanvas; ++x) {
      out.set(y, x, padded[static_cast<size_t>(y + 1) * kWidth + (x + 1)] != 0);
    }
  }
}

// The 1D squared-distance transform of Felzenszwalb and Huttenlocher: the
// lower envelope of the parabolas rooted at each sample. Exact on the integer
// grid, which is what scipy's distance_transform_edt computes too.
constexpr double kFar = 1e18;
constexpr double kEnvelopeLimit = 1e30;

void transform1d(const std::array<double, kCanvas>& f, std::array<double, kCanvas>& d, std::array<int, kCanvas>& vertex, std::array<double, kCanvas + 1>& boundary) {
  int k = 0;
  vertex[0] = 0;
  boundary[0] = -kEnvelopeLimit;
  boundary[1] = kEnvelopeLimit;
  for (int q = 1; q < kCanvas; ++q) {
    double s = ((f[q] + static_cast<double>(q) * q) - (f[vertex[k]] + static_cast<double>(vertex[k]) * vertex[k])) / (2 * q - 2 * vertex[k]);
    while (s <= boundary[k]) {
      --k;
      s = ((f[q] + static_cast<double>(q) * q) - (f[vertex[k]] + static_cast<double>(vertex[k]) * vertex[k])) / (2 * q - 2 * vertex[k]);
    }
    ++k;
    vertex[k] = q;
    boundary[k] = s;
    boundary[k + 1] = kEnvelopeLimit;
  }
  k = 0;
  for (int q = 0; q < kCanvas; ++q) {
    while (boundary[k + 1] < q) {
      ++k;
    }
    const double delta = q - vertex[k];
    d[q] = delta * delta + f[vertex[k]];
  }
}

// thin.py: trace_paths(). Skeleton pixels are (y, x); a node is any pixel whose
// neighbour count is not two. Every node walks each of its unused neighbours
// until it reaches another node, then the pixels no walk covered are walked as
// closed loops. The neighbour order below is the reference's, because it
// decides which branch a walk takes at a fork.
int neighboursOf(const std::array<uint8_t, kCanvasPixels>& pixels, int y, int x, int* out) {
  int count = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dy == 0 && dx == 0) {
        continue;
      }
      const int ny = y + dy;
      const int nx = x + dx;
      if (ny < 0 || ny >= kCanvas || nx < 0 || nx >= kCanvas) {
        continue;
      }
      const int index = ny * kCanvas + nx;
      if (pixels[static_cast<size_t>(index)]) {
        out[count++] = index;
      }
    }
  }
  return count;
}

int edgeBit(int from, int to) {
  const int dy = to / kCanvas - from / kCanvas;
  const int dx = to % kCanvas - from % kCanvas;
  return (dy + 1) * 3 + (dx + 1);
}

// thin.py: simplify(). Despite the name in its docstring this is not
// Douglas-Peucker: it keeps a point when it lies further than epsilon from the
// chord between the last kept point and the end of the path, and it never
// revisits the points it skips. Ported as written, in (x, y) order, so the
// point counts match the reference.
Stroke simplifyPath(const std::vector<int>& path, double epsilon) {
  Stroke points;
  points.reserve(path.size());
  for (const int index : path) {
    points.push_back({static_cast<double>(index % kCanvas), static_cast<double>(index / kCanvas)});
  }
  if (points.size() <= 2) {
    return points;
  }
  Stroke keep{points.front()};
  for (size_t i = 1; i + 1 < points.size(); ++i) {
    const Point& a = keep.back();
    const Point& b = points[i];
    const Point& c = points.back();
    const double dx = c.x - a.x;
    const double dy = c.y - a.y;
    double norm = std::sqrt(dx * dx + dy * dy);
    if (norm == 0.0) {
      norm = 1e-6;
    }
    if (std::fabs(dy * (b.x - a.x) - dx * (b.y - a.y)) / norm > epsilon) {
      keep.push_back(b);
    }
  }
  keep.push_back(points.back());
  return keep;
}

} // namespace

bool Mask::any() const {
  return std::any_of(value.begin(), value.end(), [](uint8_t pixel) {
    return pixel != 0;
  });
}

int Mask::count() const {
  return static_cast<int>(std::count_if(value.begin(), value.end(), [](uint8_t pixel) {
    return pixel != 0;
  }));
}

Mask inkMaskFromGray(const QImage& gray, int threshold) {
  const QImage source = gray.format() == QImage::Format_Grayscale8 ? gray : gray.convertToFormat(QImage::Format_Grayscale8);
  Mask mask;
  for (int y = 0; y < source.height(); ++y) {
    const uint8_t* row = source.constScanLine(y);
    for (int x = 0; x < source.width(); ++x) {
      mask.set(y, x, static_cast<int>(row[x]) < threshold);
    }
  }
  return mask;
}

Ink inkFromGray(const QImage& gray, int threshold) {
  return normalizeGray(gray, threshold);
}

Ink inkFromStrokes(const Strokes& strokes, int size, int width) {
  return normalizeGray(renderStrokes(strokes, size, width), kInkThreshold);
}

Ink normalizeInk(const Ink& ink) {
  QImage gray{kCanvas, kCanvas, QImage::Format_Grayscale8};
  for (int y = 0; y < kCanvas; ++y) {
    uint8_t* row = gray.scanLine(y);
    for (int x = 0; x < kCanvas; ++x) {
      // ink.py wraps this in (255.0 * (1.0 - ink)).astype(np.uint8), a float32
      // multiply and a truncating cast.
      row[x] = static_cast<uint8_t>(255.0f * (1.0f - ink.at(y, x)));
    }
  }
  return normalizeGray(gray, static_cast<int>(0.15 * 255));
}

void inkFeatures(const Ink& ink, float* out) {
  // Pillow blurs an 8-bit copy of the ink and the gradients below see whole
  // levels.
  std::array<uint8_t, kCanvasPixels> quantized{};
  for (int i = 0; i < kCanvasPixels; ++i) {
    quantized[static_cast<size_t>(i)] = static_cast<uint8_t>(ink.value[static_cast<size_t>(i)] * 255.0f);
  }
  const std::array<uint8_t, kCanvasPixels> blurredByte = boxBlurImage(quantized);

  std::array<float, kCanvasPixels> blurred{};
  for (int i = 0; i < kCanvasPixels; ++i) {
    blurred[static_cast<size_t>(i)] = static_cast<float>(blurredByte[static_cast<size_t>(i)]) / 255.0f;
  }

  constexpr int kCell = kCanvas / kCells;
  std::array<std::array<float, kCells * kCells>, 5> pooled{};
  for (int y = 1; y + 1 < kCanvas; ++y) {
    for (int x = 1; x + 1 < kCanvas; ++x) {
      const size_t index = static_cast<size_t>(y) * kCanvas + x;
      const float gx = blurred[index + 1] - blurred[index - 1];
      const float gy = blurred[static_cast<size_t>(y + 1) * kCanvas + x] - blurred[static_cast<size_t>(y - 1) * kCanvas + x];
      const float magnitude = std::hypot(gx, gy);
      float angle = std::fmod(std::atan2(gy, gx) + static_cast<float>(kPi), static_cast<float>(kPi));
      if (angle < 0.0f) {
        angle += static_cast<float>(kPi);
      }
      int bin = -1;
      for (int k = 0; k < 3; ++k) {
        const double low = k * kPi / 4.0;
        const double high = (k + 1) * kPi / 4.0;
        if (static_cast<double>(angle) >= low && static_cast<double>(angle) < high) {
          bin = k;
          break;
        }
      }
      const size_t cell = static_cast<size_t>((y / kCell) * kCells + x / kCell);
      if (bin >= 0) {
        pooled[static_cast<size_t>(bin)][cell] += magnitude;
      }
      // ink.py builds the fourth bin as (angle >= 3 * pi / 4) | (angle < pi),
      // and angle is always below pi, so that bin collects the full magnitude
      // rather than the wrap-around direction. Reproduced as it stands: the
      // templates were built with it.
      pooled[3][cell] += magnitude;
    }
  }
  for (int y = 0; y < kCanvas; ++y) {
    for (int x = 0; x < kCanvas; ++x) {
      pooled[4][static_cast<size_t>((y / kCell) * kCells + x / kCell)] += blurred[static_cast<size_t>(y) * kCanvas + x];
    }
  }

  double norm = 0.0;
  for (int i = 0; i < kFeatureDim; ++i) {
    const float value = pooled[static_cast<size_t>(i) / (kCells * kCells)][static_cast<size_t>(i) % (kCells * kCells)];
    out[i] = value;
    norm += static_cast<double>(value) * value;
  }
  if (norm > 0.0) {
    const double inverse = 1.0 / std::sqrt(norm);
    for (int i = 0; i < kFeatureDim; ++i) {
      out[i] = static_cast<float>(out[i] * inverse);
    }
  }
}

void skeletonizeInk(const Ink& ink, Mask& out, float threshold) {
  Mask mask;
  for (int i = 0; i < kCanvasPixels; ++i) {
    mask.value[static_cast<size_t>(i)] = ink.value[static_cast<size_t>(i)] > threshold ? 1 : 0;
  }
  // recognize.py: thin() skips the thinning when nothing is inked.
  if (!mask.any()) {
    out = mask;
    return;
  }
  skeletonizeMask(mask, out);
}

void skeletonizeMask(const Mask& mask, Mask& out) {
  zhangSuen(mask, out);
}

void distanceTransform(const Mask& mask, uint8_t* out) {
  std::array<double, kCanvasPixels> squared{};
  std::array<double, kCanvas> f{};
  std::array<double, kCanvas> d{};
  std::array<int, kCanvas> vertex{};
  std::array<double, kCanvas + 1> boundary{};

  for (int y = 0; y < kCanvas; ++y) {
    for (int x = 0; x < kCanvas; ++x) {
      f[static_cast<size_t>(x)] = mask.at(y, x) ? 0.0 : kFar;
    }
    transform1d(f, d, vertex, boundary);
    for (int x = 0; x < kCanvas; ++x) {
      squared[static_cast<size_t>(y) * kCanvas + x] = d[static_cast<size_t>(x)];
    }
  }
  for (int x = 0; x < kCanvas; ++x) {
    for (int y = 0; y < kCanvas; ++y) {
      f[static_cast<size_t>(y)] = squared[static_cast<size_t>(y) * kCanvas + x];
    }
    transform1d(f, d, vertex, boundary);
    for (int y = 0; y < kCanvas; ++y) {
      squared[static_cast<size_t>(y) * kCanvas + x] = d[static_cast<size_t>(y)];
    }
  }

  for (int i = 0; i < kCanvasPixels; ++i) {
    // recognize.py stores the template distances as uint8, which truncates:
    // sqrt(2) is 1, not 2.
    out[i] = static_cast<uint8_t>(std::min(255.0, std::sqrt(squared[static_cast<size_t>(i)])));
  }
}

void dilateMask(const Mask& mask, Mask& out, int radius) {
  // recognize.py: _thickened() grows the centre line with the full 3x3
  // structuring element, `radius` times over.
  out = mask;
  for (int pass = 0; pass < radius; ++pass) {
    Mask grown;
    for (int y = 0; y < kCanvas; ++y) {
      for (int x = 0; x < kCanvas; ++x) {
        bool on = false;
        for (int dy = -1; dy <= 1 && !on; ++dy) {
          const int ny = y + dy;
          if (ny < 0 || ny >= kCanvas) {
            continue;
          }
          for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= kCanvas) {
              continue;
            }
            if (out.at(ny, nx)) {
              on = true;
              break;
            }
          }
        }
        grown.set(y, x, on);
      }
    }
    out = grown;
  }
}

Strokes strokesFromSkeleton(const Mask& skeleton, int minPoints) {
  std::array<uint8_t, kCanvasPixels> pixels{};
  std::array<uint8_t, kCanvasPixels> nodes{};
  std::array<uint8_t, kCanvasPixels> degrees{};
  std::array<uint8_t, kCanvasPixels> used{};
  std::array<uint8_t, kCanvasPixels> seen{};
  for (int index = 0; index < kCanvasPixels; ++index) {
    pixels[static_cast<size_t>(index)] = skeleton.value[static_cast<size_t>(index)] != 0 ? 1 : 0;
  }
  for (int y = 0; y < kCanvas; ++y) {
    for (int x = 0; x < kCanvas; ++x) {
      const int index = y * kCanvas + x;
      if (!pixels[static_cast<size_t>(index)]) {
        continue;
      }
      int neighbours[8];
      const int count = neighboursOf(pixels, y, x, neighbours);
      degrees[static_cast<size_t>(index)] = static_cast<uint8_t>(count);
      nodes[static_cast<size_t>(index)] = count != 2 ? 1 : 0;
    }
  }

  std::vector<std::vector<int>> paths;
  const auto walk = [&](int start, int next) {
    std::vector<int> path{start, next};
    int previous = start;
    int current = next;
    while (!nodes[static_cast<size_t>(current)]) {
      int neighbours[8];
      const int count = neighboursOf(pixels, current / kCanvas, current % kCanvas, neighbours);
      int step = -1;
      for (int i = 0; i < count; ++i) {
        if (neighbours[i] != previous) {
          step = neighbours[i];
          break;
        }
      }
      if (step < 0) {
        break;
      }
      previous = current;
      current = step;
      path.push_back(current);
      if (path.size() > 4096) {
        break;
      }
    }
    return path;
  };
  const auto markPath = [&](const std::vector<int>& path, bool markSeen) {
    for (size_t i = 0; i + 1 < path.size(); ++i) {
      used[static_cast<size_t>(path[i])] |= static_cast<uint8_t>(1 << edgeBit(path[i], path[i + 1]));
      used[static_cast<size_t>(path[i + 1])] |= static_cast<uint8_t>(1 << edgeBit(path[i + 1], path[i]));
    }
    if (markSeen) {
      for (const int index : path) {
        seen[static_cast<size_t>(index)] = 1;
      }
    }
  };

  for (int index = 0; index < kCanvasPixels; ++index) {
    if (!nodes[static_cast<size_t>(index)]) {
      continue;
    }
    int neighbours[8];
    const int count = neighboursOf(pixels, index / kCanvas, index % kCanvas, neighbours);
    for (int i = 0; i < count; ++i) {
      const int next = neighbours[i];
      if (used[static_cast<size_t>(index)] & (1 << edgeBit(index, next))) {
        continue;
      }
      paths.push_back(walk(index, next));
      markPath(paths.back(), true);
    }
  }

  // Closed loops have no node to start from, so the pixels no walk covered are
  // walked from whichever of them comes first. The reference iterates the set
  // it computed before this loop, so a loop is entered once per pixel of it.
  for (int index = 0; index < kCanvasPixels; ++index) {
    if (!pixels[static_cast<size_t>(index)] || seen[static_cast<size_t>(index)] || degrees[static_cast<size_t>(index)] != 2) {
      continue;
    }
    int neighbours[8];
    neighboursOf(pixels, index / kCanvas, index % kCanvas, neighbours);
    paths.push_back(walk(index, neighbours[0]));
    markPath(paths.back(), false);
  }

  Strokes strokes;
  for (const std::vector<int>& path : paths) {
    if (static_cast<int>(path.size()) < minPoints) {
      continue;
    }
    strokes.push_back(simplifyPath(path, 2.5));
  }
  return strokes;
}

} // namespace hanzi
