#pragma once

#include <array>
#include <cstdint>
#include <vector>

// Handwriting recognition for Hanzi, rare characters included. Three local
// recognizers vote, each on a different view of the drawing:
//
//   trajectory  the pen trajectory itself, against the 9,574 Make Me a Hanzi
//               medians; exact and fast, but its alphabet is that dictionary.
//   field       the ink against templates of every character of a charset
//               (GBK by default, 20,902), compared as thinned centre lines so
//               pen width does not matter.
//   composed    the parts of the drawing, then a lookup by Unicode ideographic
//               description: 谞 = ⿰讠胥. The parts are much easier than the
//               whole, and the lookup keeps look-alikes such as 谓 ⿰讠胃 apart.
//
// This is a port of ~/computing/lib/hanzi-handwriting, which has the
// measurements and the reasoning; every declaration here names the reference
// file it comes from so the two stay in step.
namespace hanzi {

// ink.py: N and CELLS. The normalized bitmap every route works on, and the
// pooling grid of the directional features.
constexpr int kCanvas = 128;
constexpr int kCells = 16;
constexpr int kCanvasPixels = kCanvas * kCanvas;
constexpr int kFeatureDim = 4 * kCells * kCells + kCells * kCells;

// A drawing: [[(x, y), ...], ...], y growing downwards, in whatever box the pad
// drew it. The reference's trajectories have this shape.
struct Point {
  double x = 0.0;
  double y = 0.0;
};

using Stroke = std::vector<Point>;
using Strokes = std::vector<Stroke>;

// ink.py: normalize() - kCanvas x kCanvas floats in [0, 1], 1 where the ink is.
struct Ink {
  std::array<float, kCanvasPixels> value{};

  float* data() {
    return value.data();
  }

  const float* data() const {
    return value.data();
  }

  float& at(int y, int x) {
    return value[static_cast<size_t>(y) * kCanvas + x];
  }

  float at(int y, int x) const {
    return value[static_cast<size_t>(y) * kCanvas + x];
  }

  float* row(int y) {
    return value.data() + static_cast<size_t>(y) * kCanvas;
  }

  const float* row(int y) const {
    return value.data() + static_cast<size_t>(y) * kCanvas;
  }
};

// A centre-line mask, one byte per pixel (the reference uses a bool array).
struct Mask {
  std::array<uint8_t, kCanvasPixels> value{};

  uint8_t* data() {
    return value.data();
  }

  const uint8_t* data() const {
    return value.data();
  }

  bool at(int y, int x) const {
    return value[static_cast<size_t>(y) * kCanvas + x] != 0;
  }

  void set(int y, int x, bool on) {
    value[static_cast<size_t>(y) * kCanvas + x] = on ? 1 : 0;
  }

  bool any() const;
  int count() const;
};

} // namespace hanzi
