#include "ImageResample.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

struct TriangleKernel {
  std::vector<int> first;
  std::vector<int> count;
  std::vector<int> offset;
  std::vector<float> weight;
};

TriangleKernel buildTriangleKernel(int source, int target) {
  TriangleKernel kernel;
  const double scale = static_cast<double>(source) / target;
  const double filterScale = std::max(1.0, scale);
  const double support = filterScale;

  for (int index = 0; index < target; ++index) {
    const double center = (index + 0.5) * scale;
    const int first = std::max(0, static_cast<int>(std::ceil(center - support - 0.5)));
    const int last = std::min(source - 1, static_cast<int>(std::floor(center + support - 0.5)));

    kernel.offset.push_back(static_cast<int>(kernel.weight.size()));
    kernel.first.push_back(first);
    kernel.count.push_back(last - first + 1);

    double total = 0.0;
    for (int tap = first; tap <= last; ++tap) {
      const double distance = ((tap + 0.5) - center) / filterScale;
      const double value = std::max(0.0, 1.0 - std::abs(distance));
      kernel.weight.push_back(static_cast<float>(value));
      total += value;
    }
    if (total > 0.0) {
      for (size_t slot = kernel.offset.back(); slot < kernel.weight.size(); ++slot) {
        kernel.weight[slot] = static_cast<float>(kernel.weight[slot] / total);
      }
    }
  }

  return kernel;
}

} // namespace

QImage resampleTriangle(const QImage& source, int targetWidth, int targetHeight) {
  if (source.isNull() || targetWidth < 1 || targetHeight < 1) {
    return {};
  }

  const TriangleKernel horizontal = buildTriangleKernel(source.width(), targetWidth);
  const TriangleKernel vertical = buildTriangleKernel(source.height(), targetHeight);

  std::vector<float> intermediate(static_cast<size_t>(targetWidth) * source.height());
  for (int y = 0; y < source.height(); ++y) {
    const uint8_t* row = source.constScanLine(y);
    float* row_out = intermediate.data() + static_cast<size_t>(y) * targetWidth;
    for (int x = 0; x < targetWidth; ++x) {
      float sum = 0.0f;
      const int offset = horizontal.offset[x];
      for (int tap = 0; tap < horizontal.count[x]; ++tap) {
        sum += row[horizontal.first[x] + tap] * horizontal.weight[offset + tap];
      }
      row_out[x] = sum;
    }
  }

  QImage result{targetWidth, targetHeight, QImage::Format_Grayscale8};
  for (int y = 0; y < targetHeight; ++y) {
    uint8_t* row_out = result.scanLine(y);
    const int offset = vertical.offset[y];
    for (int x = 0; x < targetWidth; ++x) {
      float sum = 0.0f;
      for (int tap = 0; tap < vertical.count[y]; ++tap) {
        sum += intermediate[static_cast<size_t>(vertical.first[y] + tap) * targetWidth + x] * vertical.weight[offset + tap];
      }
      row_out[x] = static_cast<uint8_t>(std::clamp(sum + 0.5f, 0.0f, 255.0f));
    }
  }

  return result;
}
