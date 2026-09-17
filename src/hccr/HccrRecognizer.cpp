#include "hccr/HccrRecognizer.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace {

constexpr int kCanvas = 64;
constexpr int kContent = 56;
constexpr uint8_t kForegroundThreshold = 220;
constexpr int kStemChannels = 32;
constexpr int kHeadChannels = 576;
constexpr int kClasses = 3755;

// Separable triangle-filter resample, mirroring the reference pipeline (PIL's
// BILINEAR): the kernel's support grows with the reduction factor. Plain box
// averaging over-blurs thin strokes and measured ~20 points of top-1 worse.
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

QImage resampleTriangle(const QImage& source, int targetWidth, int targetHeight) {
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

// Must stay in step with BLOCKS in tools/hccr/hccr_model.py. verify_gguf.py
// checks the file's metadata against that table, and the shape assertions in
// load() catch any drift between the two.
struct BlockSpec {
  int expand;
  int channels;
  int stride;
};

constexpr BlockSpec kBlocks[] = {
  {1, 16, 1},
  {6, 24, 2}, {6, 24, 1},
  {6, 32, 2}, {6, 32, 1}, {6, 32, 1},
  {6, 64, 2}, {6, 64, 1}, {6, 64, 1}, {6, 64, 1},
  {6, 96, 1}, {6, 96, 1}, {6, 96, 1},
  {6, 160, 2}, {6, 160, 1}, {6, 160, 1},
  {6, 320, 1},
};
constexpr int kBlockCount = sizeof(kBlocks) / sizeof(kBlocks[0]);

bool shapeMatches(const ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
  return tensor->ne[0] == ne0 && tensor->ne[1] == ne1 && tensor->ne[2] == ne2 && tensor->ne[3] == ne3;
}

// ggml broadcasts over trailing dimensions, so a (C,) bias has to be given a
// (1, 1, C, 1) shape before it can be added to a (W, H, C, N) feature map.
ggml_tensor* biasAdd(ggml_context* ctx, ggml_tensor* input, ggml_tensor* bias, int64_t channels) {
  return ggml_add(ctx, input, ggml_reshape_4d(ctx, bias, 1, 1, channels, 1));
}

ggml_tensor* pointwise(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias, int64_t channels) {
  return biasAdd(ctx, ggml_conv_2d(ctx, weight, input, 1, 1, 0, 0, 1, 1), bias, channels);
}

}  // namespace

struct HccrRecognizer::Impl {
  ggml_backend_t backend = nullptr;
  gguf_context* gguf = nullptr;
  ggml_context* meta = nullptr;
  ggml_context* weights = nullptr;
  ggml_backend_buffer_t weightsBuffer = nullptr;
  ggml_context* graph = nullptr;
  ggml_backend_buffer_t graphBuffer = nullptr;
  ggml_cgraph* forward = nullptr;

  std::unordered_map<std::string, ggml_tensor*> tensors;
  std::vector<std::string> charset;

  ggml_tensor* input = nullptr;
  ggml_tensor* output = nullptr;

  double lastMilliseconds = 0.0;
  int threads = 4;

  ~Impl() {
    if (graphBuffer) {
      ggml_backend_buffer_free(graphBuffer);
    }
    if (graph) {
      ggml_free(graph);
    }
    if (weightsBuffer) {
      ggml_backend_buffer_free(weightsBuffer);
    }
    if (weights) {
      ggml_free(weights);
    }
    if (meta) {
      ggml_free(meta);
    }
    if (gguf) {
      gguf_free(gguf);
    }
    if (backend) {
      ggml_backend_free(backend);
    }
  }

  ggml_tensor* find(const char* name) const {
    const auto found = tensors.find(name);
    return found == tensors.end() ? nullptr : found->second;
  }
};

HccrRecognizer::HccrRecognizer() : _impl{std::make_unique<Impl>()} {
}

HccrRecognizer::~HccrRecognizer() = default;

int HccrRecognizer::classCount() const {
  return static_cast<int>(_impl->charset.size());
}

double HccrRecognizer::lastInferenceMilliseconds() const {
  return _impl->lastMilliseconds;
}

// -------------------------------------------------------------------- loading

std::unique_ptr<HccrRecognizer> HccrRecognizer::load(const std::string& modelPath, std::string* error) {
  auto fail = [error](const std::string& message) {
    if (error) {
      *error = message;
    }
    return std::unique_ptr<HccrRecognizer>{};
  };

  std::unique_ptr<HccrRecognizer> recognizer{new HccrRecognizer()};
  Impl& impl = *recognizer->_impl;

  impl.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
  if (!impl.backend) {
    return fail("could not initialize the ggml CPU backend");
  }
  const unsigned hardware = std::thread::hardware_concurrency();
  impl.threads = static_cast<int>(std::max(1u, std::min(4u, hardware ? hardware : 1u)));
  if (ggml_backend_is_cpu(impl.backend)) {
    ggml_backend_cpu_set_n_threads(impl.backend, impl.threads);
  }

  // The meta context holds the tensors as they are stored in the file, pointing
  // straight at the gguf data buffer (the type conversion happens below).
  ggml_init_params metaParams = {512 * 1024, nullptr, /*.no_alloc=*/true};
  impl.meta = ggml_init(metaParams);
  gguf_init_params ggufParams = {/*.no_alloc=*/false, /*.ctx=*/&impl.meta};
  impl.gguf = gguf_init_from_file(modelPath.c_str(), ggufParams);
  if (!impl.gguf) {
    return fail("could not open the model file: " + modelPath);
  }

  const int64_t charsetKey = gguf_find_key(impl.gguf, "hccr.charset");
  if (charsetKey < 0) {
    return fail("model file has no hccr.charset metadata (not an HCCR model?)");
  }
  const size_t classCount = gguf_get_arr_n(impl.gguf, charsetKey);
  if (classCount != kClasses) {
    return fail("model file declares " + std::to_string(classCount) + " classes, expected " + std::to_string(kClasses));
  }
  impl.charset.reserve(classCount);
  for (size_t i = 0; i < classCount; ++i) {
    impl.charset.emplace_back(gguf_get_arr_str(impl.gguf, charsetKey, i));
  }

  struct Pending {
    ggml_tensor* destination;
    ggml_tensor* source;
  };
  std::vector<Pending> pending;

  // Weights stay float16 exactly as stored: ggml_conv_2d_dw takes the kernel as
  // src0 against a float16 im2col, so a float32 kernel trips an assert in the
  // CPU backend. Biases stay float32 because they are added to float32
  // activations.
  auto declare = [&](const std::string& name, ggml_type expected, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const ggml_tensor* source = ggml_get_tensor(impl.meta, name.c_str());
    if (!source) {
      throw std::runtime_error("model file is missing tensor " + name);
    }
    if (source->type != expected) {
      throw std::runtime_error("tensor " + name + " has type " + std::to_string(source->type) + ", expected " + std::to_string(expected));
    }
    if (!shapeMatches(source, ne0, ne1, ne2, ne3)) {
      throw std::runtime_error("tensor " + name + " has shape (" + std::to_string(source->ne[0]) + "," + std::to_string(source->ne[1]) + "," +
                               std::to_string(source->ne[2]) + "," + std::to_string(source->ne[3]) + "), expected (" + std::to_string(ne0) + "," +
                               std::to_string(ne1) + "," + std::to_string(ne2) + "," + std::to_string(ne3) + ")");
    }
    ggml_tensor* destination = ggml_new_tensor_4d(impl.weights, expected, ne0, ne1, ne2, ne3);
    ggml_set_name(destination, name.c_str());
    impl.tensors.emplace(name, destination);
    pending.push_back({destination, const_cast<ggml_tensor*>(source)});
  };

  auto declareWeight = [&](const std::string& name, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    declare(name, GGML_TYPE_F16, ne0, ne1, ne2, ne3);
  };

  auto declareBias = [&](const std::string& name, int64_t ne0) {
    declare(name, GGML_TYPE_F32, ne0, 1, 1, 1);
  };

  ggml_init_params weightsParams = {256 * 1024, nullptr, /*.no_alloc=*/true};
  impl.weights = ggml_init(weightsParams);

  try {
    declareWeight("stem.weight", 3, 3, 1, kStemChannels);
    declareBias("stem.bias", kStemChannels);

    int64_t inChannels = kStemChannels;
    for (int index = 0; index < kBlockCount; ++index) {
      const BlockSpec& spec = kBlocks[index];
      const int64_t hidden = inChannels * spec.expand;
      const std::string prefix = "blk." + std::to_string(index) + ".";

      if (spec.expand != 1) {
        declareWeight(prefix + "expand.weight", 1, 1, inChannels, hidden);
        declareBias(prefix + "expand.bias", hidden);
      }
      declareWeight(prefix + "dw.weight", 3, 3, 1, hidden);
      declareBias(prefix + "dw.bias", hidden);
      declareWeight(prefix + "project.weight", 1, 1, hidden, spec.channels);
      declareBias(prefix + "project.bias", spec.channels);

      inChannels = spec.channels;
    }

    declareWeight("head.weight", 1, 1, 320, kHeadChannels);
    declareBias("head.bias", kHeadChannels);
    declareWeight("classifier.weight", kHeadChannels, kClasses, 1, 1);
    declareBias("classifier.bias", kClasses);
  } catch (const std::exception& exception) {
    return fail(exception.what());
  }

  impl.weightsBuffer = ggml_backend_alloc_ctx_tensors(impl.weights, impl.backend);
  if (!impl.weightsBuffer) {
    return fail("could not allocate the weight buffer");
  }

  for (const Pending& item : pending) {
    ggml_backend_tensor_set(item.destination, item.source->data, 0, ggml_nbytes(item.destination));
  }

  // ---------------------------------------------------------------- graph
  ggml_init_params graphParams = {32 * 1024 * 1024, nullptr, /*.no_alloc=*/true};
  impl.graph = ggml_init(graphParams);
  ggml_context* ctx = impl.graph;

  auto weight = [&](const std::string& name) {
    ggml_tensor* tensor = impl.find(name.c_str());
    if (!tensor) {
      throw std::runtime_error("internal error: weight " + name + " was not loaded");
    }
    return tensor;
  };

  try {
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCanvas, kCanvas, 1, 1);
    ggml_set_name(x, "input");
    ggml_set_input(x);
    impl.input = x;

    x = ggml_clamp(ctx, biasAdd(ctx, ggml_conv_2d(ctx, weight("stem.weight"), x, 2, 2, 1, 1, 1, 1), weight("stem.bias"), kStemChannels), 0.0f, 6.0f);

    int64_t inChannels = kStemChannels;
    for (int index = 0; index < kBlockCount; ++index) {
      const BlockSpec& spec = kBlocks[index];
      const int64_t hidden = inChannels * spec.expand;
      const std::string prefix = "blk." + std::to_string(index) + ".";
      ggml_tensor* y = x;

      if (spec.expand != 1) {
        y = ggml_clamp(ctx, pointwise(ctx, y, weight(prefix + "expand.weight"), weight(prefix + "expand.bias"), hidden), 0.0f, 6.0f);
      }
      y = ggml_clamp(ctx, biasAdd(ctx, ggml_conv_2d_dw(ctx, weight(prefix + "dw.weight"), y, spec.stride, spec.stride, 1, 1, 1, 1), weight(prefix + "dw.bias"), hidden),
                     0.0f, 6.0f);
      y = pointwise(ctx, y, weight(prefix + "project.weight"), weight(prefix + "project.bias"), spec.channels);

      x = (inChannels == spec.channels && spec.stride == 1) ? ggml_add(ctx, x, y) : y;
      inChannels = spec.channels;
    }

    x = ggml_clamp(ctx, pointwise(ctx, x, weight("head.weight"), weight("head.bias"), kHeadChannels), 0.0f, 6.0f);

    // global average pooling over the remaining 2x2 grid
    ggml_tensor* pooled = ggml_mean(ctx, ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], kHeadChannels));
    pooled = ggml_reshape_2d(ctx, pooled, kHeadChannels, 1);

    ggml_tensor* logits = ggml_add(ctx, ggml_mul_mat(ctx, weight("classifier.weight"), pooled), ggml_reshape_2d(ctx, weight("classifier.bias"), kClasses, 1));
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    impl.output = logits;

    impl.forward = ggml_new_graph(ctx);
    ggml_build_forward_expand(impl.forward, logits);
  } catch (const std::exception& exception) {
    return fail(exception.what());
  }

  impl.graphBuffer = ggml_backend_alloc_ctx_tensors(ctx, impl.backend);
  if (!impl.graphBuffer) {
    return fail("could not allocate the graph buffer");
  }

  return recognizer;
}

// ------------------------------------------------------------------ inference

std::vector<float> HccrRecognizer::logitsNormalized(const uint8_t* pixels, int stride) {
  Impl& impl = *_impl;

  std::vector<float> input(static_cast<size_t>(kCanvas) * kCanvas);
  for (int y = 0; y < kCanvas; ++y) {
    const uint8_t* row = pixels + static_cast<size_t>(y) * stride;
    for (int x = 0; x < kCanvas; ++x) {
      input[static_cast<size_t>(y) * kCanvas + x] = static_cast<float>(255 - row[x]) / 255.0f;
    }
  }
  ggml_backend_tensor_set(impl.input, input.data(), 0, input.size() * sizeof(float));

  const int64_t start = ggml_time_us();
  if (ggml_backend_graph_compute(impl.backend, impl.forward) != GGML_STATUS_SUCCESS) {
    return {};
  }
  impl.lastMilliseconds = static_cast<double>(ggml_time_us() - start) / 1000.0;

  std::vector<float> logits(kClasses);
  ggml_backend_tensor_get(impl.output, logits.data(), 0, logits.size() * sizeof(float));
  return logits;
}

std::vector<HccrCandidate> HccrRecognizer::recognizeNormalized(const uint8_t* pixels, int stride, int count) {
  const std::vector<float> logits = logitsNormalized(pixels, stride);
  if (logits.empty()) {
    return {};
  }

  float maximum = logits[0];
  for (const float value : logits) {
    maximum = std::max(maximum, value);
  }
  std::vector<float> probabilities(logits.size());
  double total = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] = std::exp(logits[i] - maximum);
    total += probabilities[i];
  }
  for (float& value : probabilities) {
    value = static_cast<float>(value / total);
  }

  std::vector<int> order(probabilities.size());
  std::iota(order.begin(), order.end(), 0);
  const int take = std::min<int>(std::max(count, 1), static_cast<int>(order.size()));
  std::partial_sort(order.begin(), order.begin() + take, order.end(), [&](int left, int right) {
    return probabilities[left] > probabilities[right];
  });

  std::vector<HccrCandidate> candidates;
  candidates.reserve(take);
  for (int i = 0; i < take; ++i) {
    candidates.push_back({_impl->charset[order[i]], probabilities[order[i]]});
  }
  return candidates;
}

std::vector<HccrCandidate> HccrRecognizer::recognize(const QImage& image, int count) {
  const QImage normalized = normalize(image);
  if (normalized.isNull()) {
    return {};
  }
  return recognizeNormalized(normalized.constBits(), static_cast<int>(normalized.bytesPerLine()), count);
}

// ------------------------------------------------------------- preprocessing

QImage HccrRecognizer::normalize(const QImage& image) {
  if (image.isNull()) {
    return {};
  }
  const QImage gray = image.convertToFormat(QImage::Format_Grayscale8);

  int minX = gray.width();
  int minY = gray.height();
  int maxX = -1;
  int maxY = -1;
  for (int y = 0; y < gray.height(); ++y) {
    const uint8_t* row = gray.constScanLine(y);
    for (int x = 0; x < gray.width(); ++x) {
      if (row[x] < kForegroundThreshold) {
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
      }
    }
  }

  QImage canvas{kCanvas, kCanvas, QImage::Format_Grayscale8};
  canvas.fill(255);
  if (maxX < 0) {
    return canvas;
  }

  const QImage cropped = gray.copy(minX, minY, maxX - minX + 1, maxY - minY + 1);
  const int longest = std::max(cropped.width(), cropped.height());
  const int targetWidth = std::max(1, static_cast<int>(std::lround(cropped.width() * static_cast<double>(kContent) / longest)));
  const int targetHeight = std::max(1, static_cast<int>(std::lround(cropped.height() * static_cast<double>(kContent) / longest)));

  const QImage scaled = resampleTriangle(cropped, targetWidth, targetHeight);

  const int offsetX = (kCanvas - targetWidth) / 2;
  const int offsetY = (kCanvas - targetHeight) / 2;
  for (int y = 0; y < targetHeight; ++y) {
    std::memcpy(canvas.scanLine(offsetY + y) + offsetX, scaled.constScanLine(y), targetWidth);
  }
  return canvas;
}
