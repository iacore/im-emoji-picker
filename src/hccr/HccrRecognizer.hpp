#pragma once

#include <QImage>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct HccrCandidate {
  std::string character;  // UTF-8
  float probability = 0.0f;
};

// Handwritten Hanzi recognizer: a 3755-class MobileNetV2 run on ggml.
//
// One glyph per call. The image is normalized the way the model was trained
// (bounding box of the ink, long side scaled to 56px, centered in a 64x64
// white canvas, then inverted so ink is 1.0), classified, and returned as a
// ranked candidate list - the natural shape for an input method.
class HccrRecognizer {
public:
  // Returns nullptr and fills `error` when the model file cannot be used.
  static std::unique_ptr<HccrRecognizer> load(const std::string& modelPath, std::string* error = nullptr);

  ~HccrRecognizer();
  HccrRecognizer(const HccrRecognizer&) = delete;
  HccrRecognizer& operator=(const HccrRecognizer&) = delete;

  // ink-on-paper image of any size -> ranked candidates (empty when blank)
  std::vector<HccrCandidate> recognize(const QImage& image, int count = 10);

  // Same, for a bitmap that is already normalized (test seam).
  std::vector<HccrCandidate> recognizeNormalized(const uint8_t* pixels, int stride, int count = 10);

  // Raw logits for a normalized bitmap (test seam).
  std::vector<float> logitsNormalized(const uint8_t* pixels, int stride);

  // Exposed so the normalization itself can be checked against the reference.
  static QImage normalize(const QImage& image);

  // Milliseconds spent in the last classify call.
  double lastInferenceMilliseconds() const;

  int classCount() const;

private:
  HccrRecognizer();

  struct Impl;
  std::unique_ptr<Impl> _impl;
};
