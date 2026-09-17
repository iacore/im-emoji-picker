// Runs the ggml port over pre-normalized 64x64 bitmaps and dumps its raw
// logits, so compare_ggml.py can diff them against the numpy reference.
//
//   hccr-verify --model hccr-mobilenetv2.gguf --dump logits.bin *.png

#include "hccr/HccrRecognizer.hpp"

#include <QGuiApplication>
#include <QImage>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  QGuiApplication app(argc, argv);

  std::string modelPath;
  std::string dumpPath;
  std::vector<std::string> images;
  int count = 10;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--model" && i + 1 < argc) {
      modelPath = argv[++i];
    } else if (argument == "--dump" && i + 1 < argc) {
      dumpPath = argv[++i];
    } else if (argument == "-k" && i + 1 < argc) {
      count = std::atoi(argv[++i]);
    } else {
      images.push_back(argument);
    }
  }

  if (modelPath.empty() || images.empty()) {
    std::fprintf(stderr, "usage: hccr-verify --model <model.gguf> [--dump logits.bin] [-k N] <image.png>...\n");
    return 2;
  }

  std::string error;
  const std::unique_ptr<HccrRecognizer> recognizer = HccrRecognizer::load(modelPath, &error);
  if (!recognizer) {
    std::fprintf(stderr, "failed to load %s: %s\n", modelPath.c_str(), error.c_str());
    return 1;
  }
  std::printf("model: %s, classes: %d\n", modelPath.c_str(), recognizer->classCount());

  std::ofstream dump;
  if (!dumpPath.empty()) {
    dump.open(dumpPath, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!dump) {
      std::fprintf(stderr, "cannot write %s\n", dumpPath.c_str());
      return 1;
    }
  }

  double totalMilliseconds = 0.0;
  for (const std::string& path : images) {
    QImage image;
    if (!image.load(QString::fromStdString(path))) {
      std::fprintf(stderr, "cannot read %s\n", path.c_str());
      return 1;
    }
    image = image.convertToFormat(QImage::Format_Grayscale8);

    const std::vector<float> logits = recognizer->logitsNormalized(image.constBits(), static_cast<int>(image.bytesPerLine()));
    if (logits.empty()) {
      std::fprintf(stderr, "inference failed for %s\n", path.c_str());
      return 1;
    }
    if (dump) {
      dump.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
    }

    const std::vector<HccrCandidate> candidates = recognizer->recognizeNormalized(image.constBits(), static_cast<int>(image.bytesPerLine()), count);
    totalMilliseconds += recognizer->lastInferenceMilliseconds();

    std::printf("%s ->", path.c_str());
    for (const HccrCandidate& candidate : candidates) {
      std::printf(" %s %.2f%%", candidate.character.c_str(), candidate.probability * 100.0f);
    }
    std::printf("\n");
  }

  std::printf("mean inference: %.2f ms over %zu images\n", totalMilliseconds / images.size(), images.size());
  return 0;
}
