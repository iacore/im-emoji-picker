#pragma once

#include <QImage>

// Separable triangle-filter resample, mirroring the reference pipelines' PIL
// BILINEAR: the kernel's support grows with the reduction factor. Plain box
// averaging over-blurs thin strokes and measured ~20 points of top-1 worse on
// the classifier, and the ink normalization in the Hanzi routes is measured
// against PIL-rendered references, so both use this one kernel.
QImage resampleTriangle(const QImage& source, int targetWidth, int targetHeight);
