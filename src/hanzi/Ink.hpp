#pragma once

#include "hanzi/HanziTypes.hpp"

#include <QImage>

namespace hanzi {

// ink.py: _INK_EDGE - the ink/paper threshold for rendered or scanned ink.
constexpr int kInkThreshold = 110;

// ink.py: binarize() - gray -> boolean ink mask at the given threshold.
Mask inkMaskFromGray(const QImage& gray, int threshold = kInkThreshold);

// ink.py: normalize() - crop the ink bounding box, scale the long side to
// kCanvas - 8, centre it on a kCanvas canvas and invert so ink is 1. Blank
// input gives a blank map.
Ink inkFromGray(const QImage& gray, int threshold = kInkThreshold);

// ink.py: render_strokes() and then normalize() - the way a pad drawing reaches
// the templates. `size` is the reference's working canvas (256); a `width` of 0
// uses its pen width, 5.5% of the drawing's bounding box.
Ink inkFromStrokes(const Strokes& strokes, int size = 256, int width = 0);

// ink.py: normalize_ink() - re-normalize an already normalized map, for a part
// of a drawing that has to sit in the same frame as a whole character.
Ink normalizeInk(const Ink& ink);

// ink.py: features() - four directional gradient maps plus ink density, pooled
// into kCells x kCells and L2 normalized, so the features see the drawing's
// shape and not the pen's weight. Writes kFeatureDim floats.
void inkFeatures(const Ink& ink, float* out);

// thin.py: skeletonize() - Zhang-Suen thinning. `recognize.thin()` uses it on
// `ink > 0.15`, which is the default here.
void skeletonizeInk(const Ink& ink, Mask& out, float threshold = 0.15f);

// thin.py: skeletonize() - the same thinning for an existing boolean mask.
void skeletonizeMask(const Mask& mask, Mask& out);

// scipy.ndimage.distance_transform_edt(~skeleton), clipped to 255: for every
// pixel, the distance to the nearest centre-line pixel. `out` holds
// kCanvasPixels entries.
void distanceTransform(const Mask& mask, uint8_t* out);

// recognize.py: _thickened() - grow a centre line into a constant-width ribbon,
// which is what the directional features are extracted from.
void dilateMask(const Mask& mask, Mask& out, int radius = 1);

// thin.py: trace_paths() and paths_to_strokes() - centre-line pixels to
// polylines (Douglas-Peucker, epsilon 2.5), returned in (x, y) order.
Strokes strokesFromSkeleton(const Mask& skeleton, int minPoints = 6);

} // namespace hanzi
