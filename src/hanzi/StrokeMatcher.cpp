#include "hanzi/StrokeMatcher.hpp"

#include <QFile>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace hanzi {
namespace {

// mmah_matcher.py: NUM_POINTS, NUM_VALUES, MAGIC_PER_STROKE_WEIGHT. The encoded
// stroke is the four sampled points, then the angle, then the length.
constexpr int kNumPoints = 4;
constexpr int kNumValues = 256;
constexpr int kMagicPerStrokeWeight = 4;
constexpr int kValuesPerStroke = 2 * kNumPoints + 2;

// preprocess()'s defaults: a box narrower than this is padded out, and a box
// that is not square is padded to a ratio of this.
constexpr double kMinWidth = 8.0;
constexpr double kMaxRatio = 1.0;

// A stroke count is a byte in the file, so the dictionary cannot hold more than
// 255 of them and the group index is one longer than that.
constexpr int kMaxStrokes = 255;
constexpr int kGroupCount = kMaxStrokes + 1;

constexpr char kMagic[4] = {'H', 'Z', 'M', 'D'};
constexpr uint32_t kVersion = 1;

using EncodedStroke = std::array<uint8_t, kValuesPerStroke>;

// numpy's np.round is round-half-to-even and the reference's integers are
// whatever it produces; std::nearbyint under the default rounding mode is that
// same operation. This is what keeps the encoded query identical, and it stops
// being true if the compiler is allowed to reassociate (-ffast-math).
double roundHalfEven(double value) {
  return std::nearbyint(value);
}

// mmah_matcher._adjust_box, in the same operation order. The min-width checks
// read the box as it was found, not as padding the first axis left it, and the
// pads are whole numbers; both are visible in the points the box produces.
void adjustBox(double& lowX, double& lowY, double& highX, double& highY) {
  const double width = highX - lowX;
  const double height = highY - lowY;
  if (width < kMinWidth) {
    const double pad = std::ceil((kMinWidth - width) / 2.0);
    lowX -= pad;
    highX += pad;
  }
  if (height < kMinWidth) {
    const double pad = std::ceil((kMinWidth - height) / 2.0);
    lowY -= pad;
    highY += pad;
  }
  const double paddedWidth = highX - lowX;
  const double paddedHeight = highY - lowY;
  if (paddedWidth < paddedHeight / kMaxRatio) {
    const double pad = std::ceil((paddedHeight / kMaxRatio - paddedWidth) / 2.0);
    lowX -= pad;
    highX += pad;
  } else if (paddedHeight < paddedWidth / kMaxRatio) {
    const double pad = std::ceil((paddedWidth / kMaxRatio - paddedHeight) / 2.0);
    lowY -= pad;
    highY += pad;
  }
}

// np.interp at one target: the value of the segment that starts at the last
// knot not after the target, except that the final knot - and anything past it -
// returns the final value. numpy's own search lands on the same segment, and the
// caller rounds the result. `knots` is strictly shorter than `mapped` by one and
// never empty.
Point interpolatePoint(double target, const std::vector<double>& knots, const std::vector<Point>& mapped) {
  const size_t last = knots.size() - 1;
  if (target >= knots[last]) {
    return mapped[last];
  }
  size_t low = 0;
  size_t high = last;
  while (high - low > 1) {
    const size_t middle = (low + high) / 2;
    if (knots[middle] <= target) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const double dist = target - knots[low];
  const double slopeX = (mapped[low + 1].x - mapped[low].x) / (knots[low + 1] - knots[low]);
  const double slopeY = (mapped[low + 1].y - mapped[low].y) / (knots[low + 1] - knots[low]);
  // numpy evaluates slope * dist + value, not value + slope * dist scaled some
  // other way, and the two differ in the last bits.
  return Point{slopeX * dist + mapped[low].x, slopeY * dist + mapped[low].y};
}

// mmah_matcher._resample: kNumPoints points at equal arc length along the
// polyline, over the already-mapped points. A stroke that is a single point, or
// that maps onto one, is repeated.
void resample(const std::vector<Point>& mapped, std::vector<Point>& out) {
  out.assign(kNumPoints, mapped.front());
  if (mapped.size() < 2) {
    return;
  }
  std::vector<double> cumulative(mapped.size(), 0.0);
  for (size_t i = 1; i < mapped.size(); ++i) {
    cumulative[i] = cumulative[i - 1] + std::hypot(mapped[i].x - mapped[i - 1].x, mapped[i].y - mapped[i - 1].y);
  }
  const double total = cumulative.back();
  if (!(total > 0.0)) {
    return;
  }
  // np.linspace(0.0, total, 4): i * step, with the endpoint pinned to the total.
  const double step = total / (kNumPoints - 1);
  for (int i = 0; i < kNumPoints; ++i) {
    const double target = i == kNumPoints - 1 ? total : static_cast<double>(i) * step;
    out[i] = interpolatePoint(target, cumulative, mapped);
  }
}

// Python's int % 256, which is never negative; the angle lands on 256 for an
// angle of exactly pi and has to wrap to 0.
int wrapAngle(int value) {
  const int wrapped = value % kNumValues;
  return wrapped < 0 ? wrapped + kNumValues : wrapped;
}

// mmah_matcher.preprocess: the drawing's bounding box fits the stroke into
// 0..255, each stroke is resampled to four points, and those plus an encoded
// angle and length are the query. False when the drawing has no strokes or an
// empty stroke; the reference raises ValueError there, and there is no error
// channel through match().
bool encodeQuery(const Strokes& strokes, std::vector<EncodedStroke>& out) {
  out.clear();
  if (strokes.empty()) {
    return false;
  }
  bool seen = false;
  double lowX = 0.0;
  double lowY = 0.0;
  double highX = 0.0;
  double highY = 0.0;
  for (const Stroke& stroke : strokes) {
    if (stroke.empty()) {
      return false;
    }
    for (const Point& point : stroke) {
      if (!seen) {
        lowX = highX = point.x;
        lowY = highY = point.y;
        seen = true;
        continue;
      }
      lowX = std::min(lowX, point.x);
      lowY = std::min(lowY, point.y);
      highX = std::max(highX, point.x);
      highY = std::max(highY, point.y);
    }
  }
  adjustBox(lowX, lowY, highX, highY);
  const double width = highX - lowX;
  const double height = highY - lowY;
  // np.where(high - low == 0, 1.0, high - low): a flat axis gets a scale of 255
  // rather than a division by zero, so its mapped points all land on 0.
  const double scaleX = (kNumValues - 1) / (width == 0.0 ? 1.0 : width);
  const double scaleY = (kNumValues - 1) / (height == 0.0 ? 1.0 : height);

  std::vector<Point> mapped;
  std::vector<Point> sampled(kNumPoints);
  out.reserve(strokes.size());
  for (const Stroke& stroke : strokes) {
    mapped.resize(stroke.size());
    for (size_t i = 0; i < stroke.size(); ++i) {
      mapped[i].x = roundHalfEven((stroke[i].x - lowX) * scaleX);
      mapped[i].y = roundHalfEven((stroke[i].y - lowY) * scaleY);
    }
    resample(mapped, sampled);
    EncodedStroke encoded{};
    for (int i = 0; i < kNumPoints; ++i) {
      // Rounding again is what the reference does; the points are already whole
      // numbers, so this only has to land on the same integers.
      encoded[2 * i] = static_cast<uint8_t>(roundHalfEven(sampled[i].x));
      encoded[2 * i + 1] = static_cast<uint8_t>(roundHalfEven(sampled[i].y));
    }
    const double dx = sampled[kNumPoints - 1].x - sampled[0].x;
    const double dy = sampled[kNumPoints - 1].y - sampled[0].y;
    const double angle = std::atan2(dy, dx);
    // ((angle + pi) * 256) / (2 * pi), multiplied before it is divided.
    encoded[2 * kNumPoints] = static_cast<uint8_t>(wrapAngle(static_cast<int>(roundHalfEven((angle + M_PI) * kNumValues / (2.0 * M_PI)))));
    encoded[2 * kNumPoints + 1] = static_cast<uint8_t>(roundHalfEven(std::hypot(dx, dy) / 2.0));
    out.push_back(encoded);
  }
  return true;
}

// mmah_matcher.score_similarity, term by term in the same order: the sum is
// exact only if the additions happen in this order and the angle penalty is
// scaled length * angle.
double scoreSimilarity(const std::vector<EncodedStroke>& query, const uint8_t* reference) {
  double score = 0.0;
  for (size_t s = 0; s < query.size(); ++s) {
    const uint8_t* queryStroke = query[s].data();
    const uint8_t* referenceStroke = reference + s * kValuesPerStroke;
    for (int p = 0; p < kNumPoints; ++p) {
      score -= std::abs(static_cast<int>(queryStroke[2 * p]) - static_cast<int>(referenceStroke[2 * p]));
      score -= std::abs(static_cast<int>(queryStroke[2 * p + 1]) - static_cast<int>(referenceStroke[2 * p + 1]));
    }
    const int difference = std::abs(static_cast<int>(queryStroke[2 * kNumPoints]) - static_cast<int>(referenceStroke[2 * kNumPoints]));
    const double angleSimilarity = std::min(difference, kNumValues - difference);
    const double lengthy = (queryStroke[2 * kNumPoints + 1] + referenceStroke[2 * kNumPoints + 1]) / static_cast<double>(kNumValues);
    score -= kMagicPerStrokeWeight * kNumPoints * lengthy * angleSimilarity;
  }
  return score;
}

// Little-endian reader over the whole dictionary file; every read is bounds
// checked so a truncated or foreign file is a message, never a fault.
class Reader {
public:
  Reader(const uchar* data, qsizetype size) : _data{data}, _size{size} {
  }

  bool readU8(uint8_t& out) {
    if (_position >= _size) {
      return false;
    }
    out = _data[_position++];
    return true;
  }

  bool readU32(uint32_t& out) {
    if (_size - _position < 4) {
      return false;
    }
    out = static_cast<uint32_t>(_data[_position]) | (static_cast<uint32_t>(_data[_position + 1]) << 8) | (static_cast<uint32_t>(_data[_position + 2]) << 16) | (static_cast<uint32_t>(_data[_position + 3]) << 24);
    _position += 4;
    return true;
  }

  bool atEnd() const {
    return _position == _size;
  }

private:
  const uchar* _data;
  qsizetype _size;
  qsizetype _position = 0;
};

} // namespace

struct StrokeDictionary::Impl {
  // One character: where its encoded strokes start in `values` and how many
  // there are. Characters stay in file order, which is the order the reference's
  // dict holds them in and therefore its tie order.
  struct Entry {
    uint32_t codePoint;
    uint32_t offset;
    uint8_t strokeCount;
  };

  std::vector<Entry> entries;
  std::vector<uint8_t> values;

  // Characters of one stroke count, as indices into `entries`: the reference
  // scores only that group. groupStart[c] .. groupStart[c + 1] is the range.
  std::vector<uint32_t> order;
  std::array<uint32_t, kGroupCount + 1> groupStart{};
};

StrokeDictionary::StrokeDictionary() : _impl{std::make_unique<Impl>()} {
}

StrokeDictionary::~StrokeDictionary() = default;

int StrokeDictionary::size() const {
  return static_cast<int>(_impl->entries.size());
}

std::unique_ptr<StrokeDictionary> StrokeDictionary::load(const QString& path, QString* error) {
  auto fail = [error](const QString& message) {
    if (error) {
      *error = message;
    }
    return std::unique_ptr<StrokeDictionary>{};
  };

  QFile file(path);
  if (!file.open(QIODevice::ReadOnly)) {
    return fail(QStringLiteral("could not open the stroke dictionary %1").arg(path));
  }
  const QByteArray bytes = file.readAll();
  Reader reader{reinterpret_cast<const uchar*>(bytes.constData()), bytes.size()};

  uint8_t header[8] = {};
  for (uint8_t& byte : header) {
    if (!reader.readU8(byte)) {
      return fail(QStringLiteral("%1 is not a stroke dictionary: it ends in the header").arg(path));
    }
  }
  if (std::memcmp(header, kMagic, sizeof(kMagic)) != 0) {
    return fail(QStringLiteral("%1 is not a stroke dictionary: bad magic").arg(path));
  }
  uint32_t version = 0;
  std::memcpy(&version, header + 4, sizeof(version));
  if (version != kVersion) {
    return fail(QStringLiteral("%1 has stroke dictionary version %2, expected %3").arg(path).arg(version).arg(kVersion));
  }

  std::unique_ptr<StrokeDictionary> dictionary{new StrokeDictionary()};
  Impl& impl = *dictionary->_impl;
  uint32_t count = 0;
  if (!reader.readU32(count)) {
    return fail(QStringLiteral("%1 is truncated: no character count").arg(path));
  }
  impl.entries.reserve(count);
  // 10 bytes per stroke is the file's own promise; reserving on a corrupt count
  // would ask for gigabytes before the length checks see the first record.
  if (bytes.size() - 12 > 0) {
    impl.values.reserve(static_cast<size_t>(bytes.size() - 12));
  }
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t codePoint = 0;
    uint8_t strokeCount = 0;
    if (!reader.readU32(codePoint) || !reader.readU8(strokeCount)) {
      return fail(QStringLiteral("%1 is truncated in record %2").arg(path).arg(i));
    }
    if (strokeCount == 0) {
      return fail(QStringLiteral("%1 has a character with no strokes in record %2").arg(path).arg(i));
    }
    Impl::Entry entry{codePoint, static_cast<uint32_t>(impl.values.size()), strokeCount};
    impl.values.resize(impl.values.size() + static_cast<size_t>(strokeCount) * kValuesPerStroke);
    uint8_t* destination = impl.values.data() + entry.offset;
    for (int value = 0; value < strokeCount * kValuesPerStroke; ++value) {
      if (!reader.readU8(destination[value])) {
        return fail(QStringLiteral("%1 is truncated in record %2").arg(path).arg(i));
      }
    }
    impl.entries.push_back(entry);
  }
  if (!reader.atEnd()) {
    return fail(QStringLiteral("%1 has trailing data").arg(path));
  }

  // Counting sort by stroke count, which keeps each group in file order.
  std::array<uint32_t, kGroupCount> histogram{};
  for (const Impl::Entry& entry : impl.entries) {
    ++histogram[entry.strokeCount];
  }
  uint32_t total = 0;
  for (int c = 0; c < kGroupCount; ++c) {
    impl.groupStart[c] = total;
    total += histogram[c];
  }
  impl.groupStart[kGroupCount] = total;
  std::vector<uint32_t> next(impl.groupStart.begin(), impl.groupStart.begin() + kGroupCount);
  impl.order.resize(impl.entries.size());
  for (uint32_t entry = 0; entry < impl.entries.size(); ++entry) {
    impl.order[next[impl.entries[entry].strokeCount]++] = entry;
  }
  return dictionary;
}

std::vector<std::pair<QString, double>> StrokeDictionary::match(const Strokes& strokes, int k) const {
  std::vector<std::pair<QString, double>> ranked;
  if (k <= 0) {
    return ranked;
  }
  // mmah_matcher.Matcher.match: no strokes, no candidates.
  std::vector<EncodedStroke> query;
  if (!encodeQuery(strokes, query) || query.size() > static_cast<size_t>(kMaxStrokes)) {
    return ranked;
  }

  const Impl& impl = *_impl;
  const uint32_t begin = impl.groupStart[query.size()];
  const uint32_t end = impl.groupStart[query.size() + 1];
  struct Scored {
    uint32_t entry;
    double score;
  };
  std::vector<Scored> scored;
  scored.reserve(end - begin);
  for (uint32_t i = begin; i < end; ++i) {
    const uint32_t entry = impl.order[i];
    scored.push_back({entry, scoreSimilarity(query, impl.values.data() + impl.entries[entry].offset)});
  }
  // The reference sorts with a stable sort on the descending score, so
  // characters that score the same keep the dictionary's order.
  std::stable_sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
    return a.score > b.score;
  });

  const size_t take = std::min(static_cast<size_t>(k), scored.size());
  ranked.reserve(take);
  for (size_t i = 0; i < take; ++i) {
    const uint codePoint = impl.entries[scored[i].entry].codePoint;
    ranked.emplace_back(QString::fromUcs4(&codePoint, 1), scored[i].score);
  }
  return ranked;
}

} // namespace hanzi
