#include "hanzi/GlyphTemplates.hpp"

#include "hanzi/Ink.hpp"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFontDatabase>
#include <QFontMetrics>
#include <QHash>
#include <QImage>
#include <QPainter>
#include <QSet>
#include <QTextCodec>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace hanzi {

namespace {

// ----------------------------------------------------------------- rendering

// ink.py: render_char(size=256, fill=0.84) - the glyph is drawn at
// int(256 * 0.84) = 215 px on a 256 px canvas. normalize() crops the ink box
// afterwards, so the canvas only has to be large enough that no glyph touches
// its edge.
constexpr int kRenderSize = 256;
constexpr int kFontPixelSize = static_cast<int>(kRenderSize * 0.84);

// recognize.py: thin() takes the centre line of the ink at this level.
constexpr float kInkLevel = 0.15f;

// recognize.py: rank()/rank_all() use tolerance = max(2.0, N / 32.0).
constexpr double kCoverageTolerance = kCanvas / 32.0;
// The stored field is whole pixels, so the comparison against the tolerance is
// a comparison against 4: the reference compares the float field, which its own
// uint8 cache has already truncated in the same way.
constexpr int kCoverageSteps = static_cast<int>(kCoverageTolerance);

// recognize.py: chamfer() reports this for either side being empty.
constexpr double kEmptyDistance = 1e3;

// scipy.ndimage.distance_transform_edt of an empty side is infinite;
// np.clip(..., 0, 255) is what both the stored and the query field go through.
constexpr double kMaxDistance = 255.0;

// ---------------------------------------------------------------- cache file

// The reference keeps one cache per charset as three numpy files (a 107 MB
// feature matrix, a 342 MB distance-field array and a character list). Porting
// that to one file keeps the mapping a single mmap and the regions page
// aligned, so the header stays inside the first page and every region can be
// mapped on its own.
constexpr char kMagic[4] = {'H', 'Z', 'T', '1'};
// 2: the feature matrix is stored feature-major (see fillSimilarities).
// 3: the centre-line and near-line masks are stored per template, so a rank
//    reads two kilobytes instead of scanning a sixteen kilobyte field twice.
constexpr uint32_t kFormatVersion = 3;
constexpr uint64_t kRegionAlignment = 4096;
constexpr int kCharsetNameBytes = 16;
constexpr int kFingerprintBytes = 16;

struct Header {
  char magic[4];
  uint32_t formatVersion;
  uint32_t canvas;
  uint32_t cells;
  uint32_t featureDim;
  uint32_t count;
  uint32_t reserved; // keeps the region offsets below at 8-byte alignment
  uint64_t featuresOffset;
  uint64_t distanceOffset;
  uint64_t skeletonOffset;
  uint64_t nearOffset;
  uint64_t charactersOffset;
  uint64_t fileBytes;
  char charset[kCharsetNameBytes];
  char fingerprint[kFingerprintBytes];
};

// One bit per pixel, in pixel order (y * kCanvas + x). A template's centre line
// and the pixels within the coverage tolerance of it are both fixed, so storing
// them turns the two coverage directions into a bitwise and and a population
// count instead of a walk over the field.
constexpr int kMaskBytes = kCanvasPixels / 8;

static_assert(sizeof(Header) <= kRegionAlignment, "the header must fit in the first page");

uint64_t alignRegion(uint64_t offset) {
  return (offset + kRegionAlignment - 1) / kRegionAlignment * kRegionAlignment;
}

// recognize.py: charset_chars() - a decoded two-byte code is kept only when it
// is one ideograph of the U+4E00..U+9FFF block. QTextCodec reports a bad pair
// as invalidChars instead of throwing, which is the UnicodeDecodeError of the
// reference, and it never returns a two-character string.
bool isCjkIdeograph(const QString& text) {
  if (text.size() != 1) {
    return false;
  }
  const ushort code = text.at(0).unicode();
  return code >= 0x4E00 && code <= 0x9FFF;
}

// Python's gb2312 codec accepts exactly the level 1 and 2 codes: any lead of
// 0xB0..0xF7 with a trail of 0xA1..0xFE. Qt aliases the name "GB2312" to its
// GBK codec, so the range - not a second decoder - is what separates the
// charsets (recognize.py: _in_gb2312()).
bool isGb2312Code(int lead, int trail) {
  return lead >= 0xB0 && lead <= 0xF7 && trail >= 0xA1 && trail <= 0xFE;
}

// recognize.py: _fingerprint() - the charset plus the fonts. The reference
// hashes the font files and their mtimes; a family and a weight are what this
// side can see.
QString fontFingerprint(const QString& charset, const std::vector<TemplateFont>& fonts) {
  QCryptographicHash digest(QCryptographicHash::Sha256);
  digest.addData(QByteArray(kMagic, 4));
  digest.addData(charset.toUtf8());
  for (const TemplateFont& font : fonts) {
    digest.addData(font.family.toUtf8());
    digest.addData(QByteArray::number(font.weight));
    digest.addData(";");
  }
  return QString::fromLatin1(digest.result().toHex().left(12));
}

QFont templateQFont(const TemplateFont& font) {
  QFont resolved(font.family, -1, font.weight);
  resolved.setPixelSize(kFontPixelSize);
  return resolved;
}

// A family that has no glyph for the sample would silently be replaced by Qt's
// own fallback, which is exactly the substitution defaultTemplateFonts() is
// supposed to make explicitly.
bool rendersHanzi(const TemplateFont& font) {
  const QFontMetrics metrics(templateQFont(font));
  return metrics.inFont(QChar(0x4E00)) && metrics.inFont(QChar(0x8C1E));
}

// QFontDatabase hands the families over in fontconfig's order, which is the
// order the reference's own font lookup would walk.
QString firstHanziFamily() {
  const QFontDatabase database;
  const QStringList families = database.families();
  for (const QString& family : families) {
    if (rendersHanzi(TemplateFont{family, QFont::Normal})) {
      return family;
    }
  }
  return QString();
}

// ------------------------------------------------------------------- masks

int countBits(const uint8_t* mask) {
  int total = 0;
  for (int word = 0; word < kMaskBytes / 8; ++word) {
    uint64_t value;
    std::memcpy(&value, mask + word * 8, sizeof(value));
    total += __builtin_popcountll(value);
  }
  return total;
}

int countBitsAnd(const uint8_t* left, const uint8_t* right) {
  int total = 0;
  for (int word = 0; word < kMaskBytes / 8; ++word) {
    uint64_t a;
    uint64_t b;
    std::memcpy(&a, left + word * 8, sizeof(a));
    std::memcpy(&b, right + word * 8, sizeof(b));
    total += __builtin_popcountll(a & b);
  }
  return total;
}

void setBit(uint8_t* mask, int pixel) {
  mask[pixel >> 3] |= static_cast<uint8_t>(1u << (pixel & 7));
}

// The two masks of one template, as the cache stores them: the centre line, and
// every pixel within the coverage tolerance of it.
void maskFromDistance(const uint8_t* field, uint8_t* skeleton, uint8_t* near) {
  std::memset(skeleton, 0, kMaskBytes);
  std::memset(near, 0, kMaskBytes);
  for (int pixel = 0; pixel < kCanvasPixels; ++pixel) {
    if (field[pixel] == 0) {
      setBit(skeleton, pixel);
    }
    if (field[pixel] <= kCoverageSteps) {
      setBit(near, pixel);
    }
  }
}

// ------------------------------------------------------------------ geometry

Mask maskFromInk(const Ink& ink, float threshold) {
  Mask mask;
  for (int pixel = 0; pixel < kCanvasPixels; ++pixel) {
    mask.value[pixel] = ink.value[pixel] > threshold ? 1 : 0;
  }
  return mask;
}

// recognize.py: thin() - the ink above the level, thinned. No fallback to the
// unthinned level: rank() and rank_all() bail out on an empty centre line.
Mask thinInk(const Ink& ink) {
  Mask mask;
  skeletonizeInk(ink, mask, kInkLevel);
  return mask;
}

// recognize.py: _thickened() - the centre line as a 0/1 float map, the input
// the directional features are pooled from.
Ink ribbonFromMask(const Mask& mask) {
  Ink ribbon;
  for (int pixel = 0; pixel < kCanvasPixels; ++pixel) {
    ribbon.value[pixel] = mask.value[pixel] ? 1.0f : 0.0f;
  }
  return ribbon;
}

// The 1D squared-distance transform of Felzenszwalb and Huttenlocher: given
// f[q], computes d[q] = min_p (q - p)^2 + f[p]. Exact in double precision,
// which is what scipy's distance_transform_edt is too.
void squaredDistanceTransform1d(const double* f, double* d) {
  constexpr double kFar = 1e18;
  std::array<int, kCanvas> v{};
  std::array<double, kCanvas + 1> z{};
  int k = 0;
  v[0] = 0;
  z[0] = -kFar;
  z[1] = kFar;
  for (int q = 1; q < kCanvas; ++q) {
    const double fq = f[q] + static_cast<double>(q) * q;
    double s = (fq - (f[v[k]] + static_cast<double>(v[k]) * v[k])) / (2.0 * q - 2.0 * v[k]);
    while (s <= z[k]) {
      --k;
      s = (fq - (f[v[k]] + static_cast<double>(v[k]) * v[k])) / (2.0 * q - 2.0 * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kFar;
  }
  k = 0;
  for (int q = 0; q < kCanvas; ++q) {
    while (z[k + 1] < q) {
      ++k;
    }
    const double delta = q - v[k];
    d[q] = delta * delta + f[v[k]];
  }
}

// scipy.ndimage.distance_transform_edt(~mask), in float: for every pixel, the
// exact distance to the nearest centre-line pixel. The Ink slice's
// distanceTransform() returns the uint8 field the cache stores - and the
// reference stores uint8 as well - but the reference keeps the *query's* field
// in float64. Quantising that one moves pixels across the coverage tolerance
// (4.236 becomes 4), which measured 0.0008 of coverage on 谞 and swapped two
// candidates, so the query side needs the untruncated transform.
void exactDistanceField(const Mask& mask, float* out) {
  std::vector<double> squared(kCanvasPixels, 0.0);
  std::array<double, kCanvas> line{};
  std::array<double, kCanvas> transformed{};
  for (int x = 0; x < kCanvas; ++x) {
    for (int y = 0; y < kCanvas; ++y) {
      line[y] = mask.at(y, x) ? 0.0 : kMaxDistance * kMaxDistance;
    }
    squaredDistanceTransform1d(line.data(), transformed.data());
    for (int y = 0; y < kCanvas; ++y) {
      squared[static_cast<size_t>(y) * kCanvas + x] = transformed[y];
    }
  }
  for (int y = 0; y < kCanvas; ++y) {
    std::copy_n(squared.begin() + static_cast<size_t>(y) * kCanvas, kCanvas, line.begin());
    squaredDistanceTransform1d(line.data(), transformed.data());
    for (int x = 0; x < kCanvas; ++x) {
      out[static_cast<size_t>(y) * kCanvas + x] = static_cast<float>(std::min(kMaxDistance, std::sqrt(transformed[x])));
    }
  }
}

// One template's coverage of a query, from the two masks: exact integer hit
// counts, so the value is the reference's to the last bit. This is the primary
// key of the ordering and the only thing a rank needs for most templates.
double coverageFromMasks(const uint8_t* queryPixels, int queryPixelCount, const uint8_t* queryNear, const uint8_t* skeleton, const uint8_t* near,
                         int templatePixels) {
  if (queryPixelCount == 0 || templatePixels == 0) {
    return 0.0;
  }
  const int queryHits = countBitsAnd(queryPixels, near);
  const int templateHits = countBitsAnd(queryNear, skeleton);
  return 0.5 * (static_cast<double>(queryHits) / queryPixelCount + static_cast<double>(templateHits) / templatePixels);
}

// recognize.py: chamfer() - the symmetric mean distance between the two centre
// lines, each read off the other's distance field. It is the tie-break behind
// the coverage, so a rank only pays for it where it can still change the answer.
double chamferDistance(const uint16_t* queryPixels, int queryPixelCount, const float* queryField, const uint8_t* templateField, const uint8_t* templateSkeleton, int templatePixels) {
  if (queryPixelCount == 0 || templatePixels == 0) {
    return kEmptyDistance;
  }
  double toTemplate = 0.0;
  for (int p = 0; p < queryPixelCount; ++p) {
    toTemplate += templateField[queryPixels[p]];
  }
  double toQuery = 0.0;
  for (int word = 0; word < kMaskBytes / 8; ++word) {
    uint64_t bits;
    std::memcpy(&bits, templateSkeleton + word * 8, sizeof(bits));
    while (bits != 0) {
      // A word is eight bytes and a byte holds eight pixels, so the word index
      // counts 64 pixels; the trailing-zero count is the pixel within it.
      const int pixel = word * 64 + static_cast<int>(__builtin_ctzll(bits));
      toQuery += queryField[pixel];
      bits &= bits - 1;
    }
  }
  return 0.5 * (toTemplate / queryPixelCount + toQuery / templatePixels);
}

// recognize.py: render_char() through Qt. PIL anchors "mm" on the text box
// where Qt centres the line box, and FreeType and Qt's rasteriser hint
// differently, so the two never agree pixel for pixel: measured on 谞, the ink
// bbox is within one pixel and the ink pixel count within 0.3%. normalize()
// crops the ink box, which makes the residual translation harmless, and the
// thinning that follows the rendering smooths the rest out.
Ink renderCharacter(const QString& character, const QFont& font) {
  QImage canvas(kRenderSize, kRenderSize, QImage::Format_Grayscale8);
  canvas.fill(255);
  QPainter painter(&canvas);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setFont(font);
  painter.setPen(Qt::black);
  painter.drawText(QRect(0, 0, kRenderSize, kRenderSize), Qt::AlignCenter, character);
  painter.end();
  return inkFromGray(canvas);
}

struct Scored {
  int index = 0;
  double coverage = 0.0;
  double distance = 0.0;
  double similarity = 0.0;
};

// recognize.py: rank()'s key, (-coverage, distance, -similarity), as a strict
// weak ordering.
bool betterScored(const Scored& a, const Scored& b) {
  if (a.coverage != b.coverage) {
    return a.coverage > b.coverage;
  }
  if (a.distance != b.distance) {
    return a.distance < b.distance;
  }
  return a.similarity > b.similarity;
}

TemplateCandidate candidateFrom(const QString& character, const Scored& scored) {
  TemplateCandidate candidate;
  candidate.character = character;
  candidate.coverage = scored.coverage;
  candidate.distance = scored.distance;
  candidate.similarity = scored.similarity;
  return candidate;
}

} // namespace

// ------------------------------------------------------------------ charsets

std::vector<QString> charsetCharacters(const QString& charset, QString* error) {
  const bool gb2312 = charset == QLatin1String("gb2312");
  const bool gb18030 = charset == QLatin1String("gb18030");
  if (!gb2312 && !gb18030 && charset != QLatin1String("gbk")) {
    if (error) {
      *error = QStringLiteral("unknown charset \"%1\", expected gb2312, gbk or gb18030").arg(charset);
    }
    return {};
  }

  // recognize.py: charset_chars() walks every two-byte code, then - for
  // gb18030 only - the four-byte range the standard adds for the characters
  // GBK cannot reach.
  QSet<QString> found;
  QTextCodec* twoByte = QTextCodec::codecForName("GBK");
  if (!twoByte) {
    if (error) {
      *error = QStringLiteral("Qt has no GBK codec, so the charset cannot be enumerated");
    }
    return {};
  }
  for (int lead = 0x81; lead <= 0xFE; ++lead) {
    for (int trail = 0x40; trail <= 0xFE; ++trail) {
      if (gb2312 && !isGb2312Code(lead, trail)) {
        continue;
      }
      const char bytes[2] = {static_cast<char>(lead), static_cast<char>(trail)};
      QTextCodec::ConverterState state;
      const QString text = twoByte->toUnicode(bytes, 2, &state);
      if (state.invalidChars == 0 && isCjkIdeograph(text)) {
        found.insert(text);
      }
    }
  }
  if (gb18030) {
    QTextCodec* fourByte = QTextCodec::codecForName("GB18030");
    if (!fourByte) {
      if (error) {
        *error = QStringLiteral("Qt has no GB18030 codec, so the charset cannot be enumerated");
      }
      return {};
    }
    for (int b1 = 0x81; b1 <= 0xFE; ++b1) {
      for (int b2 = 0x30; b2 <= 0x39; ++b2) {
        for (int b3 = 0x81; b3 <= 0xFE; ++b3) {
          for (int b4 = 0x30; b4 <= 0x39; ++b4) {
            const char bytes[4] = {static_cast<char>(b1), static_cast<char>(b2), static_cast<char>(b3), static_cast<char>(b4)};
            QTextCodec::ConverterState state;
            const QString text = fourByte->toUnicode(bytes, 4, &state);
            if (state.invalidChars == 0 && isCjkIdeograph(text)) {
              found.insert(text);
            }
          }
        }
      }
    }
  }

  std::vector<uint32_t> codes;
  codes.reserve(static_cast<size_t>(found.size()));
  for (const QString& text : found) {
    codes.push_back(text.at(0).unicode());
  }
  std::sort(codes.begin(), codes.end());
  std::vector<QString> characters;
  characters.reserve(codes.size());
  for (const uint32_t code : codes) {
    characters.push_back(QString(QChar(static_cast<ushort>(code))));
  }
  return characters;
}

std::vector<TemplateFont> defaultTemplateFonts() {
  // ink.py: DEFAULT_FONTS - the SC faces of Noto Sans CJK Regular, Noto Serif
  // CJK Regular and Noto Sans CJK Light (TTC index 2 of each file), which are
  // separate families on a fontconfig system.
  static const std::array<TemplateFont, 3> kWanted = {{
      {QStringLiteral("Noto Sans CJK SC"), QFont::Normal},
      {QStringLiteral("Noto Serif CJK SC"), QFont::Normal},
      {QStringLiteral("Noto Sans CJK SC Light"), QFont::Light},
  }};

  std::vector<TemplateFont> fonts;
  fonts.reserve(kWanted.size());
  bool announcedSubstitute = false;
  for (const TemplateFont& wanted : kWanted) {
    if (rendersHanzi(wanted)) {
      fonts.push_back(wanted);
      continue;
    }
    // A family the system does not have is replaced by the first CJK font
    // fontconfig offers, keeping the wanted weight: the light face of the
    // substitute is then picked by Qt instead of by a second family name.
    const QString substitute = firstHanziFamily();
    if (substitute.isEmpty()) {
      // No CJK font at all is a build failure, not a substitution, so this is
      // left to build() to report.
      return {};
    }
    if (!announcedSubstitute) {
      qWarning("hanzi: no %s, using %s for the templates", qPrintable(wanted.family), qPrintable(substitute));
      announcedSubstitute = true;
    }
    fonts.push_back(TemplateFont{substitute, wanted.weight});
  }
  return fonts;
}

// -------------------------------------------------------------------- storage

struct GlyphTemplates::Impl {
  // A mapped cache owns its file; fromArrays() owns plain vectors instead.
  std::unique_ptr<QFile> file;
  uchar* mapping = nullptr;
  const float* features = nullptr;
  const uint8_t* distances = nullptr;
  const uint8_t* skeletons = nullptr;
  const uint8_t* near = nullptr;
  const uint32_t* codes = nullptr;
  int count = 0;
  QString charset;
  QString fingerprint;
  std::vector<QString> characters;
  std::vector<float> ownedFeatures; // feature-major, transposed on adoption
  std::vector<uint8_t> ownedDistances;
  std::vector<uint8_t> ownedSkeletons;
  std::vector<uint8_t> ownedNear;
  QHash<QString, int> index;

  ~Impl() {
    if (file && mapping) {
      file->unmap(mapping);
    }
  }

  const uint8_t* distanceRow(int i) const {
    return distances + static_cast<size_t>(i) * kCanvasPixels;
  }

  // One feature across every template: a rank reads the matrix a feature at a
  // time, which turns the cosine into one vectorizable pass over contiguous
  // templates instead of a row-at-a-time reduction.
  const float* featureColumn(int d) const {
    return features + static_cast<size_t>(d) * count;
  }

  const uint8_t* skeletonMask(int i) const {
    return skeletons + static_cast<size_t>(i) * kMaskBytes;
  }

  const uint8_t* nearMask(int i) const {
    return near + static_cast<size_t>(i) * kMaskBytes;
  }

  // Points the regions at the owned arrays and builds the lookup the component
  // route filters its candidates with.
  void adoptOwned() {
    count = static_cast<int>(ownedFeatures.size() / kFeatureDim);
    // The arrays handed in come from the reference's numpy cache, which is
    // row-major; the engine wants feature-major, so they are transposed once.
    std::vector<float> transposed(ownedFeatures.size());
    for (int i = 0; i < count; ++i) {
      for (int d = 0; d < kFeatureDim; ++d) {
        transposed[static_cast<size_t>(d) * count + i] = ownedFeatures[static_cast<size_t>(i) * kFeatureDim + d];
      }
    }
    ownedFeatures = std::move(transposed);
    features = ownedFeatures.data();
    distances = ownedDistances.data();
    ownedSkeletons.assign(static_cast<size_t>(count) * kMaskBytes, 0);
    ownedNear.assign(static_cast<size_t>(count) * kMaskBytes, 0);
    for (int i = 0; i < count; ++i) {
      maskFromDistance(distances + static_cast<size_t>(i) * kCanvasPixels, ownedSkeletons.data() + static_cast<size_t>(i) * kMaskBytes,
                       ownedNear.data() + static_cast<size_t>(i) * kMaskBytes);
    }
    skeletons = ownedSkeletons.data();
    near = ownedNear.data();
    index.clear();
    for (int i = 0; i < count; ++i) {
      index.insert(characters[static_cast<size_t>(i)], i);
    }
  }
};

GlyphTemplates::GlyphTemplates() : _impl{std::make_unique<Impl>()} {
}

GlyphTemplates::~GlyphTemplates() = default;

int GlyphTemplates::size() const {
  return _impl->count;
}

const std::vector<QString>& GlyphTemplates::characters() const {
  return _impl->characters;
}

bool GlyphTemplates::contains(const QString& character) const {
  return _impl->index.contains(character);
}

QString GlyphTemplates::cachePath(const QString& charset, const QString& cacheDirectory) {
  return QDir(cacheDirectory).filePath(QStringLiteral("templates-%1.hzt").arg(charset));
}

std::unique_ptr<GlyphTemplates> GlyphTemplates::open(const QString& charset, const QString& cacheDirectory, QString* error) {
  const QString path = cachePath(charset, cacheDirectory);
  auto file = std::make_unique<QFile>(path);
  if (!file->open(QIODevice::ReadOnly)) {
    if (error) {
      *error = QStringLiteral("cannot open %1: %2").arg(path, file->errorString());
    }
    return nullptr;
  }
  const qint64 fileBytes = file->size();
  Header header{};
  if (fileBytes < static_cast<qint64>(sizeof(Header))) {
    if (error) {
      *error = QStringLiteral("%1 is too small to be a template cache").arg(path);
    }
    return nullptr;
  }
  uchar* mapping = file->map(0, fileBytes);
  if (!mapping) {
    if (error) {
      *error = QStringLiteral("cannot map %1: %2").arg(path, file->errorString());
    }
    return nullptr;
  }
  std::memcpy(&header, mapping, sizeof(Header));

  QString failure;
  const QString storedCharset = QString::fromLatin1(header.charset, strnlen(header.charset, kCharsetNameBytes));
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
    failure = QStringLiteral("%1 is not a Hanzi template cache").arg(path);
  } else if (header.formatVersion != kFormatVersion) {
    failure = QStringLiteral("%1 has format version %2, expected %3").arg(path).arg(header.formatVersion).arg(kFormatVersion);
  } else if (header.canvas != static_cast<uint32_t>(kCanvas) || header.cells != static_cast<uint32_t>(kCells) || header.featureDim != static_cast<uint32_t>(kFeatureDim)) {
    // A cache built by other constants would be read with the wrong strides,
    // so it is rejected rather than misread.
    failure = QStringLiteral("%1 was built for a %2x%2 canvas with %3x%3 cells, not %4x%4 with %5x%5").arg(path).arg(header.canvas).arg(header.cells).arg(kCanvas).arg(kCells);
  } else if (header.count == 0) {
    failure = QStringLiteral("%1 holds no templates").arg(path);
  } else if (storedCharset != charset) {
    failure = QStringLiteral("%1 holds charset %2, not %3").arg(path, storedCharset, charset);
  } else if (header.featuresOffset < sizeof(Header) || header.distanceOffset < header.featuresOffset || header.skeletonOffset < header.distanceOffset || header.nearOffset < header.skeletonOffset || header.charactersOffset < header.nearOffset) {
    failure = QStringLiteral("%1 has a damaged header").arg(path);
  } else if (header.fileBytes != static_cast<uint64_t>(fileBytes)) {
    failure = QStringLiteral("%1 is %2 bytes, its header claims %3").arg(path).arg(fileBytes).arg(header.fileBytes);
  } else if (header.featuresOffset + static_cast<uint64_t>(header.count) * kFeatureDim * sizeof(float) > header.distanceOffset ||
             header.distanceOffset + static_cast<uint64_t>(header.count) * kCanvasPixels > header.skeletonOffset ||
             header.skeletonOffset + static_cast<uint64_t>(header.count) * kMaskBytes > header.nearOffset ||
             header.nearOffset + static_cast<uint64_t>(header.count) * kMaskBytes > header.charactersOffset ||
             header.charactersOffset + static_cast<uint64_t>(header.count) * sizeof(uint32_t) > header.fileBytes) {
    failure = QStringLiteral("%1 has regions that do not fit the file").arg(path);
  }
  if (!failure.isEmpty()) {
    file->unmap(mapping);
    if (error) {
      *error = failure;
    }
    return nullptr;
  }

  auto templates = std::unique_ptr<GlyphTemplates>(new GlyphTemplates());
  Impl& impl = *templates->_impl;
  impl.file = std::move(file);
  impl.mapping = mapping;
  impl.count = static_cast<int>(header.count);
  impl.charset = charset;
  impl.fingerprint = QString::fromLatin1(header.fingerprint, strnlen(header.fingerprint, kFingerprintBytes));
  impl.features = reinterpret_cast<const float*>(mapping + header.featuresOffset);
  impl.distances = mapping + header.distanceOffset;
  impl.skeletons = mapping + header.skeletonOffset;
  impl.near = mapping + header.nearOffset;
  impl.codes = reinterpret_cast<const uint32_t*>(mapping + header.charactersOffset);
  impl.characters.reserve(static_cast<size_t>(impl.count));
  for (int i = 0; i < impl.count; ++i) {
    const uint32_t code = impl.codes[i];
    impl.characters.push_back(code <= 0xFFFF ? QString(QChar(static_cast<ushort>(code))) : QString::fromUcs4(&code, 1));
    impl.index.insert(impl.characters.back(), i);
  }
  return templates;
}

std::unique_ptr<GlyphTemplates> GlyphTemplates::build(const QString& charset, const QString& cacheDirectory, const std::vector<TemplateFont>& fonts, const std::function<void(int, int)>& progress, QString* error) {
  QString failure;
  const std::vector<QString> characters = charsetCharacters(charset, &failure);
  if (characters.empty()) {
    if (error) {
      *error = failure.isEmpty() ? QStringLiteral("charset %1 holds no CJK ideographs").arg(charset) : failure;
    }
    return nullptr;
  }

  std::vector<TemplateFont> resolved;
  resolved.reserve(fonts.size());
  for (const TemplateFont& font : fonts) {
    if (rendersHanzi(font)) {
      resolved.push_back(font);
      continue;
    }
    const QString substitute = firstHanziFamily();
    if (substitute.isEmpty()) {
      if (error) {
        *error = QStringLiteral("no CJK-capable font found: the field route renders every character of the charset from Noto Sans CJK SC, "
                                "Noto Serif CJK SC and Noto Sans CJK SC Light, and this system has none of them (install noto-fonts-cjk)");
      }
      return nullptr;
    }
    qWarning("hanzi: no %s, using %s for the templates", qPrintable(font.family), qPrintable(substitute));
    resolved.push_back(TemplateFont{substitute, font.weight});
  }
  if (resolved.empty()) {
    if (error) {
      *error = QStringLiteral("no template font given: the field route needs at least one CJK font");
    }
    return nullptr;
  }

  const int count = static_cast<int>(characters.size());
  const uint64_t featuresOffset = kRegionAlignment;
  const uint64_t distanceOffset = alignRegion(featuresOffset + static_cast<uint64_t>(count) * kFeatureDim * sizeof(float));
  const uint64_t skeletonOffset = alignRegion(distanceOffset + static_cast<uint64_t>(count) * kCanvasPixels);
  const uint64_t nearOffset = alignRegion(skeletonOffset + static_cast<uint64_t>(count) * kMaskBytes);
  const uint64_t charactersOffset = alignRegion(nearOffset + static_cast<uint64_t>(count) * kMaskBytes);
  const uint64_t fileBytes = charactersOffset + static_cast<uint64_t>(count) * sizeof(uint32_t);

  if (!QDir().mkpath(cacheDirectory)) {
    if (error) {
      *error = QStringLiteral("cannot create %1").arg(cacheDirectory);
    }
    return nullptr;
  }
  const QString path = cachePath(charset, cacheDirectory);
  // The build writes next to the final name and renames at the end, so a
  // killed build leaves a stray temp file and never a half-written cache.
  const QString tempPath = path + QStringLiteral(".tmp-%1").arg(QCoreApplication::applicationPid());
  QFile file(tempPath);
  if (!file.open(QIODevice::ReadWrite | QIODevice::Truncate) || !file.resize(static_cast<qint64>(fileBytes))) {
    if (error) {
      *error = QStringLiteral("cannot write %1: %2").arg(tempPath, file.errorString());
    }
    return nullptr;
  }
  uchar* mapping = file.map(0, static_cast<qint64>(fileBytes));
  if (!mapping) {
    if (error) {
      *error = QStringLiteral("cannot map %1: %2").arg(tempPath, file.errorString());
    }
    file.close();
    file.remove();
    return nullptr;
  }

  Header header{};
  std::memcpy(header.magic, kMagic, sizeof(kMagic));
  header.formatVersion = kFormatVersion;
  header.canvas = kCanvas;
  header.cells = kCells;
  header.featureDim = kFeatureDim;
  header.count = static_cast<uint32_t>(count);
  header.featuresOffset = featuresOffset;
  header.distanceOffset = distanceOffset;
  header.skeletonOffset = skeletonOffset;
  header.nearOffset = nearOffset;
  header.charactersOffset = charactersOffset;
  header.fileBytes = fileBytes;
  const QByteArray charsetBytes = charset.toUtf8();
  std::memcpy(header.charset, charsetBytes.constData(), std::min<size_t>(charsetBytes.size(), kCharsetNameBytes - 1));
  const QByteArray fingerprintBytes = fontFingerprint(charset, resolved).toLatin1();
  std::memcpy(header.fingerprint, fingerprintBytes.constData(), std::min<size_t>(fingerprintBytes.size(), kFingerprintBytes - 1));
  std::memcpy(mapping, &header, sizeof(Header));

  float* features = reinterpret_cast<float*>(mapping + featuresOffset);
  uint8_t* distances = mapping + distanceOffset;
  uint8_t* skeletons = mapping + skeletonOffset;
  uint8_t* near = mapping + nearOffset;
  uint32_t* codes = reinterpret_cast<uint32_t*>(mapping + charactersOffset);

  // recognize.py: build() runs the same loop for every character, one
  // character per row, so the rows can be produced in parallel: threads only
  // ever touch their own range of the mapping.
  const int threadCount = std::max(1, std::min(count, static_cast<int>(std::thread::hardware_concurrency())));
  std::atomic<int> finished{0};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(threadCount));
  for (int thread = 0; thread < threadCount; ++thread) {
    const int begin = static_cast<int>(static_cast<qint64>(count) * thread / threadCount);
    const int end = static_cast<int>(static_cast<qint64>(count) * (thread + 1) / threadCount);
    workers.emplace_back([&, begin, end]() {
      std::vector<QFont> threadFonts;
      threadFonts.reserve(resolved.size());
      for (const TemplateFont& font : resolved) {
        threadFonts.push_back(templateQFont(font));
      }
      std::array<float, kFeatureDim> feature{};
      // The features are stored feature-major, so a character's vector is
      // gathered into this block and written out by feature: writing each value
      // straight to its column would touch a cache line per value. A chunk
      // keeps the block small enough to stay in cache.
      constexpr int kChunk = 128;
      std::vector<float> block(static_cast<size_t>(kChunk) * kFeatureDim);
      for (int chunk = begin; chunk < end; chunk += kChunk) {
        const int stop = std::min(end, chunk + kChunk);
        for (int i = chunk; i < stop; ++i) {
          Ink ink;
          for (const QFont& font : threadFonts) {
            const Ink rendered = renderCharacter(characters[static_cast<size_t>(i)], font);
            for (int pixel = 0; pixel < kCanvasPixels; ++pixel) {
              ink.value[pixel] += rendered.value[pixel];
            }
          }
          const float scale = 1.0f / static_cast<float>(threadFonts.size());
          for (float& value : ink.value) {
            value *= scale;
          }
          Mask skeleton = thinInk(ink);
          if (!skeleton.any()) {
            // No font has this glyph: the unthinned ink is all there is.
            skeleton = maskFromInk(ink, kInkLevel);
          }
          Mask ribbon;
          dilateMask(skeleton, ribbon, 1);
          const Ink ribbonInk = ribbonFromMask(ribbon);
          inkFeatures(ribbonInk, feature.data());
          double norm = 0.0;
          for (const float value : feature) {
            norm += static_cast<double>(value) * value;
          }
          norm = std::sqrt(norm);
          if (norm > 0.0) {
            for (float& value : feature) {
              value = static_cast<float>(value / norm);
            }
          }
          std::memcpy(&block[static_cast<size_t>(i - chunk) * kFeatureDim], feature.data(), kFeatureDim * sizeof(float));
          uint8_t* distance = distances + static_cast<size_t>(i) * kCanvasPixels;
          distanceTransform(skeleton, distance);
          maskFromDistance(distance, skeletons + static_cast<size_t>(i) * kMaskBytes, near + static_cast<size_t>(i) * kMaskBytes);
          codes[i] = characters[static_cast<size_t>(i)].at(0).unicode();
          const int done = ++finished;
          if (progress && done % 256 == 0) {
            progress(done, count);
          }
        }
        const int rows = stop - chunk;
        for (int d = 0; d < kFeatureDim; ++d) {
          float* column = features + static_cast<size_t>(d) * count + chunk;
          for (int row = 0; row < rows; ++row) {
            column[row] = block[static_cast<size_t>(row) * kFeatureDim + d];
          }
        }
      }
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  if (progress) {
    progress(count, count);
  }

  file.unmap(mapping);
  file.flush();
  file.close();
  // POSIX rename replaces an existing cache atomically; QFile::rename refuses
  // to overwrite, which would make a rebuild fail.
  if (std::rename(QFile::encodeName(tempPath).constData(), QFile::encodeName(path).constData()) != 0) {
    if (error) {
      *error = QStringLiteral("cannot move %1 to %2: %3").arg(tempPath, path, QString::fromLocal8Bit(std::strerror(errno)));
    }
    file.remove();
    return nullptr;
  }
  return open(charset, cacheDirectory, error);
}

std::unique_ptr<GlyphTemplates> GlyphTemplates::fromArrays(std::vector<QString> characters, std::vector<float> features, std::vector<uint8_t> distances, QString* error) {
  const size_t count = characters.size();
  if (features.size() != count * kFeatureDim) {
    if (error) {
      *error = QStringLiteral("%1 characters need %2 features, got %3").arg(count).arg(count * kFeatureDim).arg(features.size());
    }
    return nullptr;
  }
  if (distances.size() != count * kCanvasPixels) {
    if (error) {
      *error = QStringLiteral("%1 characters need %2 distance bytes, got %3").arg(count).arg(count * kCanvasPixels).arg(distances.size());
    }
    return nullptr;
  }
  auto templates = std::unique_ptr<GlyphTemplates>(new GlyphTemplates());
  Impl& impl = *templates->_impl;
  impl.characters = std::move(characters);
  impl.ownedFeatures = std::move(features);
  impl.ownedDistances = std::move(distances);
  impl.adoptOwned();
  return templates;
}

// ---------------------------------------------------------------- query side

// The query's centre-line pixels in ascending order plus its distance field:
// every shortlisted template is scored against exactly these, so the query side
// is prepared once per rank instead of walked once per template.
struct QuerySide {
  std::array<uint8_t, kMaskBytes> pixels{};
  std::array<uint8_t, kMaskBytes> near{};
  std::vector<uint16_t> pixelList;
  std::array<float, kCanvasPixels> field{};
  int pixelCount = 0;
};

QuerySide prepareQuery(const Mask& skeleton) {
  QuerySide query;
  query.pixelList.reserve(1024);
  for (int pixel = 0; pixel < kCanvasPixels; ++pixel) {
    if (skeleton.value[pixel]) {
      setBit(query.pixels.data(), pixel);
      query.pixelList.push_back(static_cast<uint16_t>(pixel));
    }
  }
  query.pixelCount = static_cast<int>(query.pixelList.size());
  exactDistanceField(skeleton, query.field.data());
  for (int pixel = 0; pixel < kCanvasPixels; ++pixel) {
    if (query.field[pixel] <= kCoverageTolerance) {
      setBit(query.near.data(), pixel);
    }
  }
  return query;
}

// recognize.py: rank()'s first pass, the feature cosine over every template.
//
// One feature at a time across all templates: the inner loop has no reduction,
// so it vectorizes as written, and the accumulator stays in the last level cache
// while the matrix streams past once. The summation order per template is the
// feature order, which is the order the reference's BLAS reduces in, and the
// sum is taken in double because the reference's float32 product carries more
// round-off than that (measured 5.8e-7 against the exact sum on 谞).
// The pass is 26.7 million multiply-adds per rank, and a baseline x86-64 build
// cannot vectorize it at all: GCC declines to vectorize floating point loops
// without an ISA that offers wider registers. The clones carry both versions -
// every machine runs the widest one it has, and the package stays valid for the
// ones that have neither. The accumulator is double in both, each template's sum
// keeps the feature order, and the clones only differ in how many of them are
// added at once, so a machine that takes the AVX2 clone reports the same
// candidates as one that does not.
__attribute__((target_clones("default", "avx2")))
void fillSimilarities(const float* features, int count, const float* queryFeature, float* similarities) {
  std::vector<double> accumulator(static_cast<size_t>(count), 0.0);
  for (int d = 0; d < kFeatureDim; ++d) {
    const double query = queryFeature[d];
    const float* column = features + static_cast<size_t>(d) * count;
    for (int i = 0; i < count; ++i) {
      accumulator[static_cast<size_t>(i)] += static_cast<double>(column[i]) * query;
    }
  }
  for (int i = 0; i < count; ++i) {
    similarities[i] = static_cast<float>(accumulator[static_cast<size_t>(i)]);
  }
}

// ------------------------------------------------------------------- matching

std::vector<TemplateCandidate> GlyphTemplates::rank(const Ink& query, int k, int refine) const {
  const Impl& impl = *_impl;
  if (impl.count == 0 || k <= 0) {
    return {};
  }
  const Mask querySkeleton = thinInk(query);
  if (!querySkeleton.any()) {
    return {};
  }
  Mask queryRibbon;
  dilateMask(querySkeleton, queryRibbon, 1);
  const Ink ribbonInk = ribbonFromMask(queryRibbon);
  std::array<float, kFeatureDim> queryFeature{};
  inkFeatures(ribbonInk, queryFeature.data());

  // recognize.py: rank() reads the feature matrix in 4096-row blocks to keep
  // resident memory flat; here the regions are one mapping either way, and the
  // dot product of a row does not depend on the blocking. The sum is taken in
  // double: the reference's float32 BLAS product carries its own round-off
  // (measured 5.8e-7 against the exact sum on 谞), and the exact sum lands
  // closer to the reference's value than a second float32 reduction does.
  std::vector<float> similarities(static_cast<size_t>(impl.count), 0.0f);
  fillSimilarities(impl.features, impl.count, queryFeature.data(), similarities.data());

  const int shortlistSize = std::min(impl.count, std::max(refine, k));
  std::vector<int> shortlist(static_cast<size_t>(impl.count));
  for (int i = 0; i < impl.count; ++i) {
    shortlist[static_cast<size_t>(i)] = i;
  }
  const auto betterSimilarity = [&similarities](int a, int b) {
    if (similarities[static_cast<size_t>(a)] != similarities[static_cast<size_t>(b)]) {
      return similarities[static_cast<size_t>(a)] > similarities[static_cast<size_t>(b)];
    }
    return a < b;
  };
  std::partial_sort(shortlist.begin(), shortlist.begin() + shortlistSize, shortlist.end(), betterSimilarity);

  const QuerySide querySide = prepareQuery(querySkeleton);

  // The whole shortlist by coverage, which is the ordering's primary key and
  // costs two bitwise ands per template.
  std::vector<double> coverages(static_cast<size_t>(shortlistSize), 0.0);
  for (int s = 0; s < shortlistSize; ++s) {
    const int i = shortlist[static_cast<size_t>(s)];
    coverages[static_cast<size_t>(s)] = coverageFromMasks(querySide.pixels.data(), querySide.pixelCount, querySide.near.data(), impl.skeletonMask(i), impl.nearMask(i),
                                                          countBits(impl.skeletonMask(i)));
  }

  // The distance only breaks ties behind the coverage, so it is computed for the
  // candidates that can still reach the answer: those at or above the k-th best
  // coverage. Everything below that sorts behind all of them, so the first k of
  // this order and of the reference's whole-shortlist order are the same.
  const int wanted = std::min(k, shortlistSize);
  std::vector<double> ranked(coverages);
  std::nth_element(ranked.begin(), ranked.begin() + (wanted - 1), ranked.end(), std::greater<double>());
  const double cutoff = ranked[static_cast<size_t>(wanted - 1)];

  std::vector<Scored> scored;
  scored.reserve(static_cast<size_t>(wanted));
  for (int s = 0; s < shortlistSize; ++s) {
    if (coverages[static_cast<size_t>(s)] < cutoff) {
      continue;
    }
    const int i = shortlist[static_cast<size_t>(s)];
    const uint8_t* skeleton = impl.skeletonMask(i);
    Scored entry;
    entry.index = i;
    entry.coverage = coverages[static_cast<size_t>(s)];
    entry.distance = chamferDistance(querySide.pixelList.data(), querySide.pixelCount, querySide.field.data(), impl.distanceRow(i), skeleton, countBits(skeleton));
    entry.similarity = similarities[static_cast<size_t>(i)];
    scored.push_back(entry);
  }
  // Stable, like the reference's sort on a key: ties keep the shortlist order,
  // which is the similarity order.
  std::stable_sort(scored.begin(), scored.end(), betterScored);

  std::vector<TemplateCandidate> candidates;
  const int answer = std::min(k, static_cast<int>(scored.size()));
  candidates.reserve(static_cast<size_t>(answer));
  for (int i = 0; i < answer; ++i) {
    candidates.push_back(candidateFrom(impl.characters[static_cast<size_t>(scored[static_cast<size_t>(i)].index)], scored[static_cast<size_t>(i)]));
  }
  return candidates;
}

std::vector<TemplateCandidate> GlyphTemplates::rankAll(const Ink& query, int k) const {
  const Impl& impl = *_impl;
  if (impl.count == 0 || k <= 0) {
    return {};
  }
  const Mask querySkeleton = thinInk(query);
  if (!querySkeleton.any()) {
    return {};
  }
  const QuerySide querySide = prepareQuery(querySkeleton);

  // recognize.py: rank_all() scores every row and sorts on coverage alone, so
  // the cosine first pass can be checked without a shortlist. The candidates
  // carry no distance and no similarity, exactly as the reference returns them.
  std::vector<Scored> scored;
  scored.reserve(static_cast<size_t>(impl.count));
  for (int i = 0; i < impl.count; ++i) {
    Scored entry;
    entry.index = i;
    entry.coverage = coverageFromMasks(querySide.pixels.data(), querySide.pixelCount, querySide.near.data(), impl.skeletonMask(i), impl.nearMask(i), countBits(impl.skeletonMask(i)));
    scored.push_back(entry);
  }
  const auto betterCoverage = [](const Scored& a, const Scored& b) {
    return a.coverage > b.coverage;
  };
  std::stable_sort(scored.begin(), scored.end(), betterCoverage);

  std::vector<TemplateCandidate> candidates;
  const int wanted = std::min(k, static_cast<int>(scored.size()));
  candidates.reserve(static_cast<size_t>(wanted));
  for (int i = 0; i < wanted; ++i) {
    candidates.push_back(candidateFrom(impl.characters[static_cast<size_t>(scored[static_cast<size_t>(i)].index)], scored[static_cast<size_t>(i)]));
  }
  return candidates;
}

} // namespace hanzi
