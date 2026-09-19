#pragma once

#include "hanzi/HanziTypes.hpp"

#include <QFont>
#include <QString>

#include <functional>
#include <memory>
#include <vector>

namespace hanzi {

// recognize.py: charset_chars() - every CJK ideograph of the charset, in
// codepoint order: "gb2312" (6763), "gbk" (20902), "gb18030" (20992).
std::vector<QString> charsetCharacters(const QString& charset, QString* error = nullptr);

// The families the templates are rendered from, mirroring ink.py's
// DEFAULT_FONTS (Noto Sans CJK SC, Noto Serif CJK SC, Noto Sans CJK SC Light).
// Families the system does not have are replaced by the best CJK font
// fontconfig offers; if there is none, the build fails with a message naming
// the requirement.
struct TemplateFont {
  QString family;
  int weight = QFont::Normal;
};

std::vector<TemplateFont> defaultTemplateFonts();

// One scored template: `coverage` and `similarity` are the reference's
// verification score and first-pass cosine, `distance` its Chamfer distance.
struct TemplateCandidate {
  QString character;
  double coverage = 0.0;
  double distance = 0.0;
  double similarity = 0.0;
};

// The field route: every character of the charset rendered from a few fonts,
// thinned to a centre line, and matched in two stages - a directional-feature
// cosine over the whole alphabet, then a symmetric centre-line Chamfer and
// coverage over the shortlist (recognize.Templates, rank() and rank_all()).
//
// The templates live in one memory-mapped file per charset, so a query pays for
// the rows it touches and not for the 450 MB the alphabet weighs:
//
//   offset 0               header: magic "HZT1", format version, kCanvas,
//                          kCells, kFeatureDim, count, the three region offsets
//                          (page aligned), the charset name and a fingerprint of
//                          the fonts it was built from.
//   featuresOffset         count * kFeatureDim float32, L2 normalized, stored
//                          feature-major: one feature across every template, so
//                          the cosine first pass is a vectorizable sweep
//   distanceOffset         count * kCanvasPixels uint8, distance to centre line
//   skeletonOffset         count * kCanvasPixels / 8 bytes, one bit per pixel:
//                          the centre line, which is where the distance field
//                          is zero
//   nearOffset             the same size: every pixel within the coverage
//                          tolerance of that centre line
//   charactersOffset       count * uint32 code points
//
// The two masks turn each direction of the coverage into a bitwise and and a
// population count, which is why they are stored although the first is implied
// by the distance field: a rank walks them for every shortlisted template, and
// scanning the field instead is the difference between ten milliseconds and
// fifty. Building takes a few tens of seconds over a whole charset; callers are
// expected to run it off the GUI thread and show progress.
class GlyphTemplates {
public:
  ~GlyphTemplates();
  GlyphTemplates(const GlyphTemplates&) = delete;
  GlyphTemplates& operator=(const GlyphTemplates&) = delete;
  GlyphTemplates(GlyphTemplates&&) = delete;
  GlyphTemplates& operator=(GlyphTemplates&&) = delete;

  // The cache file for a charset, whether or not it exists.
  static QString cachePath(const QString& charset, const QString& cacheDirectory);

  // Loads the cache; nullptr with a reason in `*error` when it is missing or
  // does not match this build. Never builds.
  static std::unique_ptr<GlyphTemplates> open(const QString& charset, const QString& cacheDirectory, QString* error = nullptr);

  // Renders every character of the charset and writes the cache. `progress` is
  // called with (done, total) and may come from any thread.
  static std::unique_ptr<GlyphTemplates> build(const QString& charset, const QString& cacheDirectory, const std::vector<TemplateFont>& fonts, const std::function<void(int done, int total)>& progress, QString* error = nullptr);

  // Test seam: templates from arrays already in memory, `features` with
  // count * kFeatureDim floats and `distances` with count * kCanvasPixels
  // bytes. Scoring can then be compared against the reference implementation
  // without rebuilding the alphabet behind it.
  static std::unique_ptr<GlyphTemplates> fromArrays(std::vector<QString> characters, std::vector<float> features, std::vector<uint8_t> distances, QString* error = nullptr);

  int size() const;
  const std::vector<QString>& characters() const;

  // The alphabet as a set, for the component route's lookup filter.
  bool contains(const QString& character) const;

  // recognize.py: rank() - feature cosine over every template, then Chamfer and
  // coverage over the best `refine`, best first.
  std::vector<TemplateCandidate> rank(const Ink& query, int k = 10, int refine = 2000) const;

  // recognize.py: rank_all() - the same scoring without a shortlist, so the
  // cosine first pass can be checked against it.
  std::vector<TemplateCandidate> rankAll(const Ink& query, int k = 10) const;

private:
  GlyphTemplates();

  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace hanzi
