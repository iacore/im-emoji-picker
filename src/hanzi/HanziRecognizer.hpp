#pragma once

#include "hanzi/HanziTypes.hpp"

#include <QString>

#include <functional>
#include <memory>
#include <vector>

namespace hanzi {

// One character one route produced, with the route that produced it. The
// origin reaches the UI: "trajectory", "field", "composed ⿰讠胥".
struct HanziCandidate {
  QString character;
  double score = 0.0;
  QString origin;
};

// A route's whole answer, best first. `details` is optional and parallel to
// `ranked`: the component route puts the description it matched there
// ("⿰讠胥"), so the UI can say where a rare character came from.
struct HanziSource {
  QString origin;
  double weight = 1.0;
  std::vector<QString> ranked;
  std::vector<QString> details;
};

// The picker shows one row of ten, so the routes have to be merged into one
// order. The reference prints them side by side instead (its results table has
// the per-route measurements) because it has no UI to rank for. A route's own
// rank is most of the confidence, so the merge is a weighted reciprocal rank,
// 1 / (1 + rank) - the same scale the reference's part_candidates() merges in.
//
// The weights say how much each route is trusted on *pad input*, which is what
// this picker feeds them: the trajectory route scored 21/21 top-1 there, the
// component route is what names compounds, and the field route's 39/42 was
// measured on rendered glyphs and is visibly noisy on pen strokes. The weights
// are small corrections on top of the rank, not a verdict, and nothing is
// dropped: every route's rank 1 still reaches the row.
constexpr double kModelWeight = 1.0;
constexpr double kTrajectoryWeight = 0.9;
constexpr double kFieldWeight = 0.6;
constexpr double kComposedWeight = 0.85;

// Weighted reciprocal rank over every source, best first, deduplicated by
// character (the best score wins, and the earliest source keeps a tie). Ties
// keep the order in which the characters first appeared, so the row is stable.
// A source's detail, when it has one, is appended to the origin it reports.
std::vector<HanziCandidate> mergeSources(const std::vector<HanziSource>& sources, int limit);

// The three local routes of the reference tool, over a pad drawing:
//
//   trajectory  the strokes against the 9,574 Make Me a Hanzi medians.
//   field       the ink against every character of the charset, thinned.
//   composed    the parts of the drawing, named through the IDS table.
//
// Quick stops after the trajectory route, which is microseconds. Detailed adds
// the other two, which walk the whole alphabet (seconds warm, and the first
// call builds the templates, which is minutes over 20,902 characters) and
// therefore belong off the GUI thread; `status` says what is happening.
class HanziRecognizer {
public:
  struct Paths {
    QString dictionary;    // written by tools/hanzi/convert_data.py
    QString ids;           // written by tools/hanzi/convert_data.py
    QString templateCache; // directory the field templates live in
    QString charset = QStringLiteral("gbk");
  };

  enum class Depth { Quick, Detailed };

  // Empty when the dictionary or the IDS table cannot be read: the caller shows
  // the reason and goes without the route.
  static std::unique_ptr<HanziRecognizer> create(const Paths& paths, QString* error = nullptr);

  ~HanziRecognizer();
  HanziRecognizer(const HanziRecognizer&) = delete;
  HanziRecognizer& operator=(const HanziRecognizer&) = delete;
  HanziRecognizer(HanziRecognizer&&) = delete;
  HanziRecognizer& operator=(HanziRecognizer&&) = delete;

  std::vector<HanziSource> rankStrokes(const Strokes& strokes, int k, Depth depth, const std::function<void(const QString&)>& status = {});

  // For a drawing without a trajectory (a rendered or scanned image): the field
  // and component routes work on the ink, and the component route splits the
  // ink instead of the strokes. This is the reference's --char and --image path
  // and is what its own self test exercises.
  std::vector<HanziSource> rankInk(const Ink& ink, int k, const std::function<void(const QString&)>& status = {});

  // False means the next detailed call builds the templates.
  bool templatesReady() const;

private:
  HanziRecognizer();

  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace hanzi
