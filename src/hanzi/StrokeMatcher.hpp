#pragma once

#include "hanzi/HanziTypes.hpp"

#include <QString>

#include <memory>
#include <utility>
#include <vector>

namespace hanzi {

// The trajectory route: the Make Me a Hanzi stroke matcher
// (juststrokes/mmah-vite/src/mmah.ts, ported to Python in mmah_matcher.py).
// Each stroke is reduced to four sampled points plus an encoded angle and
// length, a candidate must have the same stroke count, and the score is the
// negative L1 distance over the points plus a per-stroke angle penalty.
//
// The dictionary holds 9,574 characters of real pen trajectories and is stored
// already preprocessed, so only the query is preprocessed here. 谞 is not in it
// (it is not in Make Me a Hanzi); its components 讠 and 胥 are, which is what the
// component route uses.
class StrokeDictionary {
public:
  // Reads the compact dictionary written by tools/hanzi/convert_data.py.
  static std::unique_ptr<StrokeDictionary> load(const QString& path, QString* error = nullptr);

  ~StrokeDictionary();
  StrokeDictionary(const StrokeDictionary&) = delete;
  StrokeDictionary& operator=(const StrokeDictionary&) = delete;
  StrokeDictionary(StrokeDictionary&&) = delete;
  StrokeDictionary& operator=(StrokeDictionary&&) = delete;

  int size() const;

  // Best first. Characters of a different stroke count are not candidates -
  // that is the matcher's own filter, not a heuristic. The score is the
  // reference's, so it is negative and only comparable within this route.
  std::vector<std::pair<QString, double>> match(const Strokes& strokes, int k = 10) const;

private:
  StrokeDictionary();

  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace hanzi
