#include "hanzi/HanziRecognizer.hpp"

#include "hanzi/GlyphTemplates.hpp"
#include "hanzi/IdsTable.hpp"
#include "hanzi/Ink.hpp"
#include "hanzi/StrokeMatcher.hpp"

#include <QFileInfo>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <utility>

namespace {

using hanzi::GlyphTemplates;
using hanzi::HanziCandidate;
using hanzi::HanziSource;
using hanzi::IdsTable;
using hanzi::Ink;
using hanzi::Point;
using hanzi::Stroke;
using hanzi::StrokeDictionary;
using hanzi::Strokes;
using hanzi::TemplateCandidate;

// hanzi_handwriting.py: SPLIT_RATIOS - the left-right / top-bottom cuts the
// component route tries. Too far either way and the split stops being two
// characters; these bracket where real compounds divide.
constexpr double kSplitRatios[] = {0.34, 0.42, 0.5};

// hanzi_handwriting.py: part_candidates() and compose.compose().
constexpr int kPartCandidates = 6;
constexpr int kPartRefine = 1200;
constexpr int kMergedParts = 8;
constexpr double kComposeGeometry = 1.15;

// recognize.py: rank()'s refine, raised to the reference tool's floor of 2000.
constexpr int kFieldRefine = 2000;

// hanzi_handwriting.py: as_scores() - a route's rank becomes a score in (0, 1].
std::vector<std::pair<QString, double>> asScores(const std::vector<QString>& ranked) {
  std::vector<std::pair<QString, double>> scored;
  scored.reserve(ranked.size());
  for (size_t rank = 0; rank < ranked.size(); ++rank) {
    scored.emplace_back(ranked.at(rank), 1.0 / (1.0 + static_cast<double>(rank)));
  }
  return scored;
}

// hanzi_handwriting.py: merge() - per character the best score, best first.
std::vector<std::pair<QString, double>> mergeScored(std::vector<std::pair<QString, double>> candidates, int limit) {
  std::vector<std::pair<QString, double>> best;
  best.reserve(candidates.size());
  for (const std::pair<QString, double>& candidate : candidates) {
    auto found = std::find_if(best.begin(), best.end(), [&](const std::pair<QString, double>& item) {
      return item.first == candidate.first;
    });
    if (found == best.end()) {
      best.push_back(candidate);
    } else if (found->second < candidate.second) {
      found->second = candidate.second;
    }
  }
  std::stable_sort(best.begin(), best.end(), [](const std::pair<QString, double>& left, const std::pair<QString, double>& right) {
    return left.second > right.second;
  });
  if (limit >= 0 && static_cast<int>(best.size()) > limit) {
    best.resize(limit);
  }
  return best;
}

// A character's own ink from a route, in the order the route ranked it.
std::vector<QString> rankedCharacters(const std::vector<TemplateCandidate>& candidates) {
  std::vector<QString> ranked;
  ranked.reserve(candidates.size());
  for (const TemplateCandidate& candidate : candidates) {
    ranked.push_back(candidate.character);
  }
  return ranked;
}

std::vector<QString> rankedCharacters(const std::vector<std::pair<QString, double>>& matched) {
  std::vector<QString> ranked;
  ranked.reserve(matched.size());
  for (const std::pair<QString, double>& item : matched) {
    ranked.push_back(item.first);
  }
  return ranked;
}

// compose.py: split_strokes() - the cut on the pen trajectories, which is exact:
// a stroke belongs to a side when all of its points are on it. When that leaves
// one side empty the cut falls back to "any point of the stroke is on it".
void splitStrokes(const Strokes& strokes, bool vertical, double ratio, Strokes& first, Strokes& second) {
  first.clear();
  second.clear();
  if (strokes.empty()) {
    return;
  }

  double low = 0.0;
  double high = 0.0;
  bool started = false;
  for (const Stroke& stroke : strokes) {
    for (const Point& point : stroke) {
      const double value = vertical ? point.y : point.x;
      if (!started) {
        low = value;
        high = value;
        started = true;
      } else {
        low = std::min(low, value);
        high = std::max(high, value);
      }
    }
  }
  if (!started) {
    return;
  }

  const double cut = low + (high - low) * ratio;
  std::vector<char> belongs(strokes.size(), 0);
  std::vector<char> fallback(strokes.size(), 0);
  bool anyBelongs = false;
  bool everyBelongs = true;
  for (size_t index = 0; index < strokes.size(); ++index) {
    bool allInside = true;
    bool anyInside = false;
    for (const Point& point : strokes.at(index)) {
      const double value = vertical ? point.y : point.x;
      if (value < cut) {
        anyInside = true;
      } else {
        allInside = false;
      }
    }
    // An empty stroke counts as "all inside" and "nothing inside", which is
    // what Python's all() and any() over an empty list do.
    belongs[index] = allInside ? 1 : 0;
    fallback[index] = anyInside ? 1 : 0;
    anyBelongs = anyBelongs || belongs[index] != 0;
    everyBelongs = everyBelongs && belongs[index] != 0;
  }

  if (!anyBelongs || everyBelongs) {
    belongs = fallback;
  }
  for (size_t index = 0; index < strokes.size(); ++index) {
    if (belongs[index] != 0) {
      first.push_back(strokes.at(index));
    } else {
      second.push_back(strokes.at(index));
    }
  }
}

// compose.py: split_ink() - the same cut for a drawing without a trajectory:
// the *ink bounding box* is divided, so each half keeps its own coordinates
// until normalizeInk() puts it back into the standard frame.
void splitInk(const Ink& ink, bool vertical, double ratio, Ink& first, Ink& second) {
  first = Ink{};
  second = Ink{};

  int low = hanzi::kCanvas;
  int high = -1;
  for (int y = 0; y < hanzi::kCanvas; ++y) {
    for (int x = 0; x < hanzi::kCanvas; ++x) {
      if (ink.at(y, x) <= 0.15f) {
        continue;
      }
      const int value = vertical ? y : x;
      low = std::min(low, value);
      high = std::max(high, value);
    }
  }
  if (high < low) {
    return;
  }

  const int cut = static_cast<int>(std::lround(low + (high - low) * ratio));
  for (int y = 0; y < hanzi::kCanvas; ++y) {
    for (int x = 0; x < hanzi::kCanvas; ++x) {
      if (vertical) {
        if (y >= low && y < cut) {
          first.at(y, x) = ink.at(y, x);
        } else if (y >= cut && y < high + 1) {
          second.at(y, x) = ink.at(y, x);
        }
      } else {
        if (x >= low && x < cut) {
          first.at(y, x) = ink.at(y, x);
        } else if (x >= cut && x < high + 1) {
          second.at(y, x) = ink.at(y, x);
        }
      }
    }
  }
}

// The reference skips a split whose half carries almost no ink.
int inkPixels(const Ink& ink) {
  int count = 0;
  for (int y = 0; y < hanzi::kCanvas; ++y) {
    for (int x = 0; x < hanzi::kCanvas; ++x) {
      if (ink.at(y, x) > 0.15f) {
        ++count;
      }
    }
  }
  return count;
}

// hanzi_handwriting.py: part_candidates() - what could this half of the drawing
// be? Every recognizer that has something to say votes, all on the rank-derived
// scale so no single route dominates by units.
std::vector<std::pair<QString, double>> partCandidates(const Strokes* strokes, const Ink& ink, const StrokeDictionary& dictionary, const GlyphTemplates& templates) {
  std::vector<std::pair<QString, double>> candidates;
  if (strokes != nullptr && !strokes->empty()) {
    const std::vector<std::pair<QString, double>> matched = asScores(rankedCharacters(dictionary.match(*strokes, kPartCandidates)));
    candidates.insert(candidates.end(), matched.begin(), matched.end());
  }
  const std::vector<std::pair<QString, double>> field = asScores(rankedCharacters(templates.rank(ink, kPartCandidates, kPartRefine)));
  candidates.insert(candidates.end(), field.begin(), field.end());
  return mergeScored(std::move(candidates), kMergedParts);
}

} // namespace

namespace hanzi {

std::vector<HanziCandidate> mergeSources(const std::vector<HanziSource>& sources, int limit) {
  std::vector<HanziCandidate> merged;
  for (const HanziSource& source : sources) {
    for (size_t rank = 0; rank < source.ranked.size(); ++rank) {
      const QString& character = source.ranked.at(rank);
      const double score = source.weight / (1.0 + static_cast<double>(rank));
      QString origin = source.origin;
      if (rank < source.details.size() && !source.details.at(rank).isEmpty()) {
        origin += QStringLiteral(" ") + source.details.at(rank);
      }
      auto found = std::find_if(merged.begin(), merged.end(), [&](const HanziCandidate& candidate) {
        return candidate.character == character;
      });
      if (found == merged.end()) {
        merged.push_back(HanziCandidate{character, score, origin});
      } else if (found->score < score) {
        *found = HanziCandidate{character, score, origin};
      }
    }
  }

  std::stable_sort(merged.begin(), merged.end(), [](const HanziCandidate& left, const HanziCandidate& right) {
    return left.score > right.score;
  });
  if (limit >= 0 && static_cast<int>(merged.size()) > limit) {
    merged.resize(limit);
  }
  return merged;
}

struct HanziRecognizer::Impl {
  Paths paths;
  std::unique_ptr<StrokeDictionary> dictionary;
  std::unique_ptr<IdsTable> ids;
  std::unique_ptr<GlyphTemplates> templates;
  QString error;
  bool templatesTried = false;

  // Loads the field templates, building them on the first detailed call. The
  // build renders the whole charset from the system's CJK fonts and measures
  // minutes, which is why it is announced through `status` before it starts.
  const GlyphTemplates* fields(const std::function<void(const QString&)>& status) {
    if (templates) {
      return templates.get();
    }
    if (templatesTried) {
      return nullptr;
    }
    templatesTried = true;

    QString reason;
    templates = GlyphTemplates::open(paths.charset, paths.templateCache, &reason);
    if (templates) {
      return templates.get();
    }

    if (status) {
      status(QStringLiteral("Building character templates (one time, a few minutes) ..."));
    }
    QString buildError;
    templates = GlyphTemplates::build(
        paths.charset, paths.templateCache, defaultTemplateFonts(),
        [&](int done, int total) {
          if (status && total > 0) {
            status(QStringLiteral("Building character templates: %1 / %2 characters").arg(done).arg(total));
          }
        },
        &buildError);
    if (!templates) {
      error = buildError.isEmpty() ? reason : buildError;
      return nullptr;
    }
    return templates.get();
  }

  std::vector<HanziSource> composed(const Strokes* strokes, const Ink& ink, int k) {
    const GlyphTemplates* templates = fields(nullptr);
    if (!templates || !dictionary || !ids) {
      return {};
    }

    // The reference keeps the best description per character over every axis
    // and cut it tried, in the order it found them, so a tie goes to the cut
    // that came first - an order the parallelism below has to preserve.
    struct Hit {
      QString character;
      double score = 0.0;
      QString parts;
    };

    struct Cut {
      bool vertical = false;
      QString op;
      double ratio = 0.0;
    };

    std::vector<Cut> cuts;
    const std::pair<bool, QString> axes[] = {{false, QStringLiteral("⿰")}, {true, QStringLiteral("⿱")}};
    for (const std::pair<bool, QString>& axis : axes) {
      for (const double ratio : kSplitRatios) {
        cuts.push_back(Cut{axis.first, axis.second, ratio});
      }
    }

    // A cut is two part recognitions, and a part recognition walks the whole
    // alphabet twice over, so the cuts run in parallel and are merged in order
    // afterwards. Everything they touch is read-only: the templates are a
    // mapping and the part recognizers are const.
    std::vector<std::vector<Hit>> perCut(cuts.size());

    auto runCut = [&](const Cut& cut) {
      std::vector<Hit> found;
      Strokes leftStrokes;
      Strokes rightStrokes;
      Ink leftInk;
      Ink rightInk;

      if (strokes != nullptr) {
        splitStrokes(*strokes, cut.vertical, cut.ratio, leftStrokes, rightStrokes);
        if (leftStrokes.empty() || rightStrokes.empty()) {
          return found;
        }
        leftInk = inkFromStrokes(leftStrokes);
        rightInk = inkFromStrokes(rightStrokes);
      } else {
        Ink first;
        Ink second;
        splitInk(ink, cut.vertical, cut.ratio, first, second);
        if (inkPixels(first) < 8 || inkPixels(second) < 8) {
          return found;
        }
        leftInk = normalizeInk(first);
        rightInk = normalizeInk(second);
      }

      const std::vector<std::pair<QString, double>> left = partCandidates(strokes != nullptr ? &leftStrokes : nullptr, leftInk, *dictionary, *templates);
      const std::vector<std::pair<QString, double>> right = partCandidates(strokes != nullptr ? &rightStrokes : nullptr, rightInk, *dictionary, *templates);
      for (const std::pair<QString, double>& leftCandidate : left) {
        for (const std::pair<QString, double>& rightCandidate : right) {
          const std::vector<QString> described = ids->lookup(cut.op, leftCandidate.first, rightCandidate.first);
          for (const QString& character : described) {
            if (!templates->contains(character)) {
              continue;
            }
            const double score = kComposeGeometry * (leftCandidate.second + rightCandidate.second) / 2.0;
            const QString parts = cut.op + leftCandidate.first + rightCandidate.first;
            auto same = std::find_if(found.begin(), found.end(), [&](const Hit& hit) {
              return hit.character == character;
            });
            if (same == found.end()) {
              found.push_back(Hit{character, score, parts});
            } else if (same->score < score) {
              same->score = score;
              same->parts = parts;
            }
          }
        }
      }
      return found;
    };

    std::atomic<size_t> nextCut{0};
    auto runCuts = [&]() {
      for (size_t index = nextCut.fetch_add(1); index < cuts.size(); index = nextCut.fetch_add(1)) {
        perCut.at(index) = runCut(cuts.at(index));
      }
    };
    const size_t threadCount = std::min(cuts.size(), static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency())));
    {
      std::vector<std::thread> pool;
      pool.reserve(threadCount - 1);
      for (size_t index = 1; index < threadCount; ++index) {
        pool.emplace_back(runCuts);
      }
      runCuts();
      for (std::thread& thread : pool) {
        thread.join();
      }
    }

    std::vector<Hit> hits;
    for (const std::vector<Hit>& cut : perCut) {
      for (const Hit& hit : cut) {
        auto found = std::find_if(hits.begin(), hits.end(), [&](const Hit& existing) {
          return existing.character == hit.character;
        });
        if (found == hits.end()) {
          hits.push_back(hit);
        } else if (found->score < hit.score) {
          *found = hit;
        }
      }
    }

    std::stable_sort(hits.begin(), hits.end(), [](const Hit& left, const Hit& right) {
      return left.score > right.score;
    });

    HanziSource source;
    source.origin = QStringLiteral("composed");
    source.weight = kComposedWeight;
    const int take = std::min<int>(k, static_cast<int>(hits.size()));
    for (int index = 0; index < take; ++index) {
      const Hit& hit = hits.at(index);
      source.ranked.push_back(hit.character);
      source.details.push_back(hit.parts);
    }
    if (source.ranked.empty()) {
      return {};
    }
    return {source};
  }

  std::vector<HanziSource> detailed(const Strokes* strokes, const Ink& ink, int k, const std::function<void(const QString&)>& status) {
    std::vector<HanziSource> sources;
    const GlyphTemplates* templates = fields(status);
    if (templates) {
      HanziSource field;
      field.origin = QStringLiteral("field");
      field.weight = kFieldWeight;
      field.ranked = rankedCharacters(templates->rank(ink, k, kFieldRefine));
      if (!field.ranked.empty()) {
        sources.push_back(field);
      }
    }
    std::vector<HanziSource> composedSources = composed(strokes, ink, k);
    sources.insert(sources.end(), composedSources.begin(), composedSources.end());
    return sources;
  }
};

HanziRecognizer::HanziRecognizer() : _impl{std::make_unique<Impl>()} {
}

HanziRecognizer::~HanziRecognizer() = default;

std::unique_ptr<HanziRecognizer> HanziRecognizer::create(const Paths& paths, QString* error) {
  std::unique_ptr<HanziRecognizer> recognizer{new HanziRecognizer()};
  Impl& impl = *recognizer->_impl;
  impl.paths = paths;

  auto fail = [&](const QString& message, const QString& reason) {
    if (error) {
      *error = reason.isEmpty() ? message : reason;
    }
    return std::unique_ptr<HanziRecognizer>{};
  };

  QString reason;
  impl.dictionary = StrokeDictionary::load(paths.dictionary, &reason);
  if (!impl.dictionary) {
    return fail(QStringLiteral("Cannot read %1: %2").arg(paths.dictionary, reason), reason);
  }
  impl.ids = IdsTable::load(paths.ids, &reason);
  if (!impl.ids) {
    return fail(QStringLiteral("Cannot read %1: %2").arg(paths.ids, reason), reason);
  }
  return recognizer;
}

std::vector<HanziSource> HanziRecognizer::rankStrokes(const Strokes& strokes, int k, Depth depth, const std::function<void(const QString&)>& status) {
  Impl& impl = *_impl;

  std::vector<HanziSource> sources;
  if (strokes.empty()) {
    return sources;
  }

  // The trajectory route reads the drawing itself, so it runs whether or not
  // the templates are there and whether or not the drawing can be split.
  if (impl.dictionary) {
    HanziSource trajectory;
    trajectory.origin = QStringLiteral("trajectory");
    trajectory.weight = kTrajectoryWeight;
    trajectory.ranked = rankedCharacters(impl.dictionary->match(strokes, k));
    if (!trajectory.ranked.empty()) {
      sources.push_back(trajectory);
    }
  }
  if (depth == Depth::Quick) {
    return sources;
  }

  const Ink ink = inkFromStrokes(strokes);
  std::vector<HanziSource> rest = impl.detailed(&strokes, ink, k, status);
  sources.insert(sources.end(), rest.begin(), rest.end());
  return sources;
}

std::vector<HanziSource> HanziRecognizer::rankInk(const Ink& ink, int k, const std::function<void(const QString&)>& status) {
  Impl& impl = *_impl;
  return impl.detailed(nullptr, ink, k, status);
}

bool HanziRecognizer::templatesReady() const {
  const Impl& impl = *_impl;
  if (impl.templates) {
    return true;
  }
  return QFileInfo::exists(GlyphTemplates::cachePath(impl.paths.charset, impl.paths.templateCache));
}

} // namespace hanzi
