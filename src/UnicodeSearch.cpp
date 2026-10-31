#include "UnicodeSearch.hpp"
#include <algorithm>
#include <unicode/uchar.h>

namespace {

constexpr uint32_t kLastCodePoint = 0x10FFFF;

// Categories that are never a character worth inserting: unassigned, private
// use, surrogates, and the invisible ones.
bool isPrintableCategory(int8_t category) {
  switch (category) {
  case U_UNASSIGNED:
  case U_PRIVATE_USE_CHAR:
  case U_SURROGATE:
  case U_CONTROL_CHAR:
  case U_FORMAT_CHAR:
  case U_LINE_SEPARATOR:
  case U_PARAGRAPH_SEPARATOR:
    return false;

  default:
    return true;
  }
}

// U+03C0, u+3c0, 0x3c0, and a bare 4..6 digit hex number that carries at least
// one digit (so that a word such as "beef" stays a name query).
bool parseCodePoint(const QString& query, uint32_t& codePoint) {
  QString digits = query.trimmed();

  if (digits.startsWith("U+", Qt::CaseInsensitive) || digits.startsWith("0x", Qt::CaseInsensitive)) {
    digits.remove(0, 2);
  } else {
    if (digits.size() < 4 || digits.size() > 6) {
      return false;
    }

    bool hasDigit = false;
    for (const QChar c : digits) {
      hasDigit = hasDigit || c.isDigit();
    }
    if (!hasDigit) {
      return false;
    }
  }

  if (digits.isEmpty() || digits.size() > 6) {
    return false;
  }

  bool ok = false;
  const uint value = digits.toUInt(&ok, 16);
  if (!ok || value > kLastCodePoint) {
    return false;
  }

  codePoint = value;
  return true;
}

// 0 = the name is the query, 1 = it starts with it, 2 = a word inside it starts
// with it, 3 = it merely contains it. -1 = no match.
int matchRank(const QString& name, const QString& query) {
  if (name == query) {
    return 0;
  }

  if (name.startsWith(query)) {
    return 1;
  }

  // One- and two-letter queries would otherwise match most of the character
  // space through their middle, which is noise, not a search.
  if (query.size() < 3) {
    return -1;
  }

  int index = name.indexOf(query, 1);
  while (index > 0) {
    if (name.at(index - 1) == ' ') {
      return 2;
    }
    index = name.indexOf(query, index + 1);
  }

  return name.contains(query) ? 3 : -1;
}

} // namespace

QString unicodeName(uint32_t codePoint) {
  if (codePoint > kLastCodePoint) {
    return QString();
  }

  char buffer[128];
  UErrorCode status = U_ZERO_ERROR;
  const int32_t length = u_charName(static_cast<UChar32>(codePoint), U_UNICODE_CHAR_NAME, buffer, static_cast<int32_t>(sizeof(buffer)), &status);

  if (U_FAILURE(status) || length <= 0) {
    return QString();
  }

  return QString::fromUtf8(buffer, length);
}

namespace {

struct IndexEntry {
  uint32_t codePoint;
  QString name;
};

// Every named, printable character, built once. Formatting a name through ICU
// costs far more than comparing it, and a picker searches on every keystroke,
// so the names are formatted once and then only compared.
const std::vector<IndexEntry>& characterIndex() {
  static const std::vector<IndexEntry> index = [] {
    std::vector<IndexEntry> entries;
    entries.reserve(160000);

    for (uint32_t codePoint = 0; codePoint <= kLastCodePoint; ++codePoint) {
      if (!isPrintableCategory(u_charType(static_cast<UChar32>(codePoint)))) {
        continue;
      }

      QString name = unicodeName(codePoint);
      if (name.isEmpty()) {
        continue;
      }

      entries.push_back({codePoint, std::move(name)});
    }

    return entries;
  }();

  return index;
}

} // namespace

std::vector<UnicodeMatch> searchUnicode(const QString& query, size_t limit) {
  std::vector<UnicodeMatch> matches;

  const QString wanted = query.trimmed().toUpper();
  if (wanted.isEmpty()) {
    return matches;
  }

  struct RankedMatch {
    int rank;
    uint32_t codePoint;
    QString name;
  };
  std::vector<RankedMatch> ranked;

  uint32_t exactCodePoint = 0;
  if (parseCodePoint(query, exactCodePoint) && isPrintableCategory(u_charType(static_cast<UChar32>(exactCodePoint)))) {
    ranked.push_back({0, exactCodePoint, unicodeName(exactCodePoint)});
  }

  for (const IndexEntry& entry : characterIndex()) {
    const int rank = matchRank(entry.name, wanted);
    if (rank < 0) {
      continue;
    }

    if (rank == 0 && entry.codePoint == exactCodePoint) {
      continue;
    }

    ranked.push_back({rank, entry.codePoint, entry.name});
  }

  std::stable_sort(ranked.begin(), ranked.end(), [](const RankedMatch& a, const RankedMatch& b) {
    if (a.rank != b.rank) {
      return a.rank < b.rank;
    }
    return a.codePoint < b.codePoint;
  });

  for (const RankedMatch& match : ranked) {
    if (matches.size() >= limit) {
      break;
    }
    matches.push_back({match.codePoint, match.name});
  }

  return matches;
}
