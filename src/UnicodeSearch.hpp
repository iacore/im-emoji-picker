#pragma once

#include <QString>
#include <cstdint>
#include <vector>

// A character that matched a query, together with its Unicode name.
struct UnicodeMatch {
  uint32_t codePoint = 0;
  QString name;
};

// The character's Unicode name ("GREEK SMALL LETTER PI"), or an empty string
// when it has none. Algorithmic names (CJK UNIFIED IDEOGRAPH-4E00) are included.
QString unicodeName(uint32_t codePoint);

// Characters whose name contains `query`, best match first, at most `limit` of
// them. A query that is a code point (U+03C0, 0x3c0, 03c0) also returns that
// character, first.
//
// This is the fallback source: it only runs when neither the Lean table nor an
// emoji matched, so it is allowed to walk the whole character space.
std::vector<UnicodeMatch> searchUnicode(const QString& query, size_t limit);
