// What a search query resolves to, without an emoji picker window or an input
// method in the way. Prints each source's answer in the order the picker
// consults it: Lean 4 abbreviations, then (not printed here) emoji names, then
// Unicode character names.
//
//   search-verify pi              # exact and prefix Lean matches, Unicode matches
//   search-verify --time pi       # the same, plus what the Unicode scan costs

#include "LeanSymbols.hpp"
#include "UnicodeSearch.hpp"
#include <QCoreApplication>
#include <chrono>
#include <cstdio>

static void printLean(const QString& query) {
  std::printf("lean  %-24s ", query.toUtf8().constData());

  if (const LeanSymbol* exact = findExactLeanSymbol(query)) {
    std::printf("exact: %s -> %s\n", exact->abbreviation.c_str(), exact->symbol.c_str());
  } else {
    std::printf("no exact match\n");
  }

  std::vector<const LeanSymbol*> prefix;
  searchLeanSymbols(query, prefix, 8);
  for (const LeanSymbol* symbol : prefix) {
    std::printf("      prefix: %s -> %s\n", symbol->abbreviation.c_str(), symbol->symbol.c_str());
  }
}

static void printUnicode(const QString& query, size_t limit, bool time) {
  const auto start = std::chrono::steady_clock::now();
  const std::vector<UnicodeMatch> matches = searchUnicode(query, limit);
  const auto end = std::chrono::steady_clock::now();

  const double millis = std::chrono::duration<double, std::milli>(end - start).count();
  std::printf("unicode %-21s %zu match(es) in %.1f ms\n", query.toUtf8().constData(), matches.size(), millis);

  if (!time) {
    return;
  }

  for (const UnicodeMatch& match : matches) {
    std::printf("      U+%04X %-40s %s\n", match.codePoint, match.name.toUtf8().constData(), QString::fromUcs4(&match.codePoint, 1).toUtf8().constData());
  }
}

int main(int argc, char** argv) {
  QCoreApplication app(argc, argv);
  QCoreApplication::setApplicationName("search-verify");

  bool time = false;
  QStringList queries;
  for (int i = 1; i < argc; ++i) {
    const QString argument = QString::fromLocal8Bit(argv[i]);
    if (argument == "--time") {
      time = true;
    } else {
      queries.push_back(argument);
    }
  }

  if (queries.isEmpty()) {
    std::fprintf(stderr, "usage: search-verify [--time] QUERY...\n");
    return 2;
  }

  std::printf("lean4 table: %zu abbreviations\n", leanSymbols().size());

  for (const QString& query : queries) {
    printLean(query);
    printUnicode(query, 10, time);
  }

  return 0;
}
