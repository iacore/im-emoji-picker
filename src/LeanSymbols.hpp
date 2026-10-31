#pragma once

#include <QString>
#include <string>
#include <vector>

// One row of the Lean 4 unicode input table: the text after the leader
// (the `pi` of `\pi`) and the text the editor inserts for it (π).
struct LeanSymbol {
  std::string abbreviation;
  std::string symbol;
};

// The whole table, read once from :/res/lean4-symbols.tsv (generated from the
// Lean 4 VS Code extension by tools/lean4/update_symbols.py).
const std::vector<LeanSymbol>& leanSymbols();

// The symbol whose abbreviation is exactly `query` (case-insensitively, with or
// without the leading backslash), or nullptr. This is the match that must win
// over every other source: a Lean author typing `pi` wants π, not a face.
const LeanSymbol* findExactLeanSymbol(const QString& query);

// Abbreviations starting with `query`, in table order, at most `limit` of them.
void searchLeanSymbols(const QString& query, std::vector<const LeanSymbol*>& found, size_t limit);
