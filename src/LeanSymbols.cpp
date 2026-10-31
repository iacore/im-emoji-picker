#include "LeanSymbols.hpp"
#include <QFile>
#include <QTextStream>

namespace {

std::vector<LeanSymbol> loadSymbols() {
  std::vector<LeanSymbol> symbols;

  QFile file(":/res/lean4-symbols.tsv");
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    return symbols;
  }

  QTextStream stream(&file);
  stream.setCodec("UTF-8");

  while (!stream.atEnd()) {
    const QString line = stream.readLine();
    // The header is written as "# <text>"; an abbreviation can be a bare "#"
    // (it inserts ♯), so only the hash-space form is a comment.
    if (line.isEmpty() || line.startsWith("# ")) {
      continue;
    }

    const int tab = line.indexOf('\t');
    if (tab <= 0) {
      continue;
    }

    symbols.push_back({line.left(tab).toStdString(), line.mid(tab + 1).toStdString()});
  }

  return symbols;
}

// The query as it would be written after the leader: `\pi` and `pi` are the
// same abbreviation, and Lean's table is matched case-insensitively by the
// editor's own picker.
QString normalizeQuery(const QString& query) {
  QString result = query.trimmed();
  if (result.startsWith('\\')) {
    result.remove(0, 1);
  }
  return result;
}

} // namespace

const std::vector<LeanSymbol>& leanSymbols() {
  static const std::vector<LeanSymbol> symbols = loadSymbols();
  return symbols;
}

const LeanSymbol* findExactLeanSymbol(const QString& query) {
  const QString wanted = normalizeQuery(query);
  if (wanted.isEmpty()) {
    return nullptr;
  }

  // `\pi` is π and `\Pi` is Π: the table distinguishes case, so an exact
  // spelling wins. A differently-cased spelling only matches when the exact
  // one is absent from the table.
  const LeanSymbol* folded = nullptr;
  for (const LeanSymbol& symbol : leanSymbols()) {
    const QString abbreviation = QString::fromStdString(symbol.abbreviation);
    if (abbreviation == wanted) {
      return &symbol;
    }
    if (!folded && abbreviation.compare(wanted, Qt::CaseInsensitive) == 0) {
      folded = &symbol;
    }
  }

  return folded;
}

void searchLeanSymbols(const QString& query, std::vector<const LeanSymbol*>& found, size_t limit) {
  found.clear();

  const QString wanted = normalizeQuery(query);
  if (wanted.isEmpty()) {
    return;
  }

  // Same rule as the exact lookup: the spelling the author typed comes first,
  // the case-insensitive remainder after it.
  std::vector<const LeanSymbol*> folded;
  for (const LeanSymbol& symbol : leanSymbols()) {
    const QString abbreviation = QString::fromStdString(symbol.abbreviation);
    if (abbreviation.startsWith(wanted)) {
      found.push_back(&symbol);
    } else if (abbreviation.startsWith(wanted, Qt::CaseInsensitive)) {
      folded.push_back(&symbol);
    }
  }

  for (const LeanSymbol* symbol : folded) {
    if (found.size() >= limit) {
      break;
    }
    found.push_back(symbol);
  }

  if (found.size() > limit) {
    found.resize(limit);
  }
}
