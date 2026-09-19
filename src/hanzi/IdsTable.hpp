#pragma once

#include <QString>

#include <memory>
#include <vector>

// compose.py: the Unicode ideographic description (IDS) table, restricted to
// the two-part operators the component route uses, ⿰ (left-right) and ⿱
// (top-bottom). It names the character from its parts: ⿰讠胥 is 谞, ⿰讠胃 is 谓.
namespace hanzi {

class IdsTable {
public:
  // Reads the compact table written by tools/hanzi/convert_data.py. It holds
  // the characters of a two-part description whose own code point is in GBK,
  // which is what the component route can name; a description with a part
  // outside the BMP is not in it, because parts are stored as uint16 and every
  // alphabet the routes recognize a part from is in the BMP.
  static std::unique_ptr<IdsTable> load(const QString& path, QString* error = nullptr);

  ~IdsTable();
  IdsTable(const IdsTable&) = delete;
  IdsTable& operator=(const IdsTable&) = delete;
  IdsTable(IdsTable&&) = delete;
  IdsTable& operator=(IdsTable&&) = delete;

  // How many characters are described by a two-part description.
  int size() const;

  // The characters `op` (`⿰` or `⿱`) applied to `left` and `right` describes,
  // in codepoint order. Empty when the description is not in the table.
  std::vector<QString> lookup(const QString& op, const QString& left, const QString& right) const;

private:
  IdsTable();

  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace hanzi
