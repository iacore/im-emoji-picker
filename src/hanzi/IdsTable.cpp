#include "hanzi/IdsTable.hpp"

#include <QFile>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace hanzi {
namespace {

// compose.py: OPERATORS = ("⿰", "⿱"), left-right then top-bottom. The file
// stores the index of that tuple, because the operator is a code point of its
// own and nothing else is indexed.
constexpr uint16_t kOperatorLeftRight = 0x2FF0;
constexpr uint16_t kOperatorTopBottom = 0x2FF1;

constexpr char kMagic[4] = {'H', 'Z', 'I', 'D'};
constexpr uint32_t kVersion = 1;

// The operator of a query, or -1 when it is not one of the two the component
// route uses. Both are single BMP code points, so one UTF-16 unit is enough.
int operatorIndex(const QString& op) {
  if (op.size() != 1) {
    return -1;
  }
  switch (op.at(0).unicode()) {
  case kOperatorLeftRight:
    return 0;
  case kOperatorTopBottom:
    return 1;
  default:
    return -1;
  }
}

// The first code point of a part, or -1 when the part is empty or outside the
// BMP. The table's parts are all BMP - the file stores them as uint16 - and
// every alphabet the routes recognize parts from is BMP as well, so a query
// part beyond the BMP can only be one this table does not describe.
int32_t firstCodePoint(const QString& text) {
  if (text.isEmpty()) {
    return -1;
  }
  const ushort unit = text.at(0).unicode();
  if (unit >= 0xD800 && unit <= 0xDBFF) {
    return -1;
  }
  return unit;
}

// Little-endian reader over the whole table file; every read is bounds checked
// so a truncated or foreign file is a message, never a fault.
class Reader {
public:
  Reader(const uchar* data, qsizetype size) : _data{data}, _size{size} {
  }

  bool readU8(uint8_t& out) {
    if (_position >= _size) {
      return false;
    }
    out = _data[_position++];
    return true;
  }

  bool readU16(uint16_t& out) {
    if (_size - _position < 2) {
      return false;
    }
    out = static_cast<uint16_t>(_data[_position] | (_data[_position + 1] << 8));
    _position += 2;
    return true;
  }

  bool readU32(uint32_t& out) {
    if (_size - _position < 4) {
      return false;
    }
    out = static_cast<uint32_t>(_data[_position]) | (static_cast<uint32_t>(_data[_position + 1]) << 8) | (static_cast<uint32_t>(_data[_position + 2]) << 16) | (static_cast<uint32_t>(_data[_position + 3]) << 24);
    _position += 4;
    return true;
  }

  bool atEnd() const {
    return _position == _size;
  }

private:
  const uchar* _data;
  qsizetype _size;
  qsizetype _position = 0;
};

} // namespace

struct IdsTable::Impl {
  // One (description, character) pair, sorted by description and then by code
  // point, so a description's characters are one contiguous run in code point
  // order. The converter has already dropped every character the routes cannot
  // ask about, so lookup() itself does not filter.
  struct Record {
    uint16_t op;
    uint16_t left;
    uint16_t right;
    uint32_t codePoint;
  };

  struct Description {
    uint16_t op;
    uint16_t left;
    uint16_t right;
  };

  std::vector<Record> records;

  static bool beforeDescription(const Record& record, const Description& wanted) {
    if (record.op != wanted.op) {
      return record.op < wanted.op;
    }
    if (record.left != wanted.left) {
      return record.left < wanted.left;
    }
    return record.right < wanted.right;
  }

  static bool byDescription(const Record& a, const Record& b) {
    if (a.op != b.op) {
      return a.op < b.op;
    }
    if (a.left != b.left) {
      return a.left < b.left;
    }
    if (a.right != b.right) {
      return a.right < b.right;
    }
    return a.codePoint < b.codePoint;
  }
};

IdsTable::IdsTable() : _impl{std::make_unique<Impl>()} {
}

IdsTable::~IdsTable() = default;

int IdsTable::size() const {
  return static_cast<int>(_impl->records.size());
}

std::unique_ptr<IdsTable> IdsTable::load(const QString& path, QString* error) {
  auto fail = [error](const QString& message) {
    if (error) {
      *error = message;
    }
    return std::unique_ptr<IdsTable>{};
  };

  QFile file(path);
  if (!file.open(QIODevice::ReadOnly)) {
    return fail(QStringLiteral("could not open the IDS table %1").arg(path));
  }
  const QByteArray bytes = file.readAll();
  Reader reader{reinterpret_cast<const uchar*>(bytes.constData()), bytes.size()};

  uint8_t header[8] = {};
  for (uint8_t& byte : header) {
    if (!reader.readU8(byte)) {
      return fail(QStringLiteral("%1 is not an IDS table: it ends in the header").arg(path));
    }
  }
  if (std::memcmp(header, kMagic, sizeof(kMagic)) != 0) {
    return fail(QStringLiteral("%1 is not an IDS table: bad magic").arg(path));
  }
  uint32_t version = 0;
  std::memcpy(&version, header + 4, sizeof(version));
  if (version != kVersion) {
    return fail(QStringLiteral("%1 has IDS table version %2, expected %3").arg(path).arg(version).arg(kVersion));
  }

  std::unique_ptr<IdsTable> table{new IdsTable()};
  Impl& impl = *table->_impl;
  uint32_t count = 0;
  if (!reader.readU32(count)) {
    return fail(QStringLiteral("%1 is truncated: no record count").arg(path));
  }
  impl.records.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t op = 0;
    uint16_t left = 0;
    uint16_t right = 0;
    uint32_t codePoint = 0;
    if (!reader.readU8(op) || !reader.readU16(left) || !reader.readU16(right) || !reader.readU32(codePoint)) {
      return fail(QStringLiteral("%1 is truncated in record %2").arg(path).arg(i));
    }
    if (op > 1) {
      return fail(QStringLiteral("%1 has operator %2 in record %3, only ⿰ and ⿱ are indexed").arg(path).arg(op).arg(i));
    }
    impl.records.push_back({op, left, right, codePoint});
  }
  if (!reader.atEnd()) {
    return fail(QStringLiteral("%1 has trailing data").arg(path));
  }

  std::sort(impl.records.begin(), impl.records.end(), Impl::byDescription);
  return table;
}

std::vector<QString> IdsTable::lookup(const QString& op, const QString& left, const QString& right) const {
  std::vector<QString> characters;
  const int operatorValue = operatorIndex(op);
  const int32_t leftCodePoint = firstCodePoint(left);
  const int32_t rightCodePoint = firstCodePoint(right);
  if (operatorValue < 0 || leftCodePoint < 0 || rightCodePoint < 0) {
    return characters;
  }
  const Impl::Description wanted{static_cast<uint16_t>(operatorValue), static_cast<uint16_t>(leftCodePoint), static_cast<uint16_t>(rightCodePoint)};

  // Records are sorted by description, so this description's characters are one
  // contiguous run.
  const auto begin = std::lower_bound(_impl->records.begin(), _impl->records.end(), wanted, [](const Impl::Record& record, const Impl::Description& key) {
    return Impl::beforeDescription(record, key);
  });
  for (auto record = begin; record != _impl->records.end(); ++record) {
    if (record->op != wanted.op || record->left != wanted.left || record->right != wanted.right) {
      break;
    }
    const uint codePoint = record->codePoint;
    characters.emplace_back(QString::fromUcs4(&codePoint, 1));
  }
  return characters;
}

} // namespace hanzi
