// Numeric harness for the Hanzi routes, the rare-character port of
// ~/computing/lib/hanzi-handwriting. It runs the same inputs the Python
// reference was measured with and prints what each route found, in the
// reference's own layout, so tools/hanzi/compare_hanzi.py can line the two up.
//
//   hanzi-verify --strokes /tmp/hw_xu.json              # pad trajectory
//   hanzi-verify --ink-raw /tmp/ink.raw                 # normalized ink dump
//   hanzi-verify --strokes /tmp/hw_xu.json --json       # machine readable
//
// The ink dump is the reference's normalize() output: 128 * 128 float32, little
// endian, row major, 1 = ink. Its --ink option writes the two-times upscaled
// picture of the same thing, which would blur the comparison, so the compare
// script dumps the array itself.

#include "hanzi/HanziRecognizer.hpp"
#include "hanzi/Ink.hpp"

#include <QCommandLineParser>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QGuiApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QStandardPaths>
#include <QStringList>
#include <QTextStream>

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

struct Options {
  QString strokes;
  QString inkRaw;
  QString dictionary;
  QString ids;
  QString cache;
  QString charset = QStringLiteral("gbk");
  int k = 10;
  bool quick = false;
  bool json = false;
  bool timing = false;
};

// hanzi_handwriting.py: strokes_from_json() - [[x, y], ...] per stroke,
// [x0, y0, x1, y1, ...], or [[xs], [ys], [ts]], optionally wrapped in an object
// under "strokes" or "ink".
bool readStrokes(const QString& path, hanzi::Strokes& strokes, QString& error) {
  QFile file{path};
  if (!file.open(QIODevice::ReadOnly)) {
    error = QStringLiteral("cannot read %1").arg(path);
    return false;
  }

  QJsonParseError parseError;
  const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
  if (parseError.error != QJsonParseError::NoError) {
    error = QStringLiteral("%1: %2").arg(path, parseError.errorString());
    return false;
  }

  QJsonValue payload = document.isArray() ? QJsonValue{document.array()} : QJsonValue{document.object()};
  if (payload.isObject()) {
    const QJsonObject object = payload.toObject();
    payload = object.contains(QStringLiteral("strokes")) ? object.value(QStringLiteral("strokes")) : object.value(QStringLiteral("ink"));
  }
  if (!payload.isArray()) {
    error = QStringLiteral("%1 holds no strokes").arg(path);
    return false;
  }

  const QJsonArray array = payload.toArray();
  for (const QJsonValue& value : array) {
    if (!value.isArray()) {
      continue;
    }
    const QJsonArray points = value.toArray();
    if (points.isEmpty()) {
      continue;
    }

    hanzi::Stroke stroke;
    const QJsonValue first = points.at(0);
    if (first.isArray()) {
      for (const QJsonValue& point : points) {
        const QJsonArray coordinates = point.toArray();
        if (coordinates.size() < 2) {
          continue;
        }
        stroke.push_back(hanzi::Point{coordinates.at(0).toDouble(), coordinates.at(1).toDouble()});
      }
    } else if (points.size() == 3 && points.at(0).isArray() && points.at(1).isArray()) {
      const QJsonArray xs = points.at(0).toArray();
      const QJsonArray ys = points.at(1).toArray();
      const int count = std::min(xs.size(), ys.size());
      for (int index = 0; index < count; ++index) {
        stroke.push_back(hanzi::Point{xs.at(index).toDouble(), ys.at(index).toDouble()});
      }
    } else {
      for (int index = 0; index + 1 < points.size(); index += 2) {
        stroke.push_back(hanzi::Point{points.at(index).toDouble(), points.at(index + 1).toDouble()});
      }
    }
    if (!stroke.empty()) {
      strokes.push_back(std::move(stroke));
    }
  }

  if (strokes.empty()) {
    error = QStringLiteral("%1 holds no strokes").arg(path);
    return false;
  }
  return true;
}

bool readInk(const QString& path, hanzi::Ink& ink, QString& error) {
  QFile file{path};
  if (!file.open(QIODevice::ReadOnly)) {
    error = QStringLiteral("cannot read %1").arg(path);
    return false;
  }
  const QByteArray bytes = file.readAll();
  const qint64 expected = static_cast<qint64>(hanzi::kCanvasPixels) * static_cast<qint64>(sizeof(float));
  if (bytes.size() != expected) {
    error = QStringLiteral("%1 holds %2 bytes, expected %3 (128x128 float32)").arg(path).arg(bytes.size()).arg(expected);
    return false;
  }
  std::memcpy(ink.data(), bytes.constData(), static_cast<size_t>(expected));
  return true;
}

QString describeRoutes(const std::vector<hanzi::HanziSource>& sources, const std::vector<QString>& order, const std::vector<QString>& labels, int k) {
  QStringList lines;
  for (size_t index = 0; index < order.size(); ++index) {
    const QString& origin = order.at(index);
    QStringList found;
    for (const hanzi::HanziSource& source : sources) {
      if (source.origin != origin) {
        continue;
      }
      const int take = std::min<int>(k, static_cast<int>(source.ranked.size()));
      for (int rank = 0; rank < take; ++rank) {
        QString entry = source.ranked.at(rank);
        if (rank < source.details.size() && !source.details.at(rank).isEmpty()) {
          entry += QStringLiteral("[") + source.details.at(rank) + QStringLiteral("]");
        }
        found << entry;
      }
    }
    lines << QStringLiteral("%1: %2").arg(labels.at(index), found.join(QLatin1Char(' ')));
  }
  return lines.join(QLatin1Char('\n'));
}

QString toJson(const std::vector<hanzi::HanziSource>& sources) {
  QJsonObject object;
  for (const hanzi::HanziSource& source : sources) {
    QJsonArray entries;
    for (size_t rank = 0; rank < source.ranked.size(); ++rank) {
      QJsonObject entry;
      entry.insert(QStringLiteral("char"), source.ranked.at(rank));
      if (rank < source.details.size()) {
        entry.insert(QStringLiteral("parts"), source.details.at(rank));
      }
      entries.append(entry);
    }
    object.insert(source.origin, entries);
  }
  return QString::fromUtf8(QJsonDocument{object}.toJson(QJsonDocument::Indented));
}

} // namespace

int main(int argc, char** argv) {
  // Rendering the templates goes through QFont and QPainter, which need a
  // QGuiApplication; with no display, Qt's offscreen platform provides one.
  if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM") && qEnvironmentVariableIsEmpty("DISPLAY")) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
  }
  QGuiApplication application{argc, argv};
  // The picker's own identity, so the harness warms and reads the same template
  // cache the addon uses instead of a second copy under its own name.
  QCoreApplication::setOrganizationName(QStringLiteral("gazatu.xyz"));
  QCoreApplication::setApplicationName(QStringLiteral("im-emoji-picker"));

  QCommandLineParser parser;
  parser.setApplicationDescription(QStringLiteral("Run the Hanzi handwriting routes on one drawing and print what each route found."));
  parser.addHelpOption();
  const QCommandLineOption strokesOption{QStringLiteral("strokes"), QStringLiteral("pen trajectory JSON, the reference's --strokes"), QStringLiteral("file")};
  const QCommandLineOption inkOption{QStringLiteral("ink-raw"), QStringLiteral("normalized ink: 128x128 float32, the reference's normalize() output"), QStringLiteral("file")};
  const QCommandLineOption dictionaryOption{QStringLiteral("dictionary"), QStringLiteral("Make Me a Hanzi dictionary"), QStringLiteral("file")};
  const QCommandLineOption idsOption{QStringLiteral("ids"), QStringLiteral("ideographic description table"), QStringLiteral("file")};
  const QCommandLineOption cacheOption{QStringLiteral("cache"), QStringLiteral("template cache directory"), QStringLiteral("dir")};
  const QCommandLineOption charsetOption{QStringLiteral("charset"), QStringLiteral("field alphabet: gb2312, gbk, gb18030"), QStringLiteral("name"), QStringLiteral("gbk")};
  const QCommandLineOption kOption{QStringLiteral("k"), QStringLiteral("candidates per route"), QStringLiteral("count"), QStringLiteral("10")};
  const QCommandLineOption quickOption{QStringLiteral("quick"), QStringLiteral("trajectory route only")};
  const QCommandLineOption jsonOption{QStringLiteral("json"), QStringLiteral("machine readable")};
  const QCommandLineOption timingOption{QStringLiteral("time"), QStringLiteral("report milliseconds per route")};
  parser.addOption(strokesOption);
  parser.addOption(inkOption);
  parser.addOption(dictionaryOption);
  parser.addOption(idsOption);
  parser.addOption(cacheOption);
  parser.addOption(charsetOption);
  parser.addOption(kOption);
  parser.addOption(quickOption);
  parser.addOption(jsonOption);
  parser.addOption(timingOption);
  parser.process(application);

  Options options;
  options.strokes = parser.value(strokesOption);
  options.inkRaw = parser.value(inkOption);
  options.dictionary = parser.value(dictionaryOption);
  options.ids = parser.value(idsOption);
  options.cache = parser.value(cacheOption);
  options.charset = parser.value(charsetOption);
  options.k = parser.value(kOption).toInt();
  options.quick = parser.isSet(quickOption);
  options.json = parser.isSet(jsonOption);
  options.timing = parser.isSet(timingOption);

  QTextStream out{stdout};
  out.setCodec("UTF-8");
  QTextStream err{stderr};
  err.setCodec("UTF-8");

  if (options.strokes.isEmpty() == options.inkRaw.isEmpty()) {
    err << "give exactly one of --strokes or --ink-raw\n";
    return 2;
  }

  QString directory = QStandardPaths::locate(QStandardPaths::GenericDataLocation, QStringLiteral("im-emoji-picker/hanzi"), QStandardPaths::LocateDirectory);
  const QByteArray fromEnvironment = qgetenv("IM_EMOJI_PICKER_HANZI_DATA");
  if (!fromEnvironment.isEmpty()) {
    directory = QString::fromLocal8Bit(fromEnvironment);
  }
  if (directory.isEmpty() && QFileInfo::exists(QStringLiteral("models/hanzi/hanzi-dictionary.bin"))) {
    // Running from a checkout: tools/hanzi wrote the data here.
    directory = QStringLiteral("models/hanzi");
  }
  if (options.dictionary.isEmpty()) {
    options.dictionary = directory + QStringLiteral("/hanzi-dictionary.bin");
  }
  if (options.ids.isEmpty()) {
    options.ids = directory + QStringLiteral("/hanzi-ids.bin");
  }
  if (options.cache.isEmpty()) {
    options.cache = QStandardPaths::writableLocation(QStandardPaths::CacheLocation) + QStringLiteral("/hanzi-templates");
  }

  hanzi::HanziRecognizer::Paths paths;
  paths.dictionary = options.dictionary;
  paths.ids = options.ids;
  paths.templateCache = options.cache;
  paths.charset = options.charset;

  QString error;
  std::unique_ptr<hanzi::HanziRecognizer> recognizer = hanzi::HanziRecognizer::create(paths, &error);
  if (!recognizer) {
    err << error << "\n";
    return 2;
  }

  hanzi::Strokes strokes;
  hanzi::Ink ink;
  QString description;
  if (!options.strokes.isEmpty()) {
    if (!readStrokes(options.strokes, strokes, error)) {
      err << error << "\n";
      return 2;
    }
    description = QStringLiteral("%1 (%2 strokes)").arg(options.strokes).arg(strokes.size());
  } else {
    if (!readInk(options.inkRaw, ink, error)) {
      err << error << "\n";
      return 2;
    }
    description = QStringLiteral("%1 (normalized ink)").arg(options.inkRaw);
  }

  auto status = [&](const QString& text) {
    err << text << "\n";
    err.flush();
  };

  QElapsedTimer timer;
  timer.start();
  std::vector<hanzi::HanziSource> sources;
  if (!options.strokes.isEmpty()) {
    sources = recognizer->rankStrokes(strokes, options.k, options.quick ? hanzi::HanziRecognizer::Depth::Quick : hanzi::HanziRecognizer::Depth::Detailed, status);
  } else {
    sources = recognizer->rankInk(ink, options.k, status);
  }
  const qint64 elapsed = timer.elapsed();

  if (options.json) {
    QJsonObject object;
    object.insert(QStringLiteral("input"), description);
    object.insert(QStringLiteral("routes"), QJsonDocument::fromJson(toJson(sources).toUtf8()).object());
    if (options.timing) {
      object.insert(QStringLiteral("milliseconds"), static_cast<double>(elapsed));
    }
    out << QString::fromUtf8(QJsonDocument{object}.toJson(QJsonDocument::Indented));
  } else {
    out << QStringLiteral("input     : ") << description << "\n";
    // The reference prints the routes in this order; an empty one is still worth
    // a line, because that is what "this route had nothing to say" looks like.
    out << describeRoutes(sources, {QStringLiteral("trajectory"), QStringLiteral("field"), QStringLiteral("composed")}, {QStringLiteral("trajectory"), QStringLiteral("field     "), QStringLiteral("composed  ")}, options.k) << "\n";
    if (options.timing) {
      out << QStringLiteral("time      : ") << elapsed << QStringLiteral(" ms") << "\n";
    }
  }
  out.flush();
  return 0;
}
