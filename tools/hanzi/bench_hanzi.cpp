// Timing harness for the Hanzi routes, kept so the next person can see what a
// change costs instead of guessing.
//
//   hanzi-bench --mode rank                     # one field rank, per refine
//   hanzi-bench --mode detailed --repeats 5     # trajectory + field + composed
//   hanzi-bench --mode build --cache /tmp/x     # a full template build
//   hanzi-bench                                 # all of the above
//
// It reports wall time per repetition, minor and major faults, peak resident
// memory and the machine's load average: this work is memory-bandwidth bound,
// and a loaded machine reports numbers several times worse than an idle one, so
// the load average is part of the result. Instructions and cycles are worth
// reading with `perf stat -e instructions,cycles` around a run, which stays
// comparable when the machine is busy and the wall clock does not.
//
// The drawing comes from --strokes, a pad-coordinate stroke file; tools/hanzi's
// compare_hanzi.py writes one (build/hanzi/drawing.json) when it builds its test
// sample.
//
// `--print` dumps the field rank with its scores instead of timing it. Two
// builds of the engine can then be compared value for value - check a change
// that is meant to be a pure speedup by building this tool from a worktree of
// the revision before it and diffing the two outputs. That is how the
// mask-based scoring was checked: the candidate lists, coverages, distances and
// similarities all have to come out identical, and running a second cache with
// the tool (`--cache`) is part of it, since a format change moves the regions.

#include "hanzi/GlyphTemplates.hpp"
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
#include <QStandardPaths>
#include <QTextStream>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <sys/resource.h>

namespace {

struct Options {
  QString mode = QStringLiteral("all");
  QString strokes;
  QString cache;
  QString dictionary;
  QString ids;
  int k = 10;
  int repeats = 5;
  bool json = false;
};

bool readStrokes(const QString& path, hanzi::Strokes& strokes, QString& error) {
  QFile file{path};
  if (!file.open(QIODevice::ReadOnly)) {
    error = QStringLiteral("cannot read %1").arg(path);
    return false;
  }
  const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
  QJsonValue payload = document.isArray() ? QJsonValue{document.array()} : QJsonValue{document.object()};
  if (payload.isObject()) {
    const QJsonObject object = payload.toObject();
    payload = object.contains(QStringLiteral("strokes")) ? object.value(QStringLiteral("strokes")) : object.value(QStringLiteral("ink"));
  }
  for (const QJsonValue& stroke : payload.toArray()) {
    hanzi::Stroke points;
    for (const QJsonValue& point : stroke.toArray()) {
      const QJsonArray coordinates = point.toArray();
      if (coordinates.size() < 2) {
        continue;
      }
      points.push_back(hanzi::Point{coordinates.at(0).toDouble(), coordinates.at(1).toDouble()});
    }
    if (!points.empty()) {
      strokes.push_back(std::move(points));
    }
  }
  if (strokes.empty()) {
    error = QStringLiteral("%1 holds no strokes").arg(path);
    return false;
  }
  return true;
}

double loadAverage() {
  QFile file{QStringLiteral("/proc/loadavg")};
  if (!file.open(QIODevice::ReadOnly)) {
    return 0.0;
  }
  bool ok = false;
  const double value = QString::fromLatin1(file.read(16)).trimmed().toDouble(&ok);
  return ok ? value : 0.0;
}

long residentPeakKb() {
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;
}

struct Sample {
  std::vector<long long> milliseconds;

  void add(qint64 value) {
    milliseconds.push_back(value);
  }

  long long best() const {
    return milliseconds.empty() ? 0 : *std::min_element(milliseconds.begin(), milliseconds.end());
  }

  double mean() const {
    double total = 0.0;
    for (const long long value : milliseconds) {
      total += static_cast<double>(value);
    }
    return milliseconds.empty() ? 0.0 : total / static_cast<double>(milliseconds.size());
  }
};

void report(const QString& what, const Sample& sample, QTextStream& out) {
  out << QStringLiteral("%1: best %2 ms, mean %3 ms").arg(what, QString::number(sample.best()), QString::number(sample.mean(), 'f', 1)) << "\n";
  out.flush();
}

}  // namespace

int main(int argc, char** argv) {
  if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM") && qEnvironmentVariableIsEmpty("DISPLAY")) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
  }
  QGuiApplication application{argc, argv};
  QCoreApplication::setOrganizationName(QStringLiteral("gazatu.xyz"));
  QCoreApplication::setApplicationName(QStringLiteral("im-emoji-picker"));

  QCommandLineParser parser;
  parser.setApplicationDescription(QStringLiteral("Time the Hanzi handwriting routes."));
  parser.addHelpOption();
  const QCommandLineOption modeOption{QStringLiteral("mode"), QStringLiteral("rank, detailed, build or all"), QStringLiteral("name"), QStringLiteral("all")};
  const QCommandLineOption strokesOption{QStringLiteral("strokes"), QStringLiteral("pad-coordinate stroke JSON"), QStringLiteral("file"), QStringLiteral("build/hanzi/drawing.json")};
  const QCommandLineOption cacheOption{QStringLiteral("cache"), QStringLiteral("template cache directory"), QStringLiteral("dir")};
  const QCommandLineOption dictionaryOption{QStringLiteral("dictionary"), QStringLiteral("Make Me a Hanzi dictionary"), QStringLiteral("file")};
  const QCommandLineOption idsOption{QStringLiteral("ids"), QStringLiteral("ideographic description table"), QStringLiteral("file")};
  const QCommandLineOption kOption{QStringLiteral("k"), QStringLiteral("candidates per route"), QStringLiteral("count"), QStringLiteral("10")};
  const QCommandLineOption repeatsOption{QStringLiteral("repeats"), QStringLiteral("repetitions"), QStringLiteral("count"), QStringLiteral("5")};
  const QCommandLineOption printOption{QStringLiteral("print"), QStringLiteral("print the field rank with its scores instead of timing it")};
  parser.addOption(modeOption);
  parser.addOption(strokesOption);
  parser.addOption(cacheOption);
  parser.addOption(dictionaryOption);
  parser.addOption(idsOption);
  parser.addOption(kOption);
  parser.addOption(repeatsOption);
  parser.addOption(printOption);
  parser.process(application);

  Options options;
  options.mode = parser.value(modeOption);
  options.strokes = parser.value(strokesOption);
  options.cache = parser.value(cacheOption);
  options.dictionary = parser.value(dictionaryOption);
  options.ids = parser.value(idsOption);
  options.k = parser.value(kOption).toInt();
  options.repeats = std::max(1, parser.value(repeatsOption).toInt());

  QTextStream out{stdout};
  out.setCodec("UTF-8");
  QTextStream err{stderr};
  err.setCodec("UTF-8");

  QString directory = QStandardPaths::locate(QStandardPaths::GenericDataLocation, QStringLiteral("im-emoji-picker/hanzi"), QStandardPaths::LocateDirectory);
  const QByteArray fromEnvironment = qgetenv("IM_EMOJI_PICKER_HANZI_DATA");
  if (!fromEnvironment.isEmpty()) {
    directory = QString::fromLocal8Bit(fromEnvironment);
  }
  if (directory.isEmpty() && QFileInfo::exists(QStringLiteral("models/hanzi/hanzi-dictionary.bin"))) {
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

  hanzi::Strokes strokes;
  QString error;
  if (options.mode != QStringLiteral("build")) {
    if (!readStrokes(options.strokes, strokes, error)) {
      err << error << "\n"
          << "tools/hanzi/compare_hanzi.py builds one at build/hanzi/drawing.json, or pass --strokes\n";
      return 2;
    }
  }

  const bool wantsPrint = parser.isSet(printOption);
  const bool wantsRank = !wantsPrint && (options.mode == QStringLiteral("all") || options.mode == QStringLiteral("rank"));
  const bool wantsDetailed = options.mode == QStringLiteral("all") || options.mode == QStringLiteral("detailed");
  const bool wantsBuild = options.mode == QStringLiteral("all") || options.mode == QStringLiteral("build");
  // `all` measures the build last: it needs a cache directory of its own, and a
  // fresh one would otherwise hide the warm numbers above it.
  const QString buildCache = wantsBuild && options.mode == QStringLiteral("all") ? options.cache + QStringLiteral("-bench-build") : options.cache;

  if (wantsPrint) {
    std::unique_ptr<hanzi::GlyphTemplates> templates = hanzi::GlyphTemplates::open(QStringLiteral("gbk"), options.cache, &error);
    if (!templates) {
      err << error << "\n";
      return 2;
    }
    const hanzi::Ink ink = hanzi::inkFromStrokes(strokes);
    out.setRealNumberPrecision(17);
    out << QStringLiteral("templates: %1 characters").arg(templates->size()) << "\n";
    for (const hanzi::TemplateCandidate& candidate : templates->rank(ink, options.k, 2000)) {
      out << QStringLiteral("  %1 %2 %3 %4").arg(candidate.character).arg(candidate.coverage).arg(candidate.distance).arg(candidate.similarity, 0, 'g', 9) << "\n";
    }
    out.flush();
    return 0;
  }

  if (wantsRank || wantsDetailed) {
    std::unique_ptr<hanzi::GlyphTemplates> templates = hanzi::GlyphTemplates::open(QStringLiteral("gbk"), options.cache, &error);
    if (!templates) {
      err << error << "\n"
          << "build the cache first: hanzi-bench --mode build --cache " << options.cache << "\n";
      return 2;
    }
    const hanzi::Ink ink = hanzi::inkFromStrokes(strokes);
    out << QStringLiteral("templates: %1 characters").arg(templates->size()) << "\n";
    out.flush();

    if (wantsRank) {
      struct rusage before {};
      getrusage(RUSAGE_SELF, &before);
      for (const int refine : {2000, 1200, 500}) {
        Sample sample;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
          QElapsedTimer timer;
          timer.start();
          templates->rank(ink, options.k, refine);
          sample.add(timer.elapsed());
        }
        report(QStringLiteral("field rank(refine=%1)").arg(refine), sample, out);
      }
      struct rusage after {};
      getrusage(RUSAGE_SELF, &after);
      out << QStringLiteral("faults: %1 minor, %2 major").arg(after.ru_minflt - before.ru_minflt).arg(after.ru_majflt - before.ru_majflt) << "\n";
      out.flush();
    }

    if (wantsDetailed) {
      hanzi::HanziRecognizer::Paths paths;
      paths.dictionary = options.dictionary;
      paths.ids = options.ids;
      paths.templateCache = options.cache;
      std::unique_ptr<hanzi::HanziRecognizer> recognizer = hanzi::HanziRecognizer::create(paths, &error);
      if (!recognizer) {
        err << error << "\n";
        return 2;
      }
      Sample sample;
      for (int repeat = 0; repeat < options.repeats; ++repeat) {
        QElapsedTimer timer;
        timer.start();
        recognizer->rankStrokes(strokes, options.k, hanzi::HanziRecognizer::Depth::Detailed);
        sample.add(timer.elapsed());
      }
      report(QStringLiteral("detailed (field + composed)"), sample, out);

      Sample quick;
      for (int repeat = 0; repeat < options.repeats * 10; ++repeat) {
        QElapsedTimer timer;
        timer.start();
        recognizer->rankStrokes(strokes, options.k, hanzi::HanziRecognizer::Depth::Quick);
        quick.add(timer.elapsed());
      }
      report(QStringLiteral("quick (trajectory)"), quick, out);
    }
  }

  if (wantsBuild) {
    QElapsedTimer timer;
    timer.start();
    std::unique_ptr<hanzi::GlyphTemplates> built = hanzi::GlyphTemplates::build(QStringLiteral("gbk"), buildCache, hanzi::defaultTemplateFonts(),
                                                                               [&](int done, int total) {
                                                                                 if (done == total) {
                                                                                   err << QStringLiteral("  built %1/%2\r").arg(done).arg(total);
                                                                                 }
                                                                               },
                                                                               &error);
    const qint64 elapsed = timer.elapsed();
    if (!built) {
      err << error << "\n";
      return 2;
    }
    const QFileInfo file{hanzi::GlyphTemplates::cachePath(QStringLiteral("gbk"), buildCache)};
    out << QStringLiteral("build: %1 ms, %2 characters, %3 MB on disk").arg(elapsed).arg(built->size()).arg(file.size() / (1024.0 * 1024.0), 0, 'f', 1) << "\n";
  }

  out << QStringLiteral("peak resident: %1 MB, load average: %2").arg(residentPeakKb() / 1024).arg(loadAverage(), 0, 'f', 2) << "\n";
  out.flush();
  return 0;
}
