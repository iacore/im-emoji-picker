// Development harness: runs the picker window without Fcitx5 or IBus.
//
// The input-method modules activate the window and forward keystrokes by
// pushing commands onto `emojiCommandQueue`; this does the same, and prints
// whatever the picker commits so a script can drive the window (xvfb + xdotool)
// and assert on the result.
//
// Lines on stdin are commands:
//   key <tab|backtab|return|escape|backspace|up|down|left|right|pageup|pagedown|0-9|a-z>
//   quit

#include "EmojiPickerWindow.hpp"

#include <QCoreApplication>
#include <QHash>
#include <QKeyEvent>
#include <QString>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

namespace {

int keyForName(const QString& name, QString* text) {
  static const QHash<QString, int> namedKeys{
    {"tab", Qt::Key_Tab},         {"backtab", Qt::Key_Backtab}, {"return", Qt::Key_Return},     {"escape", Qt::Key_Escape},
    {"backspace", Qt::Key_Backspace}, {"up", Qt::Key_Up},       {"down", Qt::Key_Down},         {"left", Qt::Key_Left},
    {"right", Qt::Key_Right},     {"pageup", Qt::Key_PageUp},   {"pagedown", Qt::Key_PageDown},
  };

  const auto found = namedKeys.find(name);
  if (found != namedKeys.end()) {
    *text = name == "return" ? "\r" : "";
    return found.value();
  }

  if (name.size() == 1) {
    *text = name;
    return name.at(0).unicode();
  }

  return 0;
}

void pushKey(const QString& name) {
  QString text;
  const int key = keyForName(name, &text);
  if (key == 0) {
    std::fprintf(stderr, "unknown key: %s\n", name.toStdString().c_str());
    return;
  }

  QKeyEvent* event = createKeyEventWithUserPreferences(QKeyEvent::KeyPress, key, Qt::NoModifier, text);
  emojiCommandQueue.push(std::make_shared<EmojiCommandProcessKeyEvent>(event, getEmojiActionForQKeyEvent(event)));
}

void readCommands() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream stream{line};
    std::string command;
    stream >> command;

    if (command == "quit") {
      QCoreApplication::quit();
      return;
    }
    if (command == "key") {
      std::string name;
      stream >> name;
      pushKey(QString::fromStdString(name));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Q_UNUSED(argc)
  Q_UNUSED(argv)

  std::thread activator{[]() {
    std::this_thread::sleep_for(std::chrono::milliseconds{400});

    gui_set_active(true);
    emojiCommandQueue.push(std::make_shared<EmojiCommandEnable>([](const std::string& text) {
      std::printf("COMMIT %s\n", text.c_str());
      std::fflush(stdout);
    }));
  }};
  activator.detach();

  std::thread commands{readCommands};
  commands.detach();

  gui_main(0, nullptr);
  return 0;
}
