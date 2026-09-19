# I'm Emoji Picker

an emoji picker compatible with linux systems using either XServer or Wayland with either Fcitx5 or IBus.
You could also say that the **im** in `im-emoji-picker` stands for **input method** 😉.
I'm basically using the same systems some people use to type Japanese or Chinese characters with a western keyboard.

This project took heavy inspiration from the "predecessor" [https://github.com/GaZaTu/x11-emoji-picker](https://github.com/GaZaTu/x11-emoji-picker).

## Screenshots 😮

*Recently used emoji (EndeavourOS with KDE (XServer) and Fcitx5):*

![docs/screenshot2.png](docs/screenshot2.png)

*System emoji font support (openSUSE 15.1 with KDE (XServer) and IBus):*

![docs/screenshot1.png](docs/screenshot1.png)

*Kaomoji support (EndeavourOS with KDE (XServer) and Fcitx5):*

![docs/screenshot3.png](docs/screenshot3.png)

## Motivation 🤔

### Original Motivation ([x11-emoji-picker](https://github.com/GaZaTu/x11-emoji-picker#motivation-))

I switched from Windows 10 to Linux at work and missed filling my emails with emojis. ~~(the KDE version we use doesn't have the builtin emoji picker yet)~~ (turns out that the KDE emoji picker only copies emojis to clipboard anyway so yea fuck that)

### Reason For The New Emoji Picker

The issue with *x11-emoji-picker* is `xdotool` as the success rate for it to work is a lottery. Sometimes the target window requires focus to accept input; Sometimes it doesn't support `XText` so you need to use some hacky clipboard workaround; And sometimes it doesn't work at all. All in all a frustrating experience.

Another issue is with the `x11...` in the old emoji picker. I won't switch to Wayland anytime soon but having support for it in my emoji picker would've been nice so more people can use it.

## Installation 😉

Download the [install.sh](install.sh) and run it. It downloads and installs either a `.deb` (**Ubuntu**, **Debian**) or a `.rpm` (**openSUSE**, **Fedora**).

- Terminal: `wget -q https://raw.githubusercontent.com/GaZaTu/im-emoji-picker/master/install.sh && sh install.sh`
- Specifying IMF: `sh install.sh -f fcitx5` or `sh install.sh -f ibus`
- Installing nightly build: `sh install.sh -r tags/nightly-build`

Otherwise look at the following options:

**"I use Arch btw"**:
run `yay -S fcitx5-im-emoji-picker-git` or `yay -S ibus-im-emoji-picker-git`.

- repo (fcitx5): [https://aur.archlinux.org/packages/fcitx5-im-emoji-picker-git](https://aur.archlinux.org/packages/fcitx5-im-emoji-picker-git)
- repo (ibus): [https://aur.archlinux.org/packages/ibus-im-emoji-picker-git](https://aur.archlinux.org/packages/ibus-im-emoji-picker-git)
- yay: [https://github.com/Jguer/yay#installation](https://github.com/Jguer/yay#installation)
- manjaro: [https://wiki.manjaro.org/index.php/Arch_User_Repository](https://wiki.manjaro.org/index.php/Arch_User_Repository)

**Debian**:
Download a `-debian-*.deb` from [/releases](https://github.com/GaZaTu/im-emoji-picker/releases) and install it.

- Terminal: `sudo apt install ./im-emoji-picker-*.deb`

**Ubuntu**:
Download a `-ubuntu-*.deb` from [/releases](https://github.com/GaZaTu/im-emoji-picker/releases) and install it.

- Terminal: `sudo apt install ./im-emoji-picker-*.deb`

**openSUSE**:
Download a `-opensuse-*.rpm` from [/releases](https://github.com/GaZaTu/im-emoji-picker/releases) and install it.

- Terminal: `sudo zypper install ./im-emoji-picker-*.rpm`

**Fedora**:
Download a `-fedora-*.rpm` from [/releases](https://github.com/GaZaTu/im-emoji-picker/releases) and install it.

- Terminal: `sudo dnf install ./im-emoji-picker-*.rpm`

*Note: There are [nightly releases](https://github.com/GaZaTu/im-emoji-picker/releases/tag/nightly-build) aswell which are rebuilt on every push to master*

## Handwriting (Hanzi) ✍

Press `Tab` until the status bar highlights `✍` and the picker switches to the handwriting view: a square pad where you write one character with the mouse or a stylus, and a row with the ten most likely characters.

| key | action |
| --- | --- |
| `Tab` / `Shift+Tab` | cycle the views (favorites, emoji list, kaomoji, handwriting) |
| `1`-`9` | insert that candidate |
| `←` `↑` `↓` `→` | move the highlighted candidate |
| `Enter` | insert the highlighted candidate |
| `Backspace` | undo the last stroke |
| `Ctrl+Backspace` | clear the pad |
| `Esc` | close the picker |

Recognition is two recognizers over one drawing. A 3755-class MobileNetV2 over the GB2312 level-1
set, running on [ggml](https://github.com/ggml-org/ggml), answers in about 15 ms: CPU only, loaded
on first use. The same drawing also goes through three routes ported from the
[hanzi-handwriting](https://git.sr.ht/~iacore/hanzi-handwriting) matcher, which are what reach the
characters that classifier has no label for. Only Hanzi are covered so far; emoji and kaomoji are
untouched.

### Rare characters

The classifier's alphabet is GB2312 level 1: 3,755 characters. One outside it - 谞, for instance -
can never come out of it, however well it is written, because the model folds it into the nearest
shape it knows (谓). The other three routes work on the character set instead of a training set:

| route | what it matches | alphabet | where it runs |
| --- | --- | --- | --- |
| `trajectory` | the pen trajectory itself | 9,574 Make Me a Hanzi characters | GUI thread, microseconds |
| `field` | thinned centre lines of the ink | every character of a charset (GBK: 20,902) | worker thread |
| `composed` | the parts of the drawing, then a lookup by Unicode ideographic description (谞 = ⿰讠胥) | the whole charset through its parts | worker thread |

The trajectory route answers while you write; the other two walk the whole alphabet, so they run on
a worker thread and fill the row a moment later. Hovering a candidate says where it came from
(`model`, `trajectory`, `field`, or `composed ⿰讠胥`).

The two routes that need data read two files, installed next to the model:

- `<prefix>/share/im-emoji-picker/hanzi/hanzi-dictionary.bin` - the Make Me a Hanzi medians
- `<prefix>/share/im-emoji-picker/hanzi/hanzi-ids.bin` - the CJKVI ideographic description table

To use another directory, set the `handwritingDataPath` setting or the `IM_EMOJI_PICKER_HANZI_DATA`
environment variable. Both files are generated by `tools/hanzi` from upstream sources, which is
what the PKGBUILD does:

```sh
python3 tools/hanzi/fetch_data.py --out models/hanzi
python3 tools/hanzi/convert_data.py --graphics models/hanzi/graphics.txt --ids models/hanzi/ids.txt --out models/hanzi
```

The field route renders every character of the charset from the system's CJK fonts and reduces it
to a thinned centre line; that happens once and is cached under
`$XDG_CACHE_HOME/gazatu.xyz/im-emoji-picker/hanzi-templates` (510 MiB for GBK, memory-mapped
afterwards; `IM_EMOJI_PICKER_HANZI_CACHE` overrides the location). The first drawing you make in
the handwriting view starts the build, which takes about 20 seconds on eight threads here, and the
status line shows the progress. After that the two worker routes together answer in about 60
milliseconds, and the row is updated in place. Without a CJK font installed the field route cannot
build and says so; the trajectory and component routes keep working.

Measured on a drawn 谞 (the comparison script builds one from the reference's medians): the
trajectory route returns 谓 谙 隋 ... - 谞 is not in Make Me a Hanzi - the field route on pen strokes is
noise, and the component route names 谞 as ⿰讠胥 at rank 1. `tools/hanzi/compare_hanzi.py` checks
that and the rendered-glyph case, where 谞 comes back at rank 1 from the charset-wide templates.

### Building with handwriting

```sh
# ggml must be discoverable; when built as a static library, build it with
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON so it can go into the addon
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/ggml/install
cmake --build build
```

Three targets are produced: `fcitx5imemojipicker.so`, the IBus engine, and
`picker-gui` (a development harness that runs the window on its own).

### The model file

The weights are not vendored. `tools/hccr` downloads the upstream Apache-2.0
checkpoint, folds the BatchNorm layers, and writes a GGUF with the character set
and topology embedded:

```sh
python3 tools/hccr/fetch_model.py
PYTHONPATH=/path/to/llama.cpp/gguf-py python3 tools/hccr/convert_to_gguf.py
```

Installing puts it in `<prefix>/share/im-emoji-picker/hccr-mobilenetv2.gguf`.
To use another location, set the `handwritingModelPath` setting or the
`IM_EMOJI_PICKER_HCCR_MODEL` environment variable.

### Measuring it

The harness reports numbers rather than impressions. `tools/hccr/hccr_model.py`
is a numpy reference implementation of the same network, and everything else is
checked against it:

```sh
python3 tools/hccr/eval_accuracy.py          # reference accuracy over stroke widths
python3 tools/hccr/make_test_inputs.py       # test bitmaps, raw and normalized
python3 tools/hccr/convert_to_gguf.py        # produce the model
python3 tools/hccr/verify_gguf.py            # GGUF reproduces the reference logits
python3 tools/hccr/compare_ggml.py           # ggml reproduces them on normalized input
python3 tools/hccr/compare_ggml.py --preprocess   # and after the C++ normalization
python3 tools/hccr/eval_runtime.py           # accuracy of the shipped C++ path
python3 tools/hccr/drive_picker.py --char 休 # drives the real window under Xvfb
```

The rare-character routes have their own harness, which prints what each route found for one
drawing in the reference's own layout:

```sh
build/hanzi-verify --strokes drawing.json          # per route, for a pad drawing
build/hanzi-verify --strokes drawing.json --json   # machine readable
build/hanzi-bench --mode detailed                  # what each route costs
build/hanzi-bench --mode build --cache /tmp/cache  # a full template build
build/hanzi-bench --print --k 10                   # the field rank with its scores
python3 tools/hanzi/compare_hanzi.py               # the port against the Python reference
python3 tools/hccr/drive_picker.py --strokes-json drawing.json --scan 9 --expect 谞  # through the real window
```

`hanzi-bench` reports the load average next to its times: the field route is memory-bandwidth
bound, and a busy machine reports numbers several times worse than an idle one. `--print` dumps the
candidates with their scores so a change that is meant to be a pure speedup can be diffed against a
build of the revision before it.

## Setup 😅

After installing *I'm Emoji Picker* theres some steps required to make it work.

- Setup your input method framework (fcitx5 or ibus) if not already done.
  - Install it
  - Setup `/etc/environment`
  - Add IMF to autostart
- Add *I'm Emoji Picker* to your input methods next to your keyboard language.
- Maybe configure a global shortcut with which to change the current input method and open the emoji picker.
  - Advanced: you could also use the `ibus` or `fcitx5-remote` commands to switch the input method directly

### With Fcitx5

See [https://wiki.archlinux.org/title/Fcitx5](https://wiki.archlinux.org/title/Fcitx5).

#### TLDR (Fcitx5)

Update your `/etc/environment` file to include the following snippet:

```ini
GTK_IM_MODULE=fcitx
QT_IM_MODULE=fcitx
XMODIFIERS=@im=fcitx
```

Create an autostart entry for fcitx5. (works out of the box with some DEs)

### With IBus

See [https://wiki.archlinux.org/title/IBus](https://wiki.archlinux.org/title/IBus).

#### TLDR (IBus)

Update your `/etc/environment` file to include the following snippet:

```ini
GTK_IM_MODULE=ibus
QT_IM_MODULE=ibus
XMODIFIERS=@im=ibus
```

Execute `ibus-daemon -rxR` in your terminal to create an autostart entry.

Execute `ibus-setup` to open the IBus settings so you can add *I'm Emoji Picker* to your input methods and maybe change your global shortcut to change the active input method.

### Settings 📝

```ini
; The path of this file should be: $XDG_CONFIG_HOME/gazatu.xyz/im-emoji-picker.ini
; $XDG_CONFIG_HOME is usually ~/.config
; Can also be opened by pressing F4 while the emoji picker is open.
; most options require you to restart the IMF (fcitx or ibus)

[General]
; `true` = Immediately close the emoji picker after pressing enter.
; Can be done with `false` using shift+enter.
closeAfterFirstInput=false
; `true` = Only gender neutral emojis are visible (people and jobs for example)
gendersDisabled=false
; `not -1` = Any emoji released after this number is hidden
maxEmojiVersion=-1
; `true` = remember recently used kaomoji
saveKaomojiInMRU=false
; `float` = scale the emoji picker based on this number (for example 1.25)
; `` = use QT_SCALE_FACTOR
; (requires IMF restart)
; ALERT: doesn't work currently
scaleFactor=
; `true` = Only skin tone neutral emojis are visible (hands for example)
skinTonesDisabled=false
; `not empty` = Use this (for example: Noto Color Emoji) instead of your system font to display emojis
; (requires useSystemEmojiFont=true)
systemEmojiFontOverride=
; `true` = Use your system emoji font instead of the bundled Twemoji images to display emojis
; (requires IMF restart)
useSystemEmojiFont=false
; `true` = Automatically try to scale or hide emojis based on their system emoji font support
; (May lead to false positives)
useSystemEmojiFontWidthHeuristics=true
; `true` = Use the system Qt theme instead of the builtin dark fusion theme
; (requires IMF restart)
useSystemQtTheme=false
; `0` = Invisible emoji picker window
; (requires IMF restart)
windowOpacity=0.9

; Use something like the following to add custom hotkeys (target = the default key press as seen below):
; [customHotKeys]
; 1\sourceKeyChr=#
; 1\targetKeySeq=shift+tab
; size=1
[customHotKeys]
size=0

; The files to load emoji aliases from.
; Refer to src/res/aliases/github-emojis.ini for an example
; (requires IMF restart)
[emojiAliasFiles]
1\path=:/res/aliases/github-emojis.ini
size=1
```

### Known Issues 😅

- On Debian with Gnome i had to reboot after installing to be able to configure the emoji picker input method

- When using Wayland it seems like most window managers (Sway, KDE/KWin) force focus on new windows which conflicts with the way im-emoji-picker handles inputs. This results in the emoji picker either flickering or disappearing instantly. A workaround is to manually configure a window rule.
  - Sway: `no_focus [title="im-emoji-picker"]`
  - KDE/KWin: https://github.com/GaZaTu/im-emoji-picker/issues/13#issuecomment-1879469985

- When using Wayland both Fcitx5 and IBus might not report the correct text cursor location so the emoji picker can open either in the top left corner or in the center of the screen

- If the emoji picker is ugly and doesn't follow your system theme (a bit like [this](https://api.gazatu.xyz/blog/entries/01GQCEZA5K1162PYXBRK11T76N/image.webp)) then take a look at [https://wiki.archlinux.org/title/Uniform_look_for_Qt_and_GTK_applications](https://wiki.archlinux.org/title/Uniform_look_for_Qt_and_GTK_applications).

- Required font localizations for Kaomoji support:
  - cjk (tc, sc, kr, jp)
  - kannada
  - thai
  - tibetan
  - sinhala

## Usage 🧐

- `ctrl+a` = select all text in search input
- `ctrl+c` = copy selected emoji
- `ctrl+x` = cut selection in search (not really tbh)
- `ctrl+backspace` = clear search
- `up`, `down`, `left`, `right` = change selection
- `shift+up`, `shift+down`, `pgup`, `pgdown` = change selection (faster)
- `escape` = close emoji picker
- `return` = write emoji to target input
- `shift+return` = write emoji to target input and close emoji picker
- `tab` = change view (MRU, List, Kaomoji)
- `shift+tab` = change view (MRU, List, Kaomoji) (reverse)
- `f4` = open settings file and close emoji picker

## Building 🤓

### Build Dependencies

- cmake
- make or ninja
- gcc or clang

### Dependencies

- Qt5 (core, gui, widgets)
- fcitx5 or ibus

### Example Commands To Install Dependencies (probably)

**Arch**:
`sudo pacman -S gcc make cmake qt5-base fcitx5 fcitx5-qt fcitx5-gtk`

**Debian**:
`sudo apt install gcc make cmake qtbase5-dev fcitx5 fcitx5-frontend-gtk3 fcitx5-frontend-qt5 im-config`

**openSUSE**:
`sudo zypper install gcc make cmake libqt5-qtbase-devel ibus-devel`

### CMake

- `mkdir -p build`
- `cd build`
- `cmake -DCMAKE_BUILD_TYPE=Release ..`
- `make -j$(nproc)`

## Special Thanks 🤗

- boring_nick for testing this on his arch+sway setup during the initial development phase
- [zneix](https://github.com/zneix) and other contributors for their help on the original [x11-emoji-picker](https://github.com/GaZaTu/x11-emoji-picker)

## License 😈

Project licensed under the [MIT](https://opensource.org/licenses/MIT) license: [LICENSE](LICENSE)

Emoji [graphics](src/res/72x72) licensed by [Twitter](https://github.com/twitter) under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) at [https://github.com/twitter/twemoji](https://github.com/twitter/twemoji/blob/master/LICENSE-GRAPHICS)

Emoji [list](src/emojis.cpp) licensed by [Unicode](https://github.com/unicode-org) at [https://github.com/unicode-org/cldr](https://github.com/unicode-org/cldr/blob/master/unicode-license.txt)
