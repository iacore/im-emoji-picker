<!-- item: install-package state=blocked owner=- blocked=human -->
## Install the rebuilt package

`packaging/makepkg --noextract -f` leaves a new `.pkg.tar.zst`; installing it
needs root, which the agent does not have:

```sh
doas pacman -U packaging/fcitx5-im-emoji-picker-handwriting-git-*.pkg.tar.zst
doas dinitctl restart fcitx5
```
<!-- item: harness-teardown-crash state=open owner=- blocked=- -->
## The dev harness crashes destroying the window

`build/picker-gui` prints `COMMIT …` and then dies in
`QGridLayout::~QGridLayout()`: `_emojiLayoutItems` holds a `QWidgetItemV2` per
emoji code, and the grid deletes it again after the widget is gone. Pre-existing,
visible only in the harness (an input method is killed, not asked to tear down
cleanly); it surfaced while smoke-testing the search.
<!-- item: history-home-paths state=blocked owner=- blocked=human -->
## Home paths in published history

`packaging/install.txt`, `packaging/im-emoji-picker/config` and
`tools/hanzi/compare_hanzi.py` carried `/home/<user>/…` in commits from before
2026-10-31, and the fork was already public by then. The files are fixed; the
history still holds them. Repairing history means a force-push, which only the
user can decide on.
<!-- item: tracked-build-junk state=blocked owner=- blocked=human -->
## 72 MB of build output is tracked

`packaging/im-emoji-picker/` (a bare mirror of this repo), `packaging/best.pt`
and the llama.cpp tarball are makepkg's download cache inside the tree. They are
committed, so every clone pays for them. Removing them is a `git rm` plus a
`.gitignore` entry, but it changes what the PKGBUILD's neighbours hold, so it is
the user's call.
