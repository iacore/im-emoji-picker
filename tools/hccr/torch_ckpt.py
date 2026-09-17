"""Read a torch.save() checkpoint without PyTorch installed.

A .pt file is a zip archive: `*/data.pkl` holds a pickle whose tensors are
`torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)` calls and
whose storages live in `*/data/<key>` as raw little-endian bytes. We resolve
those two indirections and hand back plain numpy arrays.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np

_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "ByteStorage": np.uint8,
    "CharStorage": np.int8,
    "BoolStorage": np.bool_,
}


class _Storage:
    __slots__ = ("key", "dtype")

    def __init__(self, key: str, dtype: np.dtype):
        self.key = key
        self.dtype = dtype


# `persistent_id` writes the storage class into the pickle stream, so unpickling
# asks find_class for it; hand back a stand-in that carries the numpy dtype.
_STORAGE_CLASSES: dict[str, type] = {
    name: type(name, (), {"dtype": np.dtype(dtype), "_torch_storage": True})
    for name, dtype in _DTYPES.items()
}


class _DictLike(dict):
    """`collections.OrderedDict` stand-in that also tolerates a pickle BUILD state."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)


class _Placeholder:
    """Any class we cannot resolve; keeps BUILD/attribute access from exploding."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        self._state = state

    def __repr__(self):
        return f"<placeholder {self._args!r} {self._kwargs!r}>"


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.archive = archive
        self.prefix = prefix
        self.cache: dict[str, np.ndarray] = {}

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return self._rebuild_tensor
        if module == "torch" and name in _STORAGE_CLASSES:
            return _STORAGE_CLASSES[name]
        if module in ("collections", "collections.abc") and name in ("OrderedDict", "dict"):
            return _DictLike
        try:
            return super().find_class(module, name)
        except Exception:
            return _Placeholder

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        if isinstance(pid, (tuple, list)) and pid and pid[0] in ("storage", b"storage"):
            storage_type = pid[1]
            key = pid[2]
            if isinstance(key, bytes):
                key = key.decode()
            dtype = getattr(storage_type, "dtype", np.dtype(np.float32))
            return _Storage(str(key), np.dtype(dtype))
        raise pickle.UnpicklingError(f"unexpected persistent id: {pid!r}")

    def _raw(self, key: str) -> np.ndarray:
        if key not in self.cache:
            self.cache[key] = np.frombuffer(self.archive.read(f"{self.prefix}data/{key}"), dtype=np.uint8)
        return self.cache[key]

    def _rebuild_tensor(self, storage, offset, size, stride, *args):
        flat = self._raw(storage.key).view(storage.dtype)
        count = 1
        for dim in size:
            count *= dim
        return np.array(flat[offset:offset + count]).reshape(tuple(size))


def load_checkpoint(path: str | Path) -> dict:
    """Return the pickled checkpoint dict with tensors as numpy arrays."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        pickle_name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        prefix = pickle_name[: -len("data.pkl")]
        with archive.open(pickle_name) as handle:
            return _Unpickler(handle, archive, prefix).load()
