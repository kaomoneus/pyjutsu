"""
Deprecated. Use pydantic, and pujutsu.pydantic_model helper instead.

Here I tried to introduce ultimate To/From JSON serialization.
Idea was to call serialization/deserialization of small objects with
single line.
It might be well implemented already, but I didn't find any good solutions,
possible candidates were
   * dataclasses.asdict with custom factory
   * json.dump with some additional encoder/decoder classes
   * pickle
All of them though can't be called within single line out-of-box.
Or I miss something?
Anyways, let it be here until we find some good alternative.
"""

import base64
import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Union

# For python3.8+ import collections.abc.Sequence
# otherwise import collections.Sequence
if sys.version_info.major > 3 or sys.version_info.minor > 8:
    from collections.abc import Sequence
else:
    from collections import Sequence

import numpy as np

from pyjutsu.loaders import load_class

JsonableType = Union[dict, list, tuple]

NULL_STR = "null"


_bypass_types = [
    int, float, type(None), str
]


def to_jsonable(o, level=0):

    if level > 0:
        for t in _bypass_types:
            if isinstance(o, t):
                return o

    if isinstance(o, dict):
        return {
            k: to_jsonable(v, level + 1)
            for k, v in o.items()
        }

    if isinstance(o, tuple) or isinstance(o, Sequence):
        return [to_jsonable(item, level + 1) for item in o]

    if isinstance(o, np.ndarray):
        data_b64 = base64.b64encode(o.data)
        return dict(
            __ndarray__=data_b64.decode("ascii"),
            dtype=str(o.dtype),
            shape=o.shape
        )

    if dataclasses.is_dataclass(o):
        fields = set(f.name for f in dataclasses.fields(o))
        encoded_data = to_jsonable(
            {
                nm: v
                for nm, v in o.__dict__.items()
                if nm in fields
            },
            level + 1
        )
        cls = type(o)

        return dict(
            __dataclass__=encoded_data,
            __module__=cls.__module__,
            __type__=cls.__qualname__
        )

    if isinstance(o, Enum):
        enum_item_name = o.name
        cls = type(o)
        return dict(
            __enum__=enum_item_name,
            __module__=cls.__module__,
            __type__=cls.__qualname__
        )

    if isinstance(o, Path):
        return dict(__path__=str(o))

    raise ValueError("Can't serialize to dict")


def _null(d):
    if isinstance(d, str):
        if d.lower() == "null":
            return None
    raise ValueError


def _float(d):
    x = float(d)
    if isinstance(d, str) and '.' in d or not x.is_integer():
        return x

    raise ValueError


_primitive_factories = [_float, int, _null]


def _decode_primitive(d):
    for f in _primitive_factories:
        try:
            return True, f(d)
        except ValueError:
            pass
        except TypeError:
            pass

    return False, None


def from_jsonable(d):
    if isinstance(d, str):
        res, v = _decode_primitive(d)
        if res:
            return v

    if isinstance(d, list):
        return [from_jsonable(item) for item in d]

    if isinstance(d, dict):
        data_b64 = d.get("__ndarray__")
        if data_b64:
            data = base64.b64decode(data_b64.encode())
            return np.frombuffer(data, d['dtype']).reshape(d['shape'])

        dc = d.get("__dataclass__")
        if dc is not None:
            dc = from_jsonable(dc)
            module_path = d["__module__"]
            cls_path = d["__type__"]
            cls = load_class(module_path, cls_path)
            return cls(**dc)

        en = d.get("__enum__")
        if en is not None:
            module_path = d["__module__"]
            cls_path = d["__type__"]
            enum_item_name = d["__enum__"]
            cls = load_class(module_path, cls_path)
            return cls[enum_item_name]

        pth = d.get("__path__")
        if pth is not None:
            return Path(pth)

        return {n: from_jsonable(v) for n, v in d.items()}

    return d


def dumps(d):
    d = to_jsonable(d)
    return json.dumps(d)


def loads(v):
    d = json.loads(v)
    return from_jsonable(d)
