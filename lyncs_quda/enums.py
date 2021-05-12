"Prepares the global variable ENUMS"


__all__ = [
    "get_precision",
]

import re
from functools import lru_cache
from numpy import dtype
from .lib import PATHS, lib

cache = lru_cache(maxsize=None)


def read_header():
    "Reads the header and returns the enums"
    header = open(PATHS[0] + "/include/enum_quda.h")
    lines = header.readlines()
    header.close()

    enums = {}
    pack = []
    replace = {
        "{": " { ",
        "}": " } ",
        ",": " , ",
        "=": " = ",
        ";": " ; ",
    }
    pattern = re.compile(
        "|".join((re.escape(k) for k in replace.keys())), flags=re.DOTALL
    )

    def parse_enum(words):
        enums = []
        assert words[-1] == ";"
        for i in range(len(words)):
            if words[i] in ["{", ","]:
                enums.append(words[i + 1])
        return {words[-2]: enums}

    for line in lines:
        # Stripping and removing comments
        line = line.strip().split("//")[0]
        if line.startswith("typedef enum"):
            pack.append(line)
        elif pack:
            pack.append(line)
        if pack and line.endswith(";"):
            words = pattern.sub(lambda x: replace[x.group(0)], " ".join(pack)).split()
            enums.update(parse_enum(words))
            pack = []

    return enums


ENUMS = read_header()


def strip(key, end, start="QUDA_"):
    if key.startswith(start):
        key = key[len(start) :]
    if key.endswith(end):
        key = key[: -len(end)]
    return key


@cache
def precisions():
    return {
        int(getattr(lib, key)): strip(key, "_PRECISION")
        for key in ENUMS["QudaPrecision"]
    }


def get_precision(key):
    if isinstance(key, str):
        try:
            key = dtype(key)
        except TypeError:
            key = strip(key.upper(), "_PRECISION")
            if key not in precisions().values():
                raise ValueError
            return key
    if isinstance(key, dtype):
        if key in ["float64", "complex128"]:
            return "DOUBLE"
        if key in ["float32", "complex64"]:
            return "SINGLE"
        if key in ["float16"]:  # , "complex32"
            return "HALF"
        raise ValueError
    if isinstance(key, int):
        if key not in precisions():
            raise ValueError
        return precisions()[key]
    raise TypeError


def get_precision_quda(key):
    return getattr(lib, f"QUDA_{get_precision(key)}_PRECISION")
