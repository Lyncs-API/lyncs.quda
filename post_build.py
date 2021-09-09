import re
import fileinput
from pathlib import Path
from os.path import commonprefix

# PATCH 1: replace relative includes in header files


def post_build(builder, ext):
    patch_include(builder, ext)
    generate_enums(builder, ext)
    generate_structs(builder, ext)


def patch_include(builder, ext):
    'Replaces #include instances in header files that use <> with "" for relative includes'
    install_dir = builder.get_install_dir(ext) + "/include"
    for path in Path(install_dir).rglob("*.h"):
        with fileinput.FileInput(str(path), inplace=True, backup=".bak") as fp:
            for fline in fp:
                line = str(fline)
                if line.strip().startswith("#include"):
                    include = line.split()[1]
                    if include[0] == "<" and include[-1] == ">":
                        include = include[1:-1]
                        if (path.parents[0] / include).exists():
                            print(line.replace(f"<{include}>", f'"{include}"'), end="")
                            continue
                print(line, end="")


# PATCH 2: generates enums.py

ENUM_OUTPUT = """
"List of QUDA enumerations"

# NOTE: This file is automathically generated by setup.py
# DO NOT CHANGE MANUALLY but reinstall the package
# python setup.py develop

from .enum import Enum
"""

ENUM_CLASS = """

class {name}(Enum):
    \"""
    {docs}
    \"""

    _prefix = "{prefix}"
    _suffix = "{suffix}"
    _values = {values}
"""

REPLACE = {
    "{": " { ",
    "}": " } ",
    ",": " , ",
    "=": " = ",
    ";": " ; ",
    "[": " [ ",
    "]": " ] ",
    "*": " * ",
}
PATTERN = re.compile("|".join((re.escape(k) for k in REPLACE.keys())), flags=re.DOTALL)

get_words = lambda line: PATTERN.sub(lambda x: REPLACE[x.group(0)], line).split()


def commonsuffix(words):
    "Finds common suffix between list of words"
    reverse = lambda word: word[::-1]
    words = list(map(reverse, words))
    return commonprefix(words)[::-1]


def parse_enum(lines):
    from cppyy import gbl

    assert lines[0].startswith("typedef enum")

    # getting comments:
    comments = {}
    for i, line in enumerate(lines):
        if "//" in line:
            line, comment = line.split("//")
            line = line.strip()
            comment = comment.strip()
            lines[i] = line
            key = get_words(line)[0]
            comments[key] = comment

    words = get_words(" ".join(lines))
    assert words[-1] == ";"

    name = words[-2]
    enums = []
    for i in range(len(words)):
        if words[i] in ["{", ","]:
            enums.append(words[i + 1])

    prefix = commonprefix(enums).lower()
    suffix = commonsuffix(enums).lower()
    clean = (
        lambda word: word[len(prefix) : -len(suffix)] if suffix else word[len(prefix) :]
    )

    comments = {
        clean(key.lower()): "    # " + val
        for key, val in comments.items()
        if key in enums
    }

    # Generating values
    values = {clean(key.lower()): int(getattr(gbl, key)) for key in enums}

    # Generating docs
    docs = [f"{key} = {val}{comments.get(key,'')}" for key, val in values.items()]
    docs = "\n    ".join(docs)

    return ENUM_CLASS.format(
        name=name, prefix=prefix, suffix=suffix, values=values, docs=docs
    )


def generate_enums(builder, ext):
    "Reads enum_quda.h and returns the content of enums.py"
    from cppyy import include

    package_dir = builder.get_install_dir(ext)
    header = package_dir + "/include/enum_quda.h"
    include(header)
    header = open(header)
    lines = header.readlines()
    header.close()

    # packing groups of enums and then calling parse_enum
    pack = []
    out = ENUM_OUTPUT
    for line in lines:
        line = line.strip()
        if line.startswith("typedef enum"):
            pack.append(line)
        elif pack:
            pack.append(line)
        if pack and line.endswith(";"):
            out += parse_enum(pack)
            pack = []

    filename = package_dir + "/enums.py"
    with open(filename, "w") as fp:
        fp.write(out)

    try:
        from black import format_file_in_place, Mode, Path, WriteBack

        format_file_in_place(Path(filename), False, Mode(), write_back=WriteBack.YES)
    except ImportError:
        pass


# PATCH 3: generates structs.py

STRUCT_OUTPUT = """
"List of QUDA parameter structures"

# NOTE: This file is automathically generated by setup.py
# DO NOT CHANGE MANUALLY but reinstall the package
# python setup.py develop

from .enums import *
from .struct import Struct
"""

STRUCT_CLASS = """

class {name}(Struct):
    \"""{docs}
    \"""

    _types = {types}
"""


def get_name_type(line):
    words = get_words(line)
    assert words[-1] == ";"
    if "[" in words:
        par = words.index("[")
        key = words[par - 1]
        typ = words[: par - 1] + words[par:-1]
        assert typ[-1] == "]"
    else:
        key = words[-2]
        typ = words[:-2]
    return " ".join(typ), key


def parse_struct(lines):
    assert lines[0].startswith("typedef struct")
    assert "}" in lines[-1]

    types = {}
    comment = ""
    comments = {}
    comment_region = False
    for i, line in enumerate(lines[1:]):
        line = line.strip()
        if comment_region:
            if "*/" in line:
                tmp, line = line.split("*/")
                comment_region = False
            else:
                tmp = line
            comment += "\n" + tmp.strip()
            if comment_region:
                continue

        if "//" in line:
            line, tmp = line.split("//")
            comment += tmp
        elif "/*" in line:
            line, tmp = line.split("/*")
            comment += tmp
            if "*/" not in comment:
                comment_region = True

        if "}" in line:
            words = get_words(line)
            assert ";" == words[-1]
            assert "}" == words[0]
            assert len(words) == 3
            name = words[1]
        elif ";" in line:
            comment = comment.strip("/*<- \n")
            line = line.strip()
            typ, key = get_name_type(line)
            types[key] = typ
            comments[key] = comment.replace("\n", "\n        ")
            comment = ""

    docs = f"\n    {name} struct:\n" + "\n".join(
        f"    - {key} : {val}" for key, val in comments.items()
    )
    return STRUCT_CLASS.format(name=name, types=types, docs=docs)


def generate_structs(builder, ext):
    "Reads enum_quda.h and returns the content of enums.py"

    package_dir = builder.get_install_dir(ext)
    header = open(package_dir + "/include/quda.h")
    lines = header.readlines()
    header.close()

    # packing groups of enums and then calling parse_enum
    pack = []
    out = STRUCT_OUTPUT
    for line in lines:
        line = line.strip()
        if line.startswith("typedef struct"):
            pack.append(line)
        elif pack:
            pack.append(line)
        if pack and "}" in line and line.endswith(";"):
            out += parse_struct(pack)
            pack = []

    filename = package_dir + "/structs.py"
    with open(filename, "w") as fp:
        fp.write(out)

    try:
        from black import format_file_in_place, Mode, Path, WriteBack

        format_file_in_place(Path(filename), False, Mode(), write_back=WriteBack.YES)
    except ImportError:
        pass