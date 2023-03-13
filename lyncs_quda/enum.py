"Enum base class"

__all__ = ["Enum"]

from collections import namedtuple


class EnumValue(namedtuple("EnumValue", ["cls", "key"])):
    "Representation of an enum entry as (enum_class_name, val)"
    "val can either be string or int"

    def __str__(self):
        return self.cls.to_string(self.key)

    def __int__(self):
        return self.cls.to_int(self.key)

    __index__ = __int__

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == self.cls.clean(other)
        if isinstance(other, int):
            return int(self) == other
        if isinstance(other, EnumValue):
            return self.cls is other.cls and int(self) == int(other)
        return False

    def __ne__(self, other):
        return not (
            self == other
        )  # TODO: perhaps better to insert if-cond for NotImpelented

    def __contains__(self, other):
        if isinstance(other, str):
            return self.cls.clean(other) in str(self)
        if isinstance(other, int):
            return self.to_string(other) in str(self)
        return False


class EnumMeta(type):
    "Metaclass for enum types"

    def keys(cls):
        "List of enum keys"
        return cls._values.keys()

    def values(cls):
        "List of enum values"
        return cls._values.values()

    def items(cls):
        "List of enum items"
        return cls._values.items()

    def clean(cls, rep):
        # should turn everything into upper for consistency
        "Strips away prefix and suffix from key"
        "See enums.py to find what is prefix and suffix for a given enum value"
        if isinstance(rep, EnumValue):
            rep = str(rep)
        if isinstance(rep, str):
            rep = rep.lower()
            if cls._prefix and rep.startswith(cls._prefix):
                rep = rep[len(cls._prefix) :]
            if cls._suffix and rep.endswith(cls._suffix):
                rep = rep[: -len(cls._suffix)]
        return rep

    def to_string(cls, rep):
        "Returns the key representative of the given enum value"
        rep = cls.clean(rep)
        if rep not in cls:
            raise ValueError(f"Unknown enum '{rep}' for {cls.__name__}")

        if isinstance(rep, str):
            return rep

        assert isinstance(rep, int)
        return list(cls.keys())[list(cls.values()).index(rep)]

    def to_int(cls, key):
        "Returns the value representative of the given enum value"
        key = cls.clean(key)
        if key not in cls:
            raise ValueError(f"Unknown enum '{key}' for {cls.__name__}")

        if isinstance(key, str):
            return cls._values[key]

        assert isinstance(key, int)
        return key

    def __contains__(cls, entry):
        entry = cls.clean(entry)

        if isinstance(entry, str):
            return entry in cls.keys()

        if isinstance(entry, int):
            return entry in cls.values()

        return False

    def __getitem__(cls, entry):
        if entry not in cls:
            raise ValueError(f"Unknown enum '{entry}' for {cls.__name__}")
        entry = cls.clean(entry)
        return EnumValue(cls, entry)


class Enum(metaclass=EnumMeta):
    "Enum base class"
    _prefix = ""
    _suffix = ""
    _values = {}

    def __init__(self, fnc, lpath=None, default=None, callback=None):
        # fnc is supposed to return either a stripped key name or value of
        # the corresponding QUDA enum type
        self.fnc = fnc
        self.lpath = lpath
        self.default = default
        self.callback = callback

    def __call__(self, instance):
        # intended for property.fget, which then invokes
        # property.__get__(self, obj, objtype=None)
        return EnumValue(type(self), self.fnc(instance))

    # not meant to be a stnadard descriptor, c.f., solver.py
    def __get__(self, instance, owner):
        if instance is None:
            raise AttributeError

        out = instance
        for key in self.lpath.split("."):
            out = getattr(out, key)
        return type(self)[out]

    def __set__(self, instance, new):
        if new is None and self.default is not None:
            if callable(self.default):
                new = self.default(instance)
            else:
                new = self.default
        new = int(type(self)[new])

        out = instance
        for key in self.lpath.split(".")[:-1]:
            out = getattr(out, key)
        key = self.lpath.split(".")[-1]
        old = int(getattr(out, key))

        setattr(out, key, new)
        if self.callback:
            self.callback(instance, old, new)
