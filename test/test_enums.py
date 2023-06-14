from lyncs_quda import enums, Enum


def test_enums():
    for enum in dir(enums):
        if enum.startswith("_"):
            continue

        enum = getattr(enums, enum)
        assert issubclass(enum, Enum)
        for key, val in enum.items():
            assert key in enum
            assert val in enum
            assert enum[key] == key
            assert enum[key] == val
            assert enum[key] == enum[val]
            assert str(enum[key]) == key
            assert int(enum[key]) == int(enum[val]) == val
