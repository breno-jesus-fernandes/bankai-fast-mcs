import importlib


def test_native_extension_imports():
    module = importlib.import_module("bankai_fast_mcs._native")

    assert module.__name__ == "bankai_fast_mcs._native"
