import importlib


def test_package_imports():
    mod = importlib.import_module("dtensor_workshop")
    assert mod is not None


def test_labs_package_imports():
    assert importlib.import_module("labs.level1") is not None
