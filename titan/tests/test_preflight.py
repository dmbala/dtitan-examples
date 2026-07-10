import preflight


def test_torchtitan_config_imports():
    _, ok, _ = preflight.check_torchtitan_import()
    assert ok is True  # torchtitan.config imports even on torch 2.10


def test_train_check_is_bool():
    name, ok, detail = preflight.check_torchtitan_train()
    assert name == "torchtitan.train import"
    assert isinstance(ok, bool)  # False now (torch 2.10), True after rebuild


def test_tokenizer_check_true_for_existing(tmp_path):
    _, ok, _ = preflight.check_tokenizer(str(tmp_path))
    assert ok is True


def test_tokenizer_check_false_for_missing():
    _, ok, _ = preflight.check_tokenizer("/nonexistent/xyz")
    assert ok is False
