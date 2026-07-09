import preflight


def test_torch_import_ok():
    name, ok, _ = preflight.check_torch_import()
    assert name == "torch import"
    assert ok is True


def test_dtensor_import_ok():
    _, ok, _ = preflight.check_dtensor_import()
    assert ok is True


def test_dir_writable(tmp_path):
    _, ok, _ = preflight.check_dir_writable(str(tmp_path))
    assert ok is True


def test_dir_not_writable():
    _, ok, _ = preflight.check_dir_writable("/nonexistent/path/xyz")
    assert ok is False


def test_run_cpu_checks_returns_rows(tmp_path):
    rows = preflight.run_cpu_checks([str(tmp_path)])
    assert all(len(r) == 3 for r in rows)
    assert any(r[0] == "torch import" for r in rows)
