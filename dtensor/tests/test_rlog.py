from dtensor_workshop import rlog


def test_prefix_default(monkeypatch):
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    assert rlog.prefix() == "[rank 0/1]"


def test_prefix_reads_env(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert rlog.prefix() == "[rank 2/4]"


def test_info_prints_prefix(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    rlog.info("hello")
    out = capsys.readouterr().out
    assert "[rank 1/4] hello" in out
