from dtensor_workshop import distenv


def test_defaults_when_unset(monkeypatch):
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    assert distenv.rank() == 0
    assert distenv.world_size() == 1
    assert distenv.local_rank() == 0
    assert distenv.is_distributed() is False


def test_reads_env(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert distenv.rank() == 3
    assert distenv.world_size() == 4
    assert distenv.local_rank() == 3
    assert distenv.is_distributed() is True


def test_device_type_is_cpu_without_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert distenv.device_type() == "cpu"
