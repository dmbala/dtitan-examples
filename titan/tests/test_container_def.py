import pathlib

DEF = pathlib.Path(__file__).resolve().parent.parent / "container" / "dtitan.def"


def test_targets_torch_211_and_pinned_torchtitan():
    text = DEF.read_text()
    assert "torchtitan==0.2.2" in text
    assert "HF_HUB_OFFLINE=1" in text
    # base image comment must state the torch>=2.11 requirement
    assert "torch" in text and "2.11" in text
    # build must fail loudly if torchtitan.train can't import
    assert "import torchtitan.train" in text


def test_documents_rebuild_reason():
    text = DEF.read_text()
    assert "_context_parallel_shard" in text or "activate_flash_attention_impl" in text
