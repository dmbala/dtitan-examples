from labs.level2 import l2_oom


def test_ac_equivalence():
    assert l2_oom.ac_equivalence_maxdiff() < 1e-6
