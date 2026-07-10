from labs.level3 import l3_compile


def test_compile_maxdiff():
    assert l3_compile.compile_maxdiff() < 1e-4
