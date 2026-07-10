from labs.level3 import l3_faultdata


def test_crash_and_resume():
    assert l3_faultdata.simulate_crash_and_resume() is True
