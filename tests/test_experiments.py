# test functions for experiments

from experiments.jkolb import \
    X1_default_setup as X1, \
    X2_scan_r_es_and_r_agg as X2, \
    X3_trade as X3, \
    X4_es_income as X4


def test_x1():
    """
    test run of X1, baseline experiment (default setup)
    """
    assert X1.run_experiment('test') == 1


def test_x2():
    """
    test run of X2, experiment concerning income
    from agriculture and ecosystem services
    """
    assert X2.run_experiment('test') == 1


def test_x3():
    """
    test run of X3, experiment concerning
    the influence of trade income
    """
    assert X3.run_experiment('test') == 1


def test_x4():
    """
    test run of X4, experiment checking whether climate
    variability actually matters for the overshoot and
    collapse pattern
    """
    assert X4.run_experiment('test') == 1
