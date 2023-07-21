# tests functions for experiments

from Experiments import mayasim_X1_default_setup as X1, \
                        mayasim_X2_scan_r_es_and_r_agg as X2, \
                        mayasim_X3_trade as X3, \
                        mayasim_X4_es_income as X4
                        

def test_X1():
    """
    test run of X1, base line experiment (default setup)
    """
    assert X1.run_experiment(['testing', 1]) == 1


def test_X2():
    """
    test run of X2, experiment concerning income 
    from agriculture and ecosystem services
    """
    assert X2.run_experiment(['testing', 1]) == 1


def test_X3(): 
    """
    test run of X3, experiment concerning 
    the influence of trade income
    """
    assert X3.run_experiment(['testing', 1]) == 1


def test_X_4():
    """
    test run of X4, experiment checking whether climate 
    variability actually matters for the overshoot and
    collapse pattern
    """
    assert X4.run_experiment(['testing', 1]) == 1
    