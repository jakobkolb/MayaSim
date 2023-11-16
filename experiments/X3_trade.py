"""
I want to know, if for increasing trade income, there is a transition to a
complex society that can sustain itself.
And I want to know, what happens at that transition. Does it shift, due to
climate variability? Hypothesis: Yes, it does.

Therefore, vary two parameters: r_trade and precipitation_amplitude
"""

# disable pylint invalid-name message on module level
# pylint: disable=invalid-name
# pylint: enable=invalid-name

import sys
import getpass

import itertools as it
import pickle as pkl
import numpy as np
import pandas as pd

from pymofa.experiment_handling import experiment_handling as handle
from mayasim.model.core import Core as Model
from mayasim.model.parameters import Parameters

TEST = True


def run_function(r_trade=6000., precip_amplitude=1.,
                 n=30, kill_cropless=False,
                 steps=350, filename='./'):
    """Initializes and runs model and retrieves and saves data afterwards.

    Parameters
    ----------
    precip_amplitude: float
        the prefactor to the precipitation modulation. 0. means no modulation
        1. means original modulation >1. means amplified modulation.
    r_trade: float
        value of trade income
    n: int
        number of initial settlements
    kill_cities_without_cropps: bool
        switch to set whether or not to kill settlements without agriculture
    steps: int
        number of steps to run the model
    rf_filename: string
    """

    # Initialize Model

    if TEST:
        n = 100
    m = Model(n=n, output_path=filename)
    m.r_trade = r_trade
    m.precipitation_amplitude = precip_amplitude
    m.output_level = 'aggregates'
    m.kill_cities_without_crops = kill_cropless


    if not filename.endswith('s0.pkl'):
        m.output_geographic_data = False
        m.output_settlement_data = False

    # Store initial conditions and parameters:

    res = {"initials": pd.DataFrame({"Settlement X Positions":
                                     m.stm_positions[0],
                                     "Settlement Y Positions":
                                     m.stm_positions[1],
                                     "Population": m.stm_population}),
           "Parameters": pd.Series({key: getattr(m, key)
                                    for key in dir(Parameters)
                                    if not key.startswith('__')
                                    and not callable(key)})
           }

    # Run model

    if TEST:
        steps = 3

    m.run(steps)

    # Save results

    res["aggregates"] = m.get_aggregates()
    res["traders aggregates"] = m.get_traders_aggregates()

    try:
        with open(filename, 'wb') as dumpfile:
            pkl.dump(res, dumpfile)
            return 1
    except IOError:
        return -1


# pylint: disable=too-many-locals
def run_experiment(argv):
    """
    Take argv input variables and run experiment accordingly.
    This happens in five steps:
    1)  parse input arguments to set switches
        for [test],
    2)  set output folders according to switches,
    3)  generate parameter combinations,
    4)  define names and dictionaries of callables to apply to sub_experiment
        data for post processing,
    5)  run computation and/or post processing and/or plotting
        depending on execution on cluster or locally or depending on
        experimentation mode.

    Parameters
    ----------
    argv: list
        List of parameters from terminal input

    Returns
    -------
    rt: int
        some return value to show whether sub_experiment succeeded
        return 1 if successful.
    """

    # Parse test switch from input
    global TEST # pylint: disable=global-statement
    if __name__ == '__main__':
        TEST = len(argv) > 1 and argv[1] == 'test'
    else:
        TEST = argv == 'test'

    # Generate paths according to switches and user name

    test_folder = ['', 'test_experiments/'][int(TEST)]
    experiment_folder = 'X3_trade/'
    raw = 'raw_data/'
    res = 'results/'

    if getpass.getuser() == 'fritz':
        save_path_raw = '/Users/fritz/Desktop/Thesis/MayaSim/' \
                        f'output/{test_folder}{experiment_folder}{raw}'
        save_path_res = '/Users/fritz/Desktop/Thesis/MayaSim/' \
                        f'output/{test_folder}{experiment_folder}{res}'
    else:
        save_path_raw = f'./output/{test_folder}{experiment_folder}{raw}'
        save_path_res = f'./output/{test_folder}{experiment_folder}{res}'

    name1 = "aggregates"
    estimators1 = {"mean_aggregates":
                  lambda fnames: pd.concat([np.load(f, allow_pickle=True)["aggregates"]
                                            for f in fnames]).groupby(
                      level=0).mean(),
                  "sigma_aggregates":
                  lambda fnames: pd.concat([np.load(f, allow_pickle=True)["aggregates"]
                                            for f in fnames]).groupby(
                          level=0).std()
                  }
    name2 = "traders_aggregates"
    estimators2 = {
                  "mean_aggregates":
                      lambda fnames:
                      pd.concat([np.load(f, allow_pickle=True)["traders aggregates"]
                                            for f in fnames]).groupby(
                          level=0).mean(),
                  "sigma_aggregates":
                      lambda fnames:
                      pd.concat([np.load(f, allow_pickle=True)["traders aggregates"]
                                            for f in fnames]).groupby(
                          level=0).std()
                  }


    precip_amplitudes = [0., 0.5, 1., 1.5, 2.] \
        if not TEST else [0., 1.]
    r_trades = [3000., 4000., 5000., 6000., 7000., 8000., 9000., 10000.] \
        if not TEST else [6000., 8000.]
    kill_cities = [True, False]

    parameter_combinations = list(it.product(precip_amplitudes,
                                             r_trades,
                                             kill_cities))

    index = {0: 'precip_amplitude',
             1: 'r_trade',
             2: 'kill_cropless'}
    sample_size = 5 if not TEST else 2


    if TEST:
        print(f'testing {experiment_folder[:-1]}')
    h = handle(sample_size=sample_size,
               parameter_combinations=parameter_combinations,
               index=index,
               path_raw=save_path_raw,
               path_res=save_path_res,
               use_kwargs=True)

    h.compute(run_func=run_function)
    h.resave(eva=estimators1, name=name1)
    h.resave(eva=estimators2, name=name2)

    if TEST:
        data = pd.read_pickle(save_path_res + name1 + '.pkl')
        print(data.head())
        data = pd.read_pickle(save_path_res + name2 + '.pkl')
        print(data.head())
        print(save_path_res)

    return 1


if __name__ == "__main__":

    run_experiment(sys.argv)
