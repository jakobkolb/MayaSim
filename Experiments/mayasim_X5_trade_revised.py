"""
I want to know, if for increasing trade income, there is a transition to a
complex society that can sustain itself.
And I want to know, what happens at that transition. Does it shift, due to
climate variability? Hypothesis: Yes, it does.

Therefore, vary two parameters: r_trade and precipitation_amplitude
"""

from __future__ import print_function

import getpass
import itertools as it
import numpy as np
import sys

import pandas as pd
try:
    import cPickle as cP
except ImportError:
    import pickle as cP

from pymofa.experiment_handling import experiment_handling as handle
from mayasim.model.ModelCore import ModelCore as Model
from mayasim.model.ModelParameters import ModelParameters as Parameters

test = True


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

    if test:
        n = 100
    m = Model(n=n, output_data_location=filename, debug=test)
    m.r_trade = r_trade
    m.precipitation_amplitude = precip_amplitude
    m.output_level = 'trajectory'
    m.kill_cities_without_crops = kill_cropless
    m.better_ess = True


    if not filename.endswith('s0.pkl'):
        m.output_geographic_data = False
        m.output_settlement_data = False

    # Store initial conditions and parameters:

    res = {"initials": pd.DataFrame({"Settlement X Possitions":
                                     m.settlement_positions[0],
                                     "Settlement Y Possitions":
                                     m.settlement_positions[1],
                                     "Population": m.population}),
           "Parameters": pd.Series({key: getattr(m, key)
                                    for key in dir(Parameters)
                                    if not key.startswith('__')
                                    and not callable(key)})
           }

    # Run model

    if test:
        steps = 1
    m.run(steps)

    # Save results

    res["trajectory"] = m.get_trajectory()
    res["traders trajectory"] = m.get_traders_trajectory()

    try:
        with open(filename, 'wb') as dumpfile:
            cP.dump(res, dumpfile)
            return 1
    except IOError:
        return -1


def run_experiment(argv):
    """
    Take arv input variables and run experiment accordingly.
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
    argv: list[N]
        List of parameters from terminal input

    Returns
    -------
    rt: int
        some return value to show whether sub_experiment succeeded
        return 1 if sucessfull.
    """

    # Parse switches from input
    global test

    if len(argv) > 1:
        test = bool(int(argv[1]))

    test_folder = ['', 'test_output/'][int(test)]
    experiment_folder = 'X5_trade_revised/'
    raw = 'raw_data/'
    res = 'results/'

    if getpass.getuser() == "kolb":
        save_path_raw = "/p/tmp/kolb/Mayasim/output_data/{}{}{}".format(
            test_folder, experiment_folder, raw)
        save_path_res = "/home/kolb/Mayasim/output_data/{}{}{}".format(
            test_folder, experiment_folder, res)
    elif getpass.getuser() == "jakob":
        save_path_raw = "/home/jakob/Project_MayaSim/Python/" \
                        "output_data/{}{}{}".format(test_folder,
                                                    experiment_folder, raw)
        save_path_res = "/home/jakob/Project_MayaSim/Python/" \
                        "output_data/{}{}{}".format(test_folder,
                                                    experiment_folder, res)
    else:
        save_path_res = './{}'.format(res)
        save_path_raw = './{}'.format(raw)

    name1 = "trajectory"
    estimators1 = {"mean_trajectories":
                  lambda fnames: pd.concat([np.load(f, allow_pickle=True)["trajectory"]
                                            for f in fnames]).groupby(
                      level=0).mean(),
                  "sigma_trajectories":
                  lambda fnames: pd.concat([np.load(f, allow_pickle=True)["trajectory"]
                                            for f in fnames]).groupby(
                          level=0).std()
                  }
    name2 = "traders_trajectory"
    estimators2 = {
                  "mean_trajectories":
                      lambda fnames:
                      pd.concat([np.load(f, allow_pickle=True)["traders trajectory"]
                                            for f in fnames]).groupby(
                          level=0).mean(),
                  "sigma_trajectories":
                      lambda fnames:
                      pd.concat([np.load(f, allow_pickle=True)["traders trajectory"]
                                            for f in fnames]).groupby(
                          level=0).std()
                  }

    # "precipitation_variation" is within the range of [-0.18, 0.06]
    # zero rain at some point would therefore imply a precipitation
    # amplitude of 5.55.
    precip_amplitudes = [0., 1., 2., 3., 4., 5.] \
        if not test else [0., 1.]
    # default r_trade is 6000.
    r_trades = [3000., 4000., 5000., 6000., 7000., 8000., 9000., 10000.] \
        if not test else [6000., 8000.]
    kill_cities = [True, False]

    parameter_combinations = list(it.product(precip_amplitudes,
                                             r_trades,
                                             kill_cities))


    index = {0: 'precip_amplitude',
             1: 'r_trade',
             2: 'kill_cropless'}
    sample_size = 5 if not test else 2

    if test:
        print('testing {}'.format(experiment_folder))
    h = handle(sample_size=sample_size,
               parameter_combinations=parameter_combinations,
               index=index,
               path_raw=save_path_raw,
               path_res=save_path_res,
               use_kwargs=True)

    h.compute(run_func=run_function)
    h.resave(eva=estimators1, name=name1)
    h.resave(eva=estimators2, name=name2)

    if test:
        data = pd.read_pickle(save_path_res + name1 + '.pkl')
        print(data.head())
        data = pd.read_pickle(save_path_res + name2 + '.pkl')
        print(data.head())
        print(save_path_res)

    return 1


if __name__ == "__main__":

    run_experiment(sys.argv)
