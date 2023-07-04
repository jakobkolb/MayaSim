"""Experiment to test the correction of calculation of income from
agriculture and ecosystem services.

This experiment tests the influence of prefactors in income
calculation (r_bca and r_es) for two scenarios:
1) for income calculated as **mean** over income from cells
2) for income calculated as **sum** over income from cells

Explanation
-----------

Previously, the income from aggriculture and ecosystem services for each city
was calculated as the mean of the income from cells which it had under its
controll.

This does not make sense, since the actual harvest is not the mean of the
harvest of different places, but obviously the sum of the harvest from
different places.

Therefore, I changed the calculation of these sources of income to calculate
the sum over different cells.

Then, to get reasonable results, one has to adjust the prefactors in the
calculation of total income, since they have been tailored to reproduce
stylized facts before (and therefore must be taylored to do so again, just
differently)


"""

from __future__ import print_function
try:
    import cPickle as cp
except ImportError:
    import pickle as cp
import getpass
import itertools as it
import numpy as np
import sys
import pandas as pd

from pymofa.experiment_handling import experiment_handling as eh
from mayasim.model.ModelCore import ModelCore as Model
from mayasim.model.ModelParameters import ModelParameters as Parameters

test = True


def run_function(r_bca=0.2, r_eco=0.0002, population_control=False,
                 N=30, crop_income_mode='sum',
                 kill_cropless=True, steps=350, filename='./'):
    """
    Set up the Model for different Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    r_bca : float > 0
        the pre factor for income from agriculture
    population_control : boolean
        determines whether the population grows
        unbounded or if population growth decreases
        with income per capita and population density.
    N : int > 0
        initial number of settlements on the map
    crop_income_mode : string
        defines the mode of crop income calculation.
        possible values are 'sum' and 'mean'
    kill_cropless: bool
        Switch to determine whether or not to kill cities
        without cropped cells.
    filename: string
        path to save the results to.
    """

    # initialize the Model

    m = Model(N, output_data_location=filename, debug=test)
    if not filename.endswith('s0.pkl'):
        m.output_geographic_data = False
        m.output_settlement_data = False

    m.population_control = population_control
    m.crop_income_mode = crop_income_mode
    m.r_bca_sum = r_bca
    m.r_es_sum = r_eco
    m.kill_cities_without_crops = kill_cropless

    # store initial conditions and Parameters

    res = {}
    res["initials"] = pd.DataFrame({"Settlement X Possitions":
                                    m.settlement_positions[0],
                                    "Settlement Y Possitions":
                                    m.settlement_positions[1],
                                    "Population": m.population})

    res["Parameters"] = pd.Series({key:
                                   getattr(m, key)
                                   for key in dir(Parameters)
                                   if not key.startswith('__')
                                   and not callable(key)})

    # run Model

    if test:
        m.run(1)
    else:
        m.run(steps)

    # Retrieve results

    res["trajectory"] = m.get_trajectory()
    res["traders trajectory"] = m.get_traders_trajectory()

    try:
        with open(filename, 'wb') as dumpfile:
            cp.dump(res, dumpfile)
            return 1
    except IOError:
        return -1


def run_experiment(argv):
    """
    Take arv input variables and run sub_experiment accordingly.
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

    global test

    # Parse switches from input
    if len(argv) > 1:
        test = int(argv[1])


    # Generate paths according to switches and user name

    test_folder = ['', 'test_output/'][int(test)]
    experiment_folder = 'X2_eco_income/'
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

        print(save_path_raw)

    # Generate parameter combinations

    index = {0: "r_bca",
             1: "r_eco",
             2: "kill_cropless"}

    if test == 0:
        r_bcas = [0.1, 0.15, 0.2, 0.25, 0.3]
        r_ecos = [0.0001, 0.00015, 0.0002, 0.00025]
        kill_cropless = [True, False]
        test=False
    if test == 1:
        r_bcas = [0.1, 0.3]
        r_ecos = [0.0001, 0.00025]
        kill_cropless = [True, False]
        test=True

    param_combs = list(it.product(r_bcas, r_ecos, kill_cropless))

    sample_size = 10 if not test else 2

    # Define names and callables for post processing

    name1 = "trajectory"

    estimators1 = {"mean_trajectories":
                      lambda fnames:
                      pd.concat([np.load(f, allow_pickle=True)["trajectory"]
                                 for f in fnames]).groupby(level=0).mean(),
                  "sigma_trajectories":
                      lambda fnames:
                      pd.concat([np.load(f, allow_pickle=True)["trajectory"]
                                 for f in fnames]).groupby(level=0).std()
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

    # Run computation and post processing.

    if test:
        print('testing {}'.format(experiment_folder))
    handle = eh(sample_size=sample_size,
                parameter_combinations=param_combs,
                index=index,
                path_raw=save_path_raw,
                path_res=save_path_res,
                use_kwargs=True)

    handle.compute(run_func=run_function)
    handle.resave(eva=estimators1, name=name1)
    handle.resave(eva=estimators2, name=name2)

    if test:
        data = pd.read_pickle(save_path_res + name1 + '.pkl')
        print(data.head())
        data = pd.read_pickle(save_path_res + name2 + '.pkl')
        print(data.head())

    return 1


if __name__ == '__main__':

    run_experiment(sys.argv)
