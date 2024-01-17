"""
Author: Fritz Kühlein

Reproduce Jakob Kolb's regime classification wrt. parameters r_trade and r_es.

"Experiment to generate trajectories for different possible of income
from ecosystem and trade. These trajectories will then be used to
analyze the bifurcation parameter of the model. Also, they can be
used to find the optimal values for r_trade and r_es that reproduce
the original behavior of the model in Heckberts publication.

Parameters that are varied are
1) r_trade
2) r_es
The model runs for t=2000 time steps as I hope that this will be short enough
to finish on the cluster in 24h and long enough to provide enough data for the
analysis of frequency of oscillations from the trajectories."
"""

import argparse
import getpass
import itertools as it
import os

import pickle as cp
import numpy as np
import pandas as pd
from pymofa.experiment_handling import experiment_handling as eh

from mayasim.model.core import Core as MayaSim
from mayasim.model.parameters import Parameters
from mayasim.visuals.custom_visuals import MapPlot

STEPS = 1500

def ens_mean(fnames):
    """calculate the mean of all files in fnames over time steps

    For each file in fnames, load the file with the load function, check,
    if it actually loaded and if so, append it to the list of data frames.
    Then concatenate the data frames, group them by time steps and take the
    mean over values for the same time step.
    """
    dfs = []

    for f in fnames:
        df = np.load(f, allow_pickle=True)

        dfs.append(
            df["aggregates"][['total_population',
                              'forest_state_3_cells']].astype(float))

    return pd.concat(dfs).groupby(level=0).mean()


def ens_sigma(fnames):
    """calculate the mean of all files in fnames over time steps

    For each file in fnames, load the file with the load function, check,
    if it actually loaded and if so, append it to the list of data frames.
    Then concatenate the data frames, group them by time steps and take the
    standard deviation over values for the same time step.
    """
    dfs = []

    for f in fnames:
        df = np.load(f, allow_pickle=True)

        dfs.append(
            df["aggregates"][['total_population',
                              'forest_state_3_cells']].astype(float))

    return pd.concat(dfs).groupby(level=0).std()


def collect(fnames):
    """
    collect all aggregate trajectories in fnames,
    concat them and save them to hdf store.

    pymofa will call this function for each parameter combination separately,
    so fnames will contain the filenames of all runs of one combination.
    """
    path_res = '/'.join(fnames[0].rsplit("/")[:-2]) + '/results'
    # print(f'saving to: {path_res}')

    # get parameter values from file names
    fnshort = fnames[0].rsplit('/', 1)[1].rsplit('.')[0]
    parts = fnshort.rsplit('-')

    if len(parts) == 3:
        [srtrade, sres, srest] = parts
    elif len(parts) == 4:
        srtrade = parts[0]
        sres = f'{parts[1]}-{parts[2]}'
        srest = parts[3]
    stest, srunid = srest.split('_')
    r_trade = int(srtrade)
    r_es = float(sres.replace('o', '.'))
    test = bool(stest)

    # load data from all files for given parameters and combine them in one
    # data frame.
    dfs = []  # list of data frames for results
    for i, f in enumerate(fnames):
        # load data
        data = np.load(f, allow_pickle=True)
        # get trajectory dataframe from results
        df = data["aggregates"][['total_population']]
        # generate multiindex with parameter values
        index = pd.MultiIndex.from_product(
            [[r_trade], [r_es], [test], [i], df.index.values],
            names=['r_trade', 'r_es', 'test', 'run_id', 'step'])
        df.index = index
        # append data to list of result data frames
        dfs.append(df)

    # put data together
    dfa = pd.concat(dfs)

    # write results to hdf (appending if file already exists)
    with pd.HDFStore(path_res + '/all_trjs.hd5') as store:
        store.put('d1',
                  dfa.astype(float),
                  append=True,
                  format='table',
                  data_columns=True)


def run_function(r_bca=0.2,
                 r_es=0.0002,
                 r_trade=6000,
                 population_control=False,
                 n=30,
                 crop_income_mode='sum',
                 better_ess=True,
                 kill_cropless=False,
                 test=False,
                 filename='./'):
    """
    Set up the Model for different Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    d_times: list of lists
        list of list of start and end dates of droughts
    d_severity : float
        severity of drought (decrease in rainfall in percent)
    r_bca : float > 0
        the prefactor for income from agriculture
    r_es : float
        the prefactor for income from ecosystem services
    r_trade : float
        the prefactor for income from trade
    population_control : boolean
        determines whether the population grows
        unbounded or if population growth decreases
        with income per capita and population density.
    n : int > 0
        initial number of settlements on the map
    crop_income_mode : string
        defines the mode of crop income calculation.
        possible values are 'sum' and 'mean'
    better_ess : bool
        switch to use forest as proxy for income from eco
        system services from net primary productivity.
    kill_cropless: bool
        Switch to determine whether or not to kill cities
        without cropped cells.
    filename: string
        path to save the results to.
    """

    # initialize the Model

    m = MayaSim(n, output_path=filename)
    m.output_geographic_data = False
    m.output_settlement_data = False

    m.population_control = population_control
    m.crop_income_mode = crop_income_mode
    m.better_ess = better_ess
    m.r_bca_sum = r_bca
    m.r_es_sum = r_es
    m.r_trade = r_trade
    m.kill_stm_without_crops = kill_cropless

    m.precipitation_modulation = False

    # store initial conditions and Parameters

    res = {
        "initials":
        pd.DataFrame({
            "Settlement Positions": m.stm_positions,
            "Population": m.stm_population
        }),
        "Parameters":
        pd.Series({
            key: getattr(m, key)
            for key in dir(Parameters)
            if not key.startswith('__') and not callable(key)
        })
    }

    # run Model
    m.run(STEPS)

    # Retrieve results
    res["aggregates"] = m.get_aggregates()

    # and save to file
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)

    # pymofa needs return value
    return 1


def run_experiment(test, mode):
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
        return 1 if successful.
    """

    global STEPS

    # Generate paths according to switches and user name

    test_folder = 'test_experiments/' if test else ''
    experiment_folder = 'x11_dynamical_regimes/'
    raw = 'raw_data/'
    res = 'results/'

    if getpass.getuser() == 'fritz':
        save_path_raw = '/Users/Fritz/Desktop/Thesis/MayaSim/' \
                        f'output/{test_folder}{experiment_folder}{raw}'
        save_path_res = '/Users/Fritz/Desktop/Thesis/MayaSim/' \
                        f'output/{test_folder}{experiment_folder}{res}'
    else:
        save_path_raw = f'./output/{test_folder}{experiment_folder}{raw}'
        save_path_res = f'./output/{test_folder}{experiment_folder}{res}'

    # Generate parameter combinations

    index = {0: "r_trade", 1: "r_es", 2: "test"}

    if test:
        r_trade = [6000, 7000]
        r_es = [0.03, 0.08]
    else:
        r_trade = [round(x, 5) for x in np.arange(5000, 9400, 100)]
        r_es = [round(x, 4) for x in np.arange(0.02, 0.13, 0.0025)]
    print(f'r_trade: {r_trade}')
    print(f'r_es: {r_es}')
    param_combs = list(it.product(r_trade, r_es, [test]))

    STEPS = 2000 if not test else 20
    sample_size = 31 if not test else 4

    print(f'parameter combinations: {len(param_combs)}, samples: {sample_size}')

    # Define names and callables for post processing

    name1 = "ensemble_means"
    estimators1 = {"<ensemble_mean>": ens_mean, "<ensemble_sigma>": ens_sigma}

    name2 = "all_trajectories"
    estimators2 = {"trajectory_list": collect}

    # Run computation and post processing.

    handle = eh(sample_size=sample_size,
                parameter_combinations=param_combs,
                index=index,
                path_raw=save_path_raw,
                path_res=save_path_res,
                use_kwargs=True)

    if mode == 0:
        # compute experiment
        handle.compute(run_func=run_function)
    elif mode == 1:
        # calculate ensemble means (and std)
        handle.resave(eva=estimators1, name=name1)
    elif mode == 2:
        # collect all trajectories and save to hdf/pkl
        handle.resave(eva=estimators2, name=name2, no_output=True)

    return 1


if __name__ == '__main__':

    # parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--testing",
                    dest='test',
                    action='store_true',
                    help="switch to production vs. testing mode")
    ap.add_argument("--production",
                    dest='test',
                    action='store_false',
                    help="switch to production vs. testing mode")
    ap.set_defaults(test=False)
    ap.add_argument("-m",
                    "--mode",
                    type=int,
                    help="""switch to set mode: 0 - computation,
    1 - calculate means, 2 - collect trajectories to single file""",
                    default=0,
                    choices=[0, 1, 2, 3])
    args = vars(ap.parse_args())

    # run experiment
    print(args)
    run_experiment(**args)
