"""
Experiment to TEST the influence of drought events.
Drought events start once the civilisation has reached
a 'complex society' state (after 200 years) and vary
in length and severity from 0 to 100 years and 0 to 100%
less precipitation.

Therefore, starting point is at t = 200 where the model has
reached a complex society state in all previous studies.
We also use parameters for income from trade, agriculture and
ecosystem services, that have previously proven to lead to
some influence of precipitation variability on the state of the
system.

Also, we vary the parameter for income from trade so see, if
there is a certain parameter value, that results in
stable complex society for some drought events and collapse for others.
"""

import getpass
import itertools as it
import pickle as cp
import sys
import os

import numpy as np
import pandas as pd
from pymofa.experiment_handling import experiment_handling as eh

from mayasim.model.ModelCore import ModelCore as Model
from mayasim.model.ModelParameters import ModelParameters as Parameters

TEST = True


def load(fname):
    """ try to load the file with name fname.

    If it worked, return the content. If not, return -1
    """
    try:
        return np.load(fname, allow_pickle=True)
    except OSError:
        try:
            os.remove(fname)
        except:
            print(f"{fname} couldn't be read or deleted")
    except IOError:
        try:
            os.remove(fname)
        except:
            print(f"{fname} couldn't be read or deleted")


def magg(fnames):
    """calculate the mean of all files in fnames over time steps

    For each file in fnames, load the file with the load function, check,
    if it actually loaded and if so, append it to the list of data frames.
    Then concatenate the data frames, group them by time steps and take the
    mean over values for the same time step.
    """
    dfs = []

    for fname in fnames:
        dft = load(fname)

        if dft is not None:
            dfs.append(dft["trajectory"].astype('float'))

    return pd.concat(dfs).groupby(level=0).mean()


def sagg(fnames):
    """calculate the mean of all files in fnames over time steps

    For each file in fnames, load the file with the load function, check,
    if it actually loaded and if so, append it to the list of data frames.
    Then concatenate the data frames, group them by time steps and take the
    standard deviation over values for the same time step.
    """
    dfs = []

    for fname in fnames:
        dft = load(fname)

        if dft is not None:
            dfs.append(dft["trajectory"].astype('float'))

    return pd.concat(dfs).groupby(level=0).std()


def trj_list(fnames):
    """load all available trajectories and return them as a list"""
    trajectory_list = []

    for fname in fnames:
        dft = load(fname)

        if dft is not None:
            trajectory_list.append(dft["trajectory"].astype('float'))

    return trajectory_list


def run_function(d_start=200,
                 d_length=20,
                 d_severity=50.,
                 r_bca=0.2,
                 r_es=0.00012,
                 r_trade=6000,
                 population_control=False,
                 n=30,
                 crop_income_mode='sum',
                 better_ess=True,
                 kill_cropless=False,
                 steps=1000,
                 filename='./'):
    """
    Set up the Model for different Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    d_start : int
        starting point of drought in model time
    d_length : int
        length of drought in timesteps
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

    model = Model(n, output_data_location=filename, debug=TEST)

    if not filename.endswith('s0.pkl'):
        model.output_geographic_data = False
        model.output_settlement_data = False

    model.population_control = population_control
    model.crop_income_mode = crop_income_mode
    model.better_ess = better_ess
    model.r_bca_sum = r_bca
    model.r_es_sum = r_es
    model.r_trade = r_trade
    model.kill_cities_without_crops = kill_cropless

    model.precipitation_modulation = False
    model.drought_times = [[d_start, d_start + d_length]]
    model.drought_severity = d_severity

    # store initial conditions and Parameters

    res = {
        "initials":
        pd.DataFrame({
            "Settlement X Possitions": model.settlement_positions[0],
            "Settlement Y Possitions": model.settlement_positions[1],
            "Population": model.population
        }),
        "Parameters":
        pd.Series({
            key: getattr(model, key)

            for key in dir(Parameters)

            if not key.startswith('__') and not callable(key)
        })
    }

    # run Model

    if TEST:
        model.run(3)
    else:
        model.run(steps)

    # Retrieve results

    res["trajectory"] = model.get_trajectory()

    res["final_climax_cells"] = np.sum(model.forest_state == 3)
    res["final_regrowth_cells"] = np.sum(model.forest_state == 2)
    res["final_cleared_cells"] = np.sum(model.forest_state == 1)
    res["final_aggriculture_cells"] = sum(model.number_cropped_cells)

    res["final population"] = sum(model.population)
    res["final trade links"] = sum(model.degree) / 2.
    res["final max cluster size"] = model.max_cluster_size

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

    global TEST

    # Parse switches from input

    if len(argv) > 1:
        TEST = bool(int(argv[1]))
    else:
        TEST = True
    if len(argv) > 2:
        mode = int(argv[2])
    else:
        mode = None

    if len(argv) > 3:
        job_id = int(argv[3])
    else:
        job_id = 1

    if len(argv) > 4:
        max_id = int(argv[4])
    else:
        max_id = 1

    # Generate paths according to switches and user name

    test_folder = 'test_output/' if TEST else ''
    experiment_folder = 'X9_stability_analysis/'
    raw = 'raw_data/'
    res = 'results/'

    if getpass.getuser() == "kolb":
        save_path_raw = "/p/tmp/kolb/Mayasim/output_data/{}{}{}".format(
            test_folder, experiment_folder, raw)
        save_path_res = "/home/kolb/Mayasim/output_data/{}{}{}".format(
            test_folder, experiment_folder, res)
    elif getpass.getuser() == "jakob":
        save_path_raw = \
            "/home/jakob/Project_MayaSim/Python/" \
            "output_data/{}{}{}".format(test_folder, experiment_folder, raw)
        save_path_res = \
            "/home/jakob/Project_MayaSim/Python/" \
            "output_data/{}{}{}".format(test_folder, experiment_folder, res)
    else:
        save_path_res = './{}'.format(res)
        save_path_raw = './{}'.format(raw)

    # Generate parameter combinations

    if not TEST:
        d_length = list(range(0, 105, 5))
        d_severity = list(range(0, 105, 5))

        # parameters for high income from trade
        # d_length = list(range(0, 105, 5))
        # d_severity = list(range(0, 105, 5))
        r_trade = [8000]  # hight trade income generates stable complex state
        t_start = [400]  # start drought after system has settled
        param_combs_high = list(
            it.product(d_length, d_severity, r_trade, t_start))

        # parameters for low income from trade
        # d_length = list(range(0, 105, 5))
        # d_severity = list(range(0, 105, 5))
        r_trade = [6000]  # low trade income generates oscillating states
        t_start = [400, 550]  # One at first low, one at first hight after
        # overshoot
        param_combs_low = list(
            it.product(d_length, d_severity, r_trade, t_start))

        # put all parameters together
        param_combs = param_combs_high + param_combs_low
    else:
        d_length = [20]
        d_severity = [0., 60.]
        r_trade = [6000]
        t_start = [350]  # start drought after system has settled
        param_combs = list(it.product(d_length, d_severity, r_trade, t_start))

    index = {0: "d_length", 1: "d_severity", 2: "r_trade", 3: "d_start"}

    print(f'computing results for {len(param_combs)} parameter combinations')
    print(len(param_combs), max_id)

    if len(param_combs) % max_id != 0:
        print(
            f'number of jobs ({len(param_combs)}) has to be multiple of max_id ({max_id})!!'
        )
        exit(-1)

    sample_size = 31 if not TEST else 3

    # Define names and callables for post processing

    name1 = "trajectory"
    estimators1 = {"<mean_trajectories>": magg, "<sigma_trajectories>": sagg}
    name2 = "all_trajectories"
    estimators2 = {"trajectory_list": trj_list}

    def final_states(fnames, keys):
        """load results and collect all that are specified by keys to return
        them in a pandas dataframe
        """
        key = keys[0]
        data = []

        for fname in fnames:
            dft = load(fname)

            if dft is not None:
                data.append(dft[key])
        dfo = pd.DataFrame(data=data, columns=[keys[0]])

        for key in keys[1:]:
            data = []

            for fname in fnames:
                dft = load(fname)

                if dft is not None:
                    data.append(dft[key])
            dfo[key] = data

        return dfo

    name3 = "all_final_states"
    estimators3 = {
        "final states":
        lambda fnames: final_states(fnames, [
            "final population", "final trade links", "final max cluster size",
            "final_climax_cells", "final_regrowth_cells", "final_cleared_cells"
        ])
    }

    # Run computation and post processing.

    cl = int(len(param_combs) / max_id)
    i = (job_id - 1) * cl
    j = job_id * cl

    handle = eh(sample_size=sample_size,
                parameter_combinations=param_combs[i:j],
                index=index,
                path_raw=save_path_raw,
                path_res=save_path_res,
                use_kwargs=True)

    if mode == 1:
        handle.compute(run_func=run_function)

        return 0
    elif mode == 2:
        handle.resave(eva=estimators1, name=name1)
        handle.resave(eva=estimators2, name=name2)
        handle.resave(eva=estimators3, name=name3)

        return 0

    return 1


if __name__ == '__main__':

    run_experiment(sys.argv)
