"""
Author: Fritz Kühlein

This experiment serves to assess population dynamics of
MayaSim on the settlement level by using MayaSim's settlement
data output.

For further qualitative insight, this script produces a
movie of the run simulations with help of the `visuals` module,
which more than doubles computation times. If not needed, the
respective line can simply be commented out.

This experiment was run locally on my own machine. It might be
valuable to create scripts like those for X1 to run it on the HPC
and get multiple samples for each setting (increase `sample_size`).
"""

# disable pylint invalid-name message on module level
# pylint: disable=invalid-name
# pylint: enable=invalid-name

import getpass

import pickle as pkl
import pandas as pd

from pymofa.experiment_handling import experiment_handling as eh
from mayasim.model.core import Core as Model
from mayasim.model.parameters import Parameters
from mayasim.visuals.custom_visuals import MapPlot

STEPS = None

def run_function(t_1, t_2, t_3, filename):
    """
    Set up the Model for default Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    n_init : int > 0
        initial number of settlements on the map,
    kill_cropless: bool
        switch to either kill settlements without crops or not
    steps: int
        number of steps to integrate the model for,
    filename: string
        path to save the results to.
    """

    # initialize the Model

    m = Model(output_path=filename)

    m.thresh_rank_1 = t_1
    m.thresh_rank_2 = t_2
    m.thresh_rank_3 = t_3

    m.better_ess = True
    m.precip_modulation = False

    # uncomment if movie processing is not needed
    # m.output_geographic_data = False

    # store initial conditions and Parameters

    res = {"initials": pd.DataFrame({"Settlement Positions":
                                     m.stm_positions,
                                     "Population":
                                     m.stm_population}),
           "Parameters": pd.Series({key: getattr(m, key)
                                    for key in dir(Parameters)
                                    if not key.startswith('__')
                                    and not callable(key)})}

    # run Model
    m.run(STEPS)

    # Retrieve results
    res["aggregates"] = m.get_aggregates()
    res["traders aggregates"] = m.get_traders_aggregates()

    try:
        with open(filename, 'wb') as dumpfile:
            pkl.dump(res, dumpfile)
            return 1
    except IOError:
        return -1


def run_experiment():
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

    global STEPS  # pylint: disable=global-statement

    # Generate paths according to switches and user name
    experiment_folder = 'X2_local_dynamics/'
    raw = 'raw_data/'
    res = 'results/'

    if getpass.getuser() == 'fritz':
        save_path_raw = '/Users/fritz/Desktop/Thesis/MayaSim/' \
                        f'output/{experiment_folder}{raw}'
        save_path_res = '/Users/fritz/Desktop/Thesis/MayaSim/' \
                        f'output/{experiment_folder}{res}'
    else:
        save_path_raw = f'./output/{experiment_folder}{raw}'
        save_path_res = f'./output/{experiment_folder}{res}'

    # Generate parameter combinations

    index = {0: "t_1", 1: "t_2", 2: "t_3"}
    trade_thresholds = [(1000000, 1000000, 1000000),    # suppressed trade
                        (4000, 7000, 9500)]             # default trade

    STEPS = 2000
    sample_size = 1

    # define post-processing function
    def plot_function(steps=1, output_location='./', fnames='./'):
        input_loc = fnames[0]
        if input_loc.endswith('.pkl'):
            input_loc = input_loc[:-4]

        tail = input_loc.rsplit('/', 1)[1]
        output_location += tail
        mp = MapPlot(t_max=steps,
                     input_location=input_loc,
                     output_location=output_location)

        mp.mplot()
        return 1

    name = "MapPlots"
    process_movie = {
        "map_plots": lambda fnames: plot_function(
            steps=STEPS,
            output_location=save_path_res,
            fnames=fnames)
    }

    # Run computation.
    handle = eh(sample_size=sample_size,
                parameter_combinations=trade_thresholds,
                index=index,
                path_raw=save_path_raw,
                path_res=save_path_res,
                use_kwargs=True)

    # compute
    handle.compute(run_func=run_function)
    # create plots and render movie (comment out if not needed)
    handle.resave(eva=process_movie, name=name, no_output=True)

    return 1


if __name__ == '__main__':

    run_experiment()
