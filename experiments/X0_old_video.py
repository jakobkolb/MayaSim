"""
This experiment is dedicated to look at single runs with different parameters
for ecosystem and agriculture benefits and make videos from them to grasp what
actually happens.

In addition, I would like to plot some macro trajectories of these runs and
compare them to see, if there are some more signatures, that can be discovered.

One result of these experiments is (so far) that settlements rarely die back.
Population peaks, but settlement number stays at its peak values.
"""

import getpass
import glob
import itertools as it
import numpy as np
import os
import shutil
import sys

import imageio
import pandas as pd
from Python.pymofa import experiment_handling as eh

from Python.MayaSim.model import Model
from Python.MayaSim.model import Parameters
from Python.MayaSim.visuals.custom_visuals import SnapshotVisuals as Visuals


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

# Experiment with ecosystem benefits calculated as sum over cells in influence
def RUN_FUNC(r_eco, r_agg, mode, N, t_max, filename):
    """
    run function sets up experiment for different parameters of ecosysten and
    agriculture income. Each experiment saves snapshots of its state (in terms
    of images) to the folder named 'filename'
    These images are collected afterwards.

    Parameters
    ----------
    r_eco : float
        weight of ecosystem income in gdp/capita
    r_agg : float
        weight of agriculture income in gdp/capita
    mode : string
        one of ['sum', 'mean'] sets the mode for income calculation
        from land use
    N : int
        number of initial settlements
    t_max : int
        number of time steps to run the model
    filename : string, path like
        path that is used to save the model output

    Returns
    -------

    """

    if os.path.exists(filename):
        shutil.rmtree(filename)
    os.makedirs(filename)
    location = filename
    m = Model(n=N, output_data_location=location, interactive_output=True)
    m.output_level = 'trajectory'
    m.population_control = False
    m.eco_income_mode = mode
    m.crop_income_mode = mode
    m.r_es_sum = r_eco
    m.r_bca_sum = r_agg
    m.precipitation_modulation = True

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

    # Saves a series of image_{:03d}.jpg images of the run to this location
    # as well as the trajectory of macro quantities
    m.run(t_max)

    return 1

# define evaluation function for video assembly

def resaving(handle):

    def make_video(filename):

        init_data = np.load(filename[0] + '/init_frame.pkl')
        init_data['location'] = SAVE_PATH_RAW
        vis = Visuals(**init_data)

        filenames = np.sort(glob.glob(filename[0] + '/frame*'))
        writer = imageio.get_writer(filename[0] + '/run.mp4', fps=10)

        for i, f in enumerate(filenames):
            fig = vis.update_plots(**np.load(f))
            fig.set_size_inches([944 / fig.dpi, 672 / fig.dpi])
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                 sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(data)
            progress(i, len(filenames))
        writer.close()

        return open(filename[0] + '/run.mp4').read()

    EVA1 = {"video":
               make_video}

    EVA2 = {"trajectory":
               lambda fnames: pd.concat([np.load(f + '/trajectory.pkl')
                                         for f in fnames])}
    handle.resave(EVA2, NAME2)
    handle.resave(EVA1, NAME1)




# check which machine we are on and set paths accordingly

if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Mayasim/output_data/X6"
    SAVE_PATH_RES = "/home/kolb/Mayasim/output_data/X6"
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = "/home/jakob/PhD/Project_MayaSim/Python/output_data/raw/X6"
    SAVE_PATH_RES = "/home/jakob/PhD/Project_MayaSim/Python/output_data/X6"
else:
    SAVE_PATH_RAW = "./RAW"
    SAVE_PATH_RES = "./RES"

print(SAVE_PATH_RAW)
print(SAVE_PATH_RES)

# get parameters from command line and set sub experiment
# and testing mode accordingly

if len(sys.argv) > 1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0
if len(sys.argv) > 2:
    testing = bool(sys.argv[2])
else:
    testing = False

# set experiment parameters

N, tmax, r_eco, r_bca = [30], [500], [0.2], [0.0002]
r_bca_m, r_eco_m = [1.1], [10.]

r_bcas = [0.1, 0.15, 0.2, 0.25, 0.3]
r_ecos = [0.0001, 0.00015, 0.0002, 0.00025]

# set up experiment specific paths and parameter combinations

if testing:
    tmax, r_bcas, r_ecos = [10], [0.2], [.0002]
    SAVE_PATH_RAW += '_test'
    SAVE_PATH_RES += '_test'

if sub_experiment == 0:
    PARAM_COMBS = list(it.product(r_ecos, r_bcas, ['sum'], N, tmax))
    SAVE_PATH_RAW += '_eco_sum'
    SAVE_PATH_RES += '_eco_sum'
elif sub_experiment == 1:
    PARAM_COMBS = list(it.product(r_eco_m, r_bca_m, ['mean'], N, tmax))
    SAVE_PATH_RAW += '_eco_mean'
    SAVE_PATH_RES += '_eco_mean'

NAME1 = "video"
NAME2 = "trajectory"

INDEX = {0: "r_eco", 1: "r_bca"}
SAMPLE_SIZE = 1

# make folders, run experiment and data collection

if sub_experiment in [0, 1]:

    for path in [SAVE_PATH_RES, SAVE_PATH_RAW]:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                print(path, ' already existed')
        path += "/"

    handle = eh.experiment_handling(SAMPLE_SIZE,
                                    PARAM_COMBS,
                                    INDEX, SAVE_PATH_RAW,
                                    SAVE_PATH_RES)

    #handle.compute(RUN_FUNC)
    resaving(handle)


# unpack video collection and save in separate files,
# analyse trajectory data.

elif sub_experiment == 2:
    for experiment in ['_eco_sum', '_eco_mean']:
        data_found = False
        loc = SAVE_PATH_RES + experiment + '/' + NAME1

        try:
            data = np.load(loc)
            data_found = True
        except IOError:
            print(loc + ' does not exist')
        if data_found:
            names = data.index.names
            values = data.index.values

            for index in values:
                filename = '{}={}_{}={}.mp4'.format(names[0], index[0],
                                                    names[1], index[1])
                with open(SAVE_PATH_RES + experiment
                                  + '/' + filename, 'wb') as of:
                    of.write(data.xs(index)[0])

        data_found = False
        loc =  SAVE_PATH_RES + experiment + '/' + NAME2
        try:
            data = np.load(loc)
            data_found = True
        except IOError:
            print(loc + ' does not exist')

        if data_found:
            print data
