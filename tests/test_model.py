"""
test script for the model itself.
Just checking whether it runs without
errors, no sanity check so far

TODO:
- split up into testing single methods separately
- write assertions for all properties calculated
  within test
"""
import os
import shutil

import matplotlib.pyplot as plt

from mayasim.model.core import Core as Model


def test_model_output():
    """
    test run of Model class, saving a plot of some aggregate
    values to 'MayaSim/output/test_model/aggregates_plot.png'
    """
    n = 30
    timesteps = 3

    # define saving location
    comment = "test_model"
    location = "output/" + comment + '/'

    if os.path.exists(location):
        shutil.rmtree(location)
    os.makedirs(location)

    # initialize Model
    model = Model(n, output_path=location)

    # run Model
    model.crop_income_mode = 'sum'
    model.r_es_sum = 0.0001
    model.r_bca_sum = 0.25
    model.population_control = 'False'
    model.run(timesteps)

    # get aggregates
    trj = model.get_aggregates()
    measures = [
        'time',
        'total_population',
        'total_settlements',
        'total_migrants'
    ]

    # plot aggregates
    fig, axes = plt.subplots(ncols=len(measures) - 1, figsize=(16, 2))
    for i, meas in enumerate(measures[1:]):
        trj.plot('time', y=meas, ax=axes[i], title=meas)

    fig.savefig(location + 'aggregates_plot.png')

    assert trj.shape[0] == timesteps
