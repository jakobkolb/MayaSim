"""
test script for the model itself.
Just checking whether it runs without
errors, no sanity check so far
"""
import os
import shutil

import matplotlib.pyplot as plt

from mayasim.model.ModelCore import ModelCore as M

def test_ModelCore():
    """
    test run of Model class, saving a trajectory plot to
    'MayaSim/output/test_model/trajectory_plot.png'

    TODO:
    - split up into testing single methods separately
    - write assertions for all properties calculated 
      within test
    """
    N = 30
    timesteps = 1

    # define saving location
    comment = "test_model"
    location = "output/" + comment + '/'

    if os.path.exists(location):
        shutil.rmtree(location)
    os.makedirs(location)

    # initialize Model
    model = M(n=N,
              output_trajectory=True,
              output_settlement_data=True,
              output_geographic_data=True,
              output_data_location=location)

    # run Model
    model.crop_income_mode = 'sum'
    model.r_es_sum = 0.0001
    model.r_bca_sum = 0.25
    model.population_control = 'False'
    model.run(timesteps)

    # get trajectory
    trj = model.get_trajectory()
    measures = [
        'time',
        'total_population', 
        'total_settlements', 
        'total_migrants'
        ]

    # plot trajectory
    fig, axes = plt.subplots(ncols=len(measures)-1, figsize=(16, 2))
    for i, meas in enumerate(measures[1:]):
        trj.plot('time', y=meas, ax=axes[i], title=meas)

    fig.savefig(location + 'trajectory_plot.png')

    assert trj.shape[0] == timesteps
