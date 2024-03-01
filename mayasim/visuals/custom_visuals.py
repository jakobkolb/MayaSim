import glob
import os
from subprocess import run
from tqdm.auto import trange

import matplotlib as mpl
# Force matplotlib to not use any Xwindows backend
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize


class MapPlot():
    """Plot simulation state on map."""

    def __init__(self,
                 t_max,
                 output_location,
                 input_location):
        """
        Initialize the MapPlot class with custom parameters
        
        Parameters
        ----------
        t_max: int
            max timestep to plot
        output_location: basestring
            output location for MapPlot frames
        input_location: basestring
            input location for simulation data to plot
        """

        self.trj = pd.read_pickle(input_location + '.pkl')['aggregates']

        self.t_max = t_max
        if not input_location.endswith('/'):
            input_location += '/'
        self.input_location = input_location
        if not output_location.endswith('/'):
            output_location += '/'
        self.output_location = output_location

        if not os.path.isdir(self.output_location):
            os.makedirs(self.output_location)
            os.makedirs(self.output_location + 'frames/')

        return

    def mplot(self):
        """Plot map frames from simulation data"""

        fig = plt.figure(figsize=(16, 9))

        for t in trange(1, self.t_max+1):

            axa = plt.subplot2grid((2, 3), (0, 2))
            axb = plt.subplot2grid((2, 3), (1, 2))
            axa2 = axb.twinx()

            axa.set_xlim([min(self.trj['forest_state_3_cells']), max(self.trj['forest_state_3_cells'])])
            axa.set_ylim([min(self.trj['total_population']), max(self.trj['total_population'])])
            axa2.set_xlim([1, max(self.trj['time'])])
            axa2.set_ylim([0,
                           max([max(self.trj['total_income_agriculture']),
                                max(self.trj['total_income_ecosystem']),
                                max(self.trj['total_income_trade'])
                                ])
                           ])

            #print(self.trj.columns)
            time = range(0, t)
            population = self.trj['total_population'][0:t]
            agg_income = self.trj['total_income_agriculture'][0:t]
            trade_income = self.trj['total_income_trade'][0:t]
            ess_income = self.trj['total_income_ecosystem'][0:t]
            climax_forest = self.trj['forest_state_3_cells'][0:t]

            ln1 = axa.plot(climax_forest, population, color='blue', label='total population')
            ln1b = axa.scatter(climax_forest.values[-1], population.values[-1], color='blue')

            axa.set_xlabel('climax forest cells')
            axa.set_ylabel('total population')

            ln2 = axa2.plot(time, agg_income, color='black', label='income from agriculture')
            ln3 = axa2.plot(time, ess_income, color='green', label='income from ecosystem')
            ln4 = axa2.plot(time, trade_income, color='red', label='income from trade')

            lns = ln2 + ln3 + ln4
            labs = [l.get_label() for l in lns]
            axb.legend(lns, labs, loc=0)

            forest_data = self.trj[['total_cropped_cells',
                                    'forest_state_1_cells',
                                    'forest_state_2_cells',
                                    'forest_state_3_cells']]
            # print('before: \n', forest_data['forest_state_1_cells'])
            forest_data.loc[:,'forest_state_1_cells'] = \
                forest_data.loc[:,'forest_state_1_cells'].sub(forest_data.loc[:,'total_cropped_cells'])
            # print('after: \n', forest_data['forest_state_1_cells'])
            forest_data[0:t].plot.area(ax=axb, stacked=True, color=['black', '#FF9900', '#66FF33', '#336600'])
            axb.set_xlim([0, self.t_max])
            axb.set_ylim([0, 102540])

            ax = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
            location = self.input_location + 'geographic_data_{0:03d}.pkl'.format(t)


            data = np.load(location, allow_pickle=True)

            forest = data['forest']

            # plot forest state

            shape = forest.shape

            cropped = np.zeros(shape)
            influenced = np.zeros(shape)

            cropped_cells = data['cropped cells']
            influenced_cells = data['cells in influence']

            for city, crp in enumerate(cropped_cells):
                if len(crp) > 0:
                    influenced[np.array(influenced_cells[city]).T[0], np.array(influenced_cells[city]).T[1]] = 1
                    cropped[np.array(cropped_cells[city]).T[0], np.array(cropped_cells[city]).T[1]] = 1

            forest[cropped == 1] = 4

            cmap1 = ListedColormap(['blue', '#FF9900', '#66FF33', '#336600', 'black'])
            norm = Normalize(vmin=0, vmax=4)
            image1 = ax.imshow(forest,
                               cmap=cmap1,
                               norm=norm,
                               interpolation='none',
                               alpha=0.9,
                               zorder=0)

            cmap2 = ListedColormap([(0, 0, 0), 'grey'])
            image2 = ax.imshow(influenced,
                               cmap=cmap2,
                               alpha=0.3,
                               zorder=0)

            # plot trade network from adjacency matrix and settlement positions

            for i, xi in enumerate(data['position']):
                for j, xj in enumerate(data['position']):
                    if data['adjacency'][i, j] == 1:
                        plt.plot([xi[1], xj[1]], [xi[0], xj[0]], linewidth=0.5, color='black', zorder=1)

            # plot settlements with population as color and rank as size

            max_population = self.trj['max_settlement_population'].max()

            cmap = plt.get_cmap('OrRd')
            sct = plt.scatter(np.array(data['position']).T[1],
                              np.array(data['position']).T[0],
                              [4 * (x + 1) for x in data['rank']],
                              c=data['population'],
                              cmap=plt.get_cmap('OrRd'),
                              edgecolors='black',
                              zorder=2,
                              vmax=max_population)
            fig.colorbar(sct, label='population')
            ax.set_ylim([shape[0], 0])
            ax.set_xlim([0, shape[1]])

            fig.tight_layout()
            ol = self.output_location + f'frames/frame_{t:03d}'
            fig.savefig(ol, dpi=150)
            fig.clear()
        try:
           self.moviefy()
        finally:
           pass

    def moviefy(self, rmold=False, namelist=['frame_']):
        # type: (str, str) -> object

        print('moviefying map plots ...')

        framerate = 8
        output_folder = self.output_location
        input_folder = self.output_location + 'frames/'

        input_folder = '/' + input_folder.strip('/') + '/'
        output_folder = '/' + output_folder.strip('/') + '/'

        # namelist = ['AG_', 'bca_', 'es_',
        #             'forest', 'influence_cropped_b_',
        #             'npp_', 'pop_grad_', 'rain_',
        #             'soil_deg_', 'trade_network_',
        #             'waterflow_']

        for name in namelist:
            input_string = input_folder + name + "%03d.png"
            del_string = input_folder + name + "*.png"
            output_string = output_folder + 'movie.mp4'
            run(["ffmpeg", "-loglevel", "panic", "-y", "-hide_banner", "-r", repr(framerate), "-i", input_string, output_string])
            if rmold:
                for fl in glob.glob(del_string):
                    os.remove(fl)


class SnapshotVisuals(object):
    """
    Class containing routines to save system snapshots in terms of images.
    """

    def __init__(self, columns=['population', 'N_settlements', 'Malthus'],
                 shape=None, t_max=None, location='pics'):
        """
        Init function of the Visuals class. Saves parameters, creates figure.

        Parameters
        ----------
        columns : list of strings
            list of column names for the subplots that are not the map
        shape : tuple or its
            shape of the map
        t_max : int
            max number of timesteps to set xlim in trajectory plots.
            not implemented yet.
        location : string
            path where the snapshot images will be saved
        """

        # make saving location for plots, in case it does not exist
        self.location = location.rstrip('/')
        if not os.path.isdir(self.location):
            os.makedirs(self.location)

        # image/frame counter for naming
        self.i_image = 0
        self.shape = shape
        # shapes of subplot grid
        self.ylen = len(columns)
        self.xlen = self.ylen + 1
        self.columns = columns
        # list to save trajectory of ad hoc calculated macro quantities.
        self.trajectory = []

        # create figure with some small plots on the left
        # and a big plot on the right.
        self.figure = plt.figure(figsize=(11.7, 8.3))
        self.axes = []
        for c, column in enumerate(self.columns):
            self.axes.append(plt.subplot2grid((self.ylen, self.xlen),
                                              (c, 0)))
            self.axes[-1].set_title(column)
        self.axes.append(plt.subplot2grid((self.ylen, self.xlen),
                                          (0, 1), rowspan=self.xlen,
                                          colspan=self.xlen))

    def update_plots(self, population, real_income, ag_income, es_income,
                     trade_income, adjacency, settlement_positions):
        """
        Creates a snapshot plot from the parameters passed.

        Parameters
        ----------
        population: list of floats
            The population of all the existing settlements,
        real_income: list of floats
            The real income of all existing settlements,
        ag_income: list of floats
            The income from agriculture per capita for all settlements,
        es_income: list of floats
            The income from ecosystem services per capita for all settlements,
        trade_income: list of floats
            The income from trade per capita for all settlements,
        adjacency: ndarray
            The adjacency matrix of the trade network between settlements,
        settlement_positions: list of lists of floats
            The x and y positions of all settlements on the map.

        Returns
        -------
        self.figure: object
            the updated matplotlib figure object.

        """

        # increment frame counter
        self.i_image += 1
        # append ad hoc macro quantitites to the trajectory
        self.trajectory.append([sum(population), len(population)])

        # plot macro trajectories to subplots
        for c, column in enumerate(self.columns[:2]):
            c_data = [d[c] for d in self.trajectory]
            self.axes[c].clear()
            self.axes[c].plot(c_data)
            self.axes[c].set_title(column)
            self.axes[c].locator_params(nbins=3)

        # clear scatterplot and prepare colors
        # (fraction of ag income in total income)
        axm = self.axes[-2]
        axm.clear()
        cmap = mpl.cm.BrBG
        colors = [cmap(1. - float(ag_income[i]) / float(r_income))
                  if float(r_income) > 0 else (0, 0, 0, 1)
                  for i, r_income in enumerate(real_income)]

        # make scatterplot
        try:
            axm.scatter(population, real_income, c=colors)
        except ValueError:
            print('printing error!!!')
            print(population, real_income, colors)

        # adjust axis settings
        axm.locator_params(nbins=3)
        axm.set_xlim([0., 50000.])
        axm.set_ylim([0., 2.])
        axm.set_title(self.columns[-1])

        # clear trade network plot
        ax = self.axes[-1]
        ax.clear()
        ax.set_title('Trade-Network')
        x = settlement_positions[0] + 0.5
        y = settlement_positions[1] + 0.5

        # plot settlements with color acccording to trade income
        cmap = mpl.cm.Blues
        t_max = max(trade_income)
        if t_max > 0:
            colors = [cmap(t / t_max) for t in trade_income]
        else:
            colors = ['w' for t in trade_income]
        try:
            ax.scatter(y, x, c=colors, zorder=2)
        except ValueError:
            print(t_max, ' t_max might have weird values?')

        # plot trade network
        generator = (i for i, x in np.ndenumerate(adjacency) if
                     adjacency[i] == 1)
        for i, j in generator:
            ax.plot([y[i], y[j]], [x[i], x[j]], color="k", linewidth=0.5,
                    zorder=1)

        # set shape of map and save.
        if self.shape is not None:
            ax.set_xlim([0, self.shape[1]])
            ax.set_ylim([self.shape[0], 0])

        return self.figure


class TrajectoryVisuals(object):
    def __init__(self, location):
        """
        Initialize the Visual class for trajectory data.
        """

        tmp = np.load(location)
        statistics = tmp.columns.values
        self.trajectories = {stat: tmp[[stat]].unstack('observables')
                             for stat in statistics}
        for trj in self.trajectories.values():
            trj.columns = trj.columns.droplevel()
            print(trj.columns)
        self.run_indices = zip(*[i.tolist() for i in tmp.index.levels[:2]])

    def plot_trajectories(self, observables=[]):

        for obs in observables:
            for key in self.trajectories.keys():
                if key == 'mean':
                    self.trajectories[key][[obs]].unstack(level=(0, 1)).plot(legend=False)
                    plt.show()
