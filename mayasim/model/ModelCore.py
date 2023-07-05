from __future__ import print_function

import datetime
import operator
import os
import sys
import traceback
import warnings
from itertools import compress

import networkx as nx
import numpy as np
import pandas
import pkg_resources
import scipy.ndimage as ndimage
import scipy.sparse as sparse

from tqdm.auto import tqdm

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

if __name__ == "__main__":
    from ModelParameters import ModelParameters as Parameters
    from f90routines import f90routines
else:
    from .f90routines import f90routines
    from .ModelParameters import ModelParameters as Parameters



class ModelCore(Parameters):
    def __init__(self,
                 n=30,
                 output_data_location=None,
                 debug=False,
                 output_trajectory=True,
                 **kwargs):
        """
        Instance of the MayaSim model.

        Parameters
        ----------
        n: int
            number of settlements to initialize,
        output_data_location: path_like
            string stating the folder path to which the output
            files will be writen,
        debug: bool
            switch for debugging output from model,
        output_trajectory: bool
            switch for output of trajectory data,
        output_settlement_data: bool
            switch for output of settlement data,
        output_geographic_data: bool
            switch for output of geographic data.
        """

        # Input/Output settings:

        # Set path to static input files

        input_data_location = pkg_resources. \
            resource_filename('mayasim', 'input_data/')

        # Debugging settings
        self.debug = debug

        # In debug mode, always print stack for warnings and errors.
        def warn_with_traceback(message,
                                category,
                                filename,
                                lineno,
                                file=None,
                                line=None):

            log = file if hasattr(file, 'write') else sys.stderr
            traceback.print_stack(file=log)
            log.write(
                warnings.formatwarning(message, category, filename, lineno,
                                       line))

        if self.debug:
            warnings.showwarning = warn_with_traceback

        # *******************************************************************
        # MODEL PARAMETERS (to be varied)
        # *******************************************************************
        self.output_trajectory = output_trajectory

        # Settlement and geographic data will be written to files in each time step,
        # Trajectory data will be kept in one data structure to be read out, when
        # the model run finished.

        if output_data_location is not None:

            # remove file ending
            self.output_data_location = output_data_location.rsplit('.', 1)[0]

            # create callable output paths
            self.settlement_output_path = \
                lambda i: self.output_data_location + \
                f'settlement_data_{i:03d}.pkl'
            self.geographic_output_path = \
                lambda i: self.output_data_location + \
                f'geographic_data_{i:03d}.pkl'

            # set switches for output generation
            self.output_geographic_data = True
            self.output_settlement_data = True
        else:
            self.output_geographic_data = False
            self.output_settlement_data = False

        self.trajectory = []
        self.traders_trajectory = []

        # *******************************************************************
        # MODEL DATA SOURCES
        # *******************************************************************

        # documentation for TEMPERATURE and PRECIPITATION data can be found
        # here: http://www.worldclim.org/formats
        # apparently temperature data is given in x*10 format to allow for
        # smaller file sizes.
        # original version of mayasim divides temperature by 12 though
        self.temp = np.load(input_data_location +
                            '0_RES_432x400_temp.npy') / 12.

        # precipitation in mm or liters per square meter
        # (comparing the numbers to numbers from Wikipedia suggests
        # that it is given per year)
        self.precip = np.load(input_data_location + '0_RES_432x400_precip.npy')

        # in meters above sea level
        self.elev = np.load(input_data_location + '0_RES_432x400_elev.npy')
        self.slope = np.load(input_data_location + '0_RES_432x400_slope.npy')

        # documentation for SOIL PRODUCTIVITY is given at:
        # http://www.fao.org/geonetwork/srv/en/
        # main.home?uuid=f7a2b3c0-bdbf-11db-a0f6-000d939bc5d8
        # The soil production index considers the suitability
        # of the best adapted crop to each soils
        # condition in an area and makes a weighted average for
        #  all soils present in a pixel based
        # on the formula: 0.9 * VS + 0.6 * S + 0.3 * MS + 0 * NS.
        # Values range from 0 (bad) to 6 (good)
        self.soilprod = np.load(input_data_location + '0_RES_432x400_soil.npy')
        # it also sets soil productivity to 1.5 where the elevation is <= 1
        self.soilprod[(self.elev <= 1) & (np.logical_not(np.isnan(self.elev)))] = 1.5
        # smoothen soil productivity dataset
        self.soilprod = ndimage.gaussian_filter(self.soilprod,
                                                sigma=(2, 2),
                                                order=0)
        # and set to zero for non land cells
        self.soilprod[np.isnan(self.elev)] = 0

        # *******************************************************************
        # MODEL MAP INITIALIZATION
        # *******************************************************************

        # dimensions of the map
        self.rows, self.columns = self.precip.shape
        self.height, self.width = 914., 840.  # height and width in km
        self.pixel_dim = self.width / self.columns
        self.cell_width = self.width / self.columns
        self.cell_height = self.height / self.rows
        self.land_patches = np.asarray(np.where(np.isfinite(self.elev)))
        self.number_of_land_patches = self.land_patches.shape[1]

        # lengh unit - total map is about 500 km wide
        self.area = 516484. / len(self.land_patches[0])
        self.elev[:, 0] = np.inf
        self.elev[:, -1] = np.inf
        self.elev[0, :] = np.inf
        self.elev[-1, :] = np.inf
        # create a list of the index values i = (x, y) of the land
        # patches with finite elevation h
        self.list_of_land_patches = [
            i for i, h in np.ndenumerate(self.elev)

            if np.isfinite(self.elev[i])
        ]

        # initialize soil degradation and population
        # gradient (influencing the forest)

        # *******************************************************************
        # INITIALIZE ECOSYSTEM
        # *******************************************************************

        # Soil (influencing primary production and agricultural productivity)
        self.soil_deg = np.zeros((self.rows, self.columns))

        # Forest
        self.forest_state = np.ones((self.rows, self.columns), dtype=int)
        self.forest_state[np.isnan(self.elev)] = 0

        self.forest_memory = np.zeros((self.rows, self.columns), dtype=int)
        self.cleared_land_neighbours = np.zeros((self.rows, self.columns),
                                                dtype=int)
        # The forest has three states: 3=climax forest,
        # 2=secondary regrowth, 1=cleared land.

        for i in self.list_of_land_patches:
            self.forest_state[i] = 3

        # Variables describing total amount of water and water flow
        self.water = np.zeros((self.rows, self.columns))
        self.flow = np.zeros((self.rows, self.columns))
        self.spaciotemporal_precipitation = np.zeros((self.rows, self.columns))

        # initialize the trajectories of the water drops
        self.x = np.zeros((self.rows, self.columns), dtype="int")
        self.y = np.zeros((self.rows, self.columns), dtype="int")

        # define relative coordinates of the neighbourhood of a cell
        self.neighbourhood = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
        self.f90neighbourhood = np.asarray(self.neighbourhood).T

        # *******************************************************************
        # INITIALIZE SOCIETY
        # *******************************************************************

        # Population gradient (influencing the forest)
        self.pop_gradient = np.zeros((self.rows, self.columns))

        self.number_settlements = n
        # distribute specified number of settlements on the map
        self.settlement_positions = self.land_patches[:,
                                                      np.random.choice(
                                                          len(self.
                                                              land_patches[1]),
                                                          n).astype('int')]

        self.age = [0] * n

        # demographic variables
        self.birth_rate = [self.birth_rate_parameter] * n
        self.death_rate = [0.1 + 0.05 * r for r in list(np.random.random(n))]
        self.population = list(
            np.random.randint(self.min_init_inhabitants,
                              self.max_init_inhabitants, n).astype(float))
        self.mig_rate = [0.] * n
        self.out_mig = [0] * n
        self.migrants = [0] * n
        self.pioneer_set = []
        self.failed = 0

        # index list for populated and abandoned cities
        # used until removal of dead cities is implemented.
        self.populated_cities = range(n)
        self.dead_cities = []

        # agricultural influence
        self.number_cells_in_influence = [0] * n
        self.area_of_influence = [0.] * n
        self.coordinates = np.indices((self.rows, self.columns))
        self.cells_in_influence = [None] * n  # will be a list of arrays

        self.cropped_cells = [None] * n
        # for now, cropped cells are only the city positions.
        # first cropped cells are added at the first call of
        # get_cropped_cells()

        for city in self.populated_cities:
            self.cropped_cells[city] = [[self.settlement_positions[0, city]],
                                        [self.settlement_positions[1, city]]]
        # print(self.cropped_cells[1])
        self.occupied_cells = np.zeros((self.rows, self.columns))
        self.number_cropped_cells = [0] * n
        self.crop_yield = [0.] * n
        self.eco_benefit = [0.] * n
        self.available = 0

        # details of income from ecosystems services
        self.s_es_ag = [0.] * n
        self.s_es_wf = [0.] * n
        self.s_es_fs = [0.] * n
        self.s_es_sp = [0.] * n
        self.s_es_pg = [0.] * n
        self.es_ag = np.zeros((self.rows, self.columns), dtype=float)
        self.es_wf = np.zeros((self.rows, self.columns), dtype=float)
        self.es_fs = np.zeros((self.rows, self.columns), dtype=float)
        self.es_sp = np.zeros((self.rows, self.columns), dtype=float)
        self.es_pg = np.zeros((self.rows, self.columns), dtype=float)

        # Trade Variables
        self.adjacency = np.zeros((n, n))
        self.rank = [0] * n
        self.degree = [0] * n
        self.comp_size = [0] * n
        self.centrality = [0] * n
        self.trade_income = [0] * n
        self.max_cluster_size = 0

        # total real income per capita
        self.real_income_pc = [0] * n

    def _get_run_variables(self):
        """
        Saves all variables and values of the class instance 'self'
        in a dictionary file at the location given by 'path'

        Parameters:
        -----------
        self: class instance
            class instance whose variables are saved
        """

        dictionary = {
            attr: getattr(self, attr)

            for attr in dir(self)

            if not attr.startswith('__') and not callable(getattr(self, attr))
        }

        return dictionary

    def update_precipitation(self, t):
        """
        Modulates the initial precip dataset with a 24 timestep period.
        Returns a field of rainfall values for each cell.
        If veg_rainfall > 0, cleared_land_neighbours decreases rain.

        TO DO: The original Model increases specialization every time
        rainfall decreases, assuming that trade gets more important to
        compensate for agriculture decline
        """

        if self.precipitation_modulation:
            self.spaciotemporal_precipitation = \
                self.precip * (
                    1 + self.precipitation_amplitude *
                    self.precipitation_variation[
                        (np.ceil(t / self.climate_var) % 8).astype(int)]) \
                - self.veg_rainfall * self.cleared_land_neighbours
        else:
            self.spaciotemporal_precipitation = \
                self.precip * (1 -
                               self.veg_rainfall * self.cleared_land_neighbours)
        # check if system time is in drought period
        drought = False

        for drought_time in self.drought_times:
            if drought_time[0] < t <= drought_time[1]:
                drought = True
        # if so, decrease precipitation by factor percentage given by
        # drought severity

        if drought:
            self.spaciotemporal_precipitation *= \
                (1. - self.drought_severity / 100.)

    def get_waterflow(self):
        """
        waterflow: takes rain as an argument, uses elev, returns
        water flow distribution
        the precip percent parameter that reduces the amount of raindrops that
        have to be moved.
        Thereby inceases performance.


        f90waterflow takes as arguments:
        list of coordinates of land cells (2xN_land)
        elevation map in (height x width)
        rain_volume per cell map in (height x width)
        rain_volume and elevation must have same units: height per cell
        neighbourhood offsets
        height and width of map as integers,
        Number of land cells, N_land
        """

        # convert precipitation from mm to meters
        # NOTE: I think, this should be 1e-3
        # to convert from mm to meters though...
        # but 1e-5 is what they do in the original version.
        rain_volume = np.nan_to_num(self.spaciotemporal_precipitation * 1e-5)
        max_x, max_y = self.rows, self.columns
        err, self.flow, self.water = \
            f90routines.f90waterflow(self.land_patches,
                                     self.elev,
                                     rain_volume,
                                     self.f90neighbourhood,
                                     max_x,
                                     max_y,
                                     self.number_of_land_patches)

        return self.water, self.flow

    def forest_evolve(self, npp):

        npp_mean = np.nanmean(npp)
        # Iterate over all cells repeatedly and regenerate or degenerate

        for repeat in range(4):
            for i in self.list_of_land_patches:
                if not np.isnan(self.elev[i]):
                    # Forest regenerates faster [slower] (linearly),
                    # if net primary productivity on the patch
                    # is above [below] average.
                    threshold = npp_mean / npp[i]
                    # Degradation:
                    # Decrement with probability 0.003
                    # if there is a settlement around,
                    # degrade with higher probability
                    probdec = self.natprobdec * (2 * self.pop_gradient[i] + 1)

                    if np.random.random() <= probdec:
                        if self.forest_state[i] == 3:
                            self.forest_state[i] = 2
                            self.forest_memory[i] = self.state_change_s2
                        elif self.forest_state[i] == 2:
                            self.forest_state[i] = 1
                            self.forest_memory[i] = 0

                    # Regeneration:"
                    # recover if tree = 1 and memory > threshold 1

                    if (self.forest_state[i] == 1 and self.forest_memory[i] >
                            self.state_change_s2 * threshold):
                        self.forest_state[i] = 2
                        self.forest_memory[i] = self.state_change_s2
                    # recover if tree = 2 and memory > threshold 2
                    # and certain number of neighbours are
                    # climax forest as well

                    if (self.forest_state[i] == 2 and self.forest_memory[i] >
                            self.state_change_s3 * threshold):
                        state_3_neighbours = \
                            np.sum(self.forest_state[i[0] - 1:i[0] + 2,
                                                     i[1] - 1:i[1] + 2] == 3)

                        if state_3_neighbours > \
                                self.min_number_of_s3_neighbours:
                            self.forest_state[i] = 3

                    # finally, increase memory by one
                    self.forest_memory[i] += 1
        # calculate cleared land neighbours for output:

        if self.veg_rainfall > 0:
            for i in self.list_of_land_patches:
                self.cleared_land_neighbours[i] = \
                    np.sum(self.forest_state[i[0] - 1:i[0] + 2,
                                             i[1] - 1:i[1] + 2] == 1)
        assert not np.any(self.forest_state[~np.isnan(self.elev)] < 1), \
            'forest state is smaller than 1 somewhere'

        return

    def net_primary_prod(self):
        """
        net_primaty_prod is the minimum of a quantity
        derived from local temperature and rain
        Why is it rain and not 'surface water'
        according to the waterflow model?
        """
        # EQUATION ############################################################
        npp = 3000 \
            * np.minimum(1 - np.exp(-6.64e-4
                                    * self.spaciotemporal_precipitation),
                         1. / (1 + np.exp(1.315 - (0.119 * self.temp))))
        # EQUATION ############################################################

        return npp

    def get_ag(self, npp, wf):
        """
        agricultural productivit is calculated via a
        linear additive model from
        net primary productivity, soil productivity,
        slope, waterflow and soil degradation
        of each patch.
        """
        # EQUATION ############################################################

        return self.a_npp * npp + self.a_sp * self.soilprod \
            - self.a_s * self.slope - self.a_wf * wf - self.soil_deg
        # EQUATION ############################################################

    def get_ecoserv(self, ag, wf):
        """
        Ecosystem Services are calculated via a linear
        additive model from agricultural productivity (ag),
        waterflow through the cell (wf) and forest
        state on the cell (forest) in [1,3],
        The recent version of mayasim limits value of
        ecosystem services to 1 < ecoserv < 250, it also proposes
        to include population density (pop_gradient) and precipitation (rain)
        """
        # EQUATION ###########################################################

        if not self.better_ess:
            self.es_ag = self.e_ag * ag
            self.es_wf = self.e_wf * wf
            self.es_fs = self.e_f * (self.forest_state - 1.)
            self.es_sp = self.e_r * self.spaciotemporal_precipitation
            self.es_pg = self.e_deg * self.pop_gradient
        else:
            # change to use forest as proxy for income from agricultural
            # productivity. Multiply by 2 to get same per cell levels as
            # before
            self.es_ag = np.zeros(np.shape(ag))
            self.es_wf = self.e_wf * wf
            self.es_fs = 2. * self.e_ag * (self.forest_state - 1.) * ag
            self.es_sp = self.e_r * self.spaciotemporal_precipitation
            self.es_pg = self.e_deg * self.pop_gradient

        return (self.es_ag + self.es_wf + self.es_fs + self.es_sp - self.es_pg)

        # EQUATION ###########################################################

    ######################################################################
    # The Society
    ######################################################################

    def benefit_cost(self, ag_in):
        # Benefit cost assessment

        return (self.max_yield *
                (1 - self.origin_shift * np.exp(np.float128(-self.slope_yield * ag_in))))

    def get_cells_in_influence(self):
        """
        creates a list of cells for each city that are under its influence.
        these are the cells that are closer than population^0.8/60 (which is
        not explained any further... change denominator to 80 and max value to
        30 from eyeballing the results
        """
        # EQUATION ####################################################################
        self.area_of_influence = [(x**0.8) / 60. for x in self.population]
        self.area_of_influence = [
            value if value < 40. else 40. for value in self.area_of_influence
        ]
        # EQUATION ####################################################################

        for city in self.populated_cities:
            distance = np.sqrt((self.cell_width *
                                (self.settlement_positions[0][city] -
                                 self.coordinates[0]))**2 +
                               (self.cell_height *
                                (self.settlement_positions[1][city] -
                                 self.coordinates[1]))**2)
            stencil = distance <= self.area_of_influence[city]
            self.cells_in_influence[city] = self.coordinates[:, stencil]
        self.number_cells_in_influence = [
            len(x[0]) for x in self.cells_in_influence
        ]

        return

    def get_cropped_cells(self, bca):
        """
        Updates the cropped cells for each city with positive population.
        Calculates the utility for each cell (depending on distance from
        the respective city) If population per cropped cell is lower then
        min_people_per_cropped_cell, cells are abandoned.
        Cells with negative utility are also abandoned.
        If population per cropped cell is higher than
        max_people_per_cropped_cell, new cells are cropped.
        Newly cropped cells are chosen such that they have highest utility
        """
        abandoned = 0
        sown = 0

        # for each settlement: how many cells are currently cropped ?
        self.number_cropped_cells = np.array(
            [len(x[0]) for x in self.cropped_cells])

        # agricultural population density (people per cropped land)
        # determines the number of cells that can be cropped.
        ag_pop_density = [
            p / (self.number_cropped_cells[c] * self.area)

            if self.number_cropped_cells[c] > 0 else 0.

            for c, p in enumerate(self.population)
        ]

        # occupied_cells is a mask of all occupied cells calculated as the
        # unification of the cropped cells of all settlements.

        if len(self.cropped_cells) > 0:
            occup = np.concatenate(self.cropped_cells, axis=1).astype('int')

            if False:
                print('population of cities without agriculture:')
                print(
                    np.array(self.population)[self.number_cropped_cells == 0])
                print('pt. migration from cities without agriculture:')
                print(np.array(self.out_mig)[self.number_cropped_cells == 0])
                print('out migration from cities without agriculture:')
                print(np.array(self.migrants)[self.number_cropped_cells == 0])

            for index in range(len(occup[0])):
                self.occupied_cells[occup[0, index], occup[1, index]] = 1
        # the age of settlements is increased here.
        self.age = [x + 1 for x in self.age]
        # for each settlement: which cells to crop ?
        # calculate utility first! This can be accelerated, if calculations
        # are only done in 40 km radius.

        for city in self.populated_cities:

            cells = list(
                zip(self.cells_in_influence[city][0],
                    self.cells_in_influence[city][1]))
            # EQUATION ########################################################
            utility = [
                bca[x, y] - self.estab_cost - (self.ag_travel_cost * np.sqrt(
                    (self.cell_width * (self.settlement_positions[0][city] -
                                        self.coordinates[0][x, y]))**2 +
                    (self.cell_height * (self.settlement_positions[1][city] -
                                         self.coordinates[1][x, y]))**2)) /
                np.sqrt(self.population[city]) for (x, y) in cells
            ]
            # EQUATION ########################################################
            available = [
                True if self.occupied_cells[x, y] == 0 else False

                for (x, y) in cells
            ]

            # jointly sort utilities, availability and cells such that cells
            # with highest utility are first.
            sorted_utility, sorted_available, sorted_cells = \
                list(zip(*sorted(list(zip(utility, available, cells)),
                                 reverse=True)))
            # of these sorted lists, sort filter only available cells
            available_util = list(
                compress(list(sorted_utility), list(sorted_available)))
            available_cells = list(
                compress(list(sorted_cells), list(sorted_available)))

            # save local copy of all cropped cells
            cropped_cells = list(zip(*self.cropped_cells[city]))
            # select utilities for these cropped cells
            cropped_utils = [
                utility[cells.index(cell)] if cell in cells else -1

                for cell in cropped_cells
            ]
            # sort utilitites and cropped cells to lowest utilities first
            city_has_crops = True if len(cropped_cells) > 0 else False

            if city_has_crops:
                occupied_util, occupied_cells = \
                    zip(*sorted(list(zip(cropped_utils, cropped_cells))))

            # 1.) include new cells if population exceeds a threshold

            # calculate number of new cells to crop
            number_of_new_cells = np.floor(ag_pop_density[city]
                                           / self.max_people_per_cropped_cell) \
                .astype('int')
            # and crop them by selecting cells with positive utility from the
            # beginning of the list

            for n in range(min([number_of_new_cells, len(available_util)])):
                if available_util[n] > 0:
                    self.occupied_cells[available_cells[n]] = 1

                    for dim in range(2):
                        self.cropped_cells[city][dim] \
                            .append(available_cells[n][dim])

            if city_has_crops:

                # 2.) abandon cells if population too low
                # after cities age > 5 years

                if (ag_pop_density[city] < self.min_people_per_cropped_cell
                        and self.age[city] > 5):

                    # There are some inconsistencies here. Cells are abandoned,
                    # if the 'people per cropped land' is lower then a
                    # threshold for 'people per cropped cells. Then the
                    # number of cells to abandon is calculated as 30/people
                    # per cropped land. Why?! (check the original version!)

                    number_of_lost_cells = np.ceil(
                        30 / ag_pop_density[city]).astype('int')

                    # TO DO: recycle utility and cell list to do this faster.
                    # therefore, filter cropped cells from utility list
                    # and delete last n cells.

                    for n in range(
                            min([number_of_lost_cells,
                                 len(occupied_cells)])):
                        dropped_cell = occupied_cells[n]
                        self.occupied_cells[dropped_cell] = 0

                        for dim in range(2):
                            self.cropped_cells[city][dim] \
                                .remove(dropped_cell[dim])
                        abandoned += 1

                # 3.) abandon cells with utility <= 0

                # find cells that have negative utility and belong
                # to city under consideration,
                useless_cropped_cells = [
                    occupied_cells[i] for i in range(len(occupied_cells))

                    if occupied_util[i] < 0
                    and occupied_cells[i] in zip(*self.cropped_cells[city])
                ]
                # and release them.

                for useless_cropped_cell in useless_cropped_cells:
                    self.occupied_cells[useless_cropped_cell] = 0

                    for dim in range(2):
                        try:
                            self.cropped_cells[city][dim] \
                                .remove(useless_cropped_cell[dim])
                        except ValueError:
                            print('ERROR: Useless cell gone already')
                    abandoned += 1

        # Finally, update list of lists containing cropped cells for each city
        # with positive population.
        self.number_cropped_cells = [
            len(self.cropped_cells[city][0])

            for city in range(len(self.population))
        ]

        return abandoned, sown

    def get_pop_mig(self):
        # gives population and out-migration
        # print("number of settlements", len(self.population))

        # death rate correlates inversely with real income per capita
        death_rate_diff = self.max_death_rate - self.min_death_rate

        self.death_rate = [
            -death_rate_diff * self.real_income_pc[i] + self.max_death_rate

            for i in range(len(self.real_income_pc))
        ]
        self.death_rate = list(
            np.clip(self.death_rate, self.min_death_rate, self.max_death_rate))

        # if population control,
        # birth rate negatively correlates with population size

        if self.population_control:
            birth_rate_diff = self.max_birth_rate - self.min_birth_rate

            self.birth_rate = [
                -birth_rate_diff / 10000. * value +
                self.shift if value > 5000 else self.birth_rate_parameter

                for value in self.population
            ]
        # population grows according to effective growth rate
        self.population = [
            int((1. + self.birth_rate[i] - self.death_rate[i]) * value)

            for i, value in enumerate(self.population)
        ]
        self.population = [
            value if value > 0 else 0 for value in self.population
        ]

        mig_rate_diffe = self.max_mig_rate - self.min_mig_rate

        # outmigration rate also correlates
        # inversely with real income per capita
        self.mig_rate = [
            -mig_rate_diffe * self.real_income_pc[i] + self.max_mig_rate

            for i in range(len(self.real_income_pc))
        ]
        self.mig_rate = list(
            np.clip(self.mig_rate, self.min_mig_rate, self.max_mig_rate))
        self.out_mig = [
            int(self.mig_rate[i] * self.population[i])

            for i in range(len(self.population))
        ]
        self.out_mig = [value if value > 0 else 0 for value in self.out_mig]

        return

    # impact of sociosphere on ecosphere
    def update_pop_gradient(self):
        # pop gradient quantifies the disturbance of the forest by population
        self.pop_gradient = np.zeros((self.rows, self.columns))

        for city in self.populated_cities:
            distance = np.sqrt(self.area * (
                (self.settlement_positions[0][city] - self.coordinates[0])**2 +
                (self.settlement_positions[1][city] - self.coordinates[1])**2))

            # EQUATION ###################################################################
            self.pop_gradient[self.cells_in_influence[city][0],
                              self.cells_in_influence[city][1]] += \
                self.population[city] \
                / (300 * (1 + distance[self.cells_in_influence[city][0],
                                       self.cells_in_influence[city][1]]))
            # EQUATION ###################################################################
            self.pop_gradient[self.pop_gradient > 15] = 15

    def evolve_soil_deg(self):
        # soil degrades for cropped cells

        cropped = np.concatenate(self.cropped_cells, axis=1).astype('int')
        self.soil_deg[cropped[0], cropped[1]] += self.deg_rate
        self.soil_deg[self.forest_state == 3] -= self.reg_rate
        self.soil_deg[self.soil_deg < 0] = 0

    def get_rank(self):
        # depending on population ranks are assigned
        # attention: ranks are reverted with respect to Netlogo MayaSim !
        # 1 => 3 ; 2 => 2 ; 3 => 1
        self.rank = [
            3

            if value > self.thresh_rank_3 else 2 if value > self.thresh_rank_2
            else 1 if value > self.thresh_rank_1 else 0

            for index, value in enumerate(self.population)
        ]

        return

    @property
    def build_routes(self):

        adj = self.adjacency.copy()
        adj[adj == -1] = 0
        built_links = 0
        lost_links = 0
        g = nx.from_numpy_array(adj, create_using=nx.DiGraph())
        self.degree = g.out_degree()
        # cities with rank>0 are traders and establish links to neighbours

        for city in self.populated_cities:
            if self.degree[city] < self.rank[city]:
                distances = \
                    (np.sqrt(self.area * (+ (self.settlement_positions[0][city]
                                             - self.settlement_positions[0]) ** 2
                                          + (self.settlement_positions[1][city]
                                             - self.settlement_positions[1]) ** 2
                                          )))

                if self.rank[city] == 3:
                    threshold = 31. * (
                        self.thresh_rank_3 / self.thresh_rank_3 * 0.5 + 1.)
                elif self.rank[city] == 2:
                    threshold = 31. * (
                        self.thresh_rank_2 / self.thresh_rank_3 * 0.5 + 1.)
                elif self.rank[city] == 1:
                    threshold = 31. * (
                        self.thresh_rank_1 / self.thresh_rank_3 * 0.5 + 1.)
                else:
                    threshold = 0

                # don't chose yourself as nearest neighbor
                distances[city] = 2 * threshold

                # collect close enough neighbors and omit those that are
                # already connected.
                a = distances <= threshold
                b = self.adjacency[city] == 0
                nearby = np.array(list(map(operator.and_, a, b)))

                # if there are traders nearby,
                # connect to the one with highest population

                if sum(nearby) != 0:
                    try:
                        new_partner = np.nanargmax(self.population * nearby)
                        self.adjacency[city, new_partner] = 1
                        self.adjacency[new_partner, city] = -1
                        built_links += 1
                    except ValueError:
                        print('ERROR in new partner')
                        print(np.shape(self.population),
                              np.shape(self.settlement_positions[0]))
                        sys.exit(-1)

            # cities who cannot maintain their trade links lose them:
            elif self.degree[city] > self.rank[city]:
                # get neighbors of node
                neighbors = g.successors(city)
                # find smallest of neighbors
                smallest_neighbor = self.population.index(
                    min([self.population[nb] for nb in neighbors]))
                # cut corresponding link
                self.adjacency[city, smallest_neighbor] = 0
                self.adjacency[smallest_neighbor, city] = 0
                lost_links += 1

        return (built_links, lost_links)

    def get_comps(self):
        # convert adjacency matrix to compressed sparse row format
        adjacency_csr = sparse.csr_matrix(np.absolute(self.adjacency))

        # extract data vector, row index vector and index pointer vector
        a = adjacency_csr.data
        # add one to make indexing compatible to fortran
        # (where indices start counting with 1)
        j_a = adjacency_csr.indices + 1
        i_c = adjacency_csr.indptr + 1

        # determine length of data vectors
        l_a = np.shape(a)[0]
        l_ic = np.shape(i_c)[0]

        # if data vector is not empty, pass data to fortran routine.
        # else, just fill the centrality vector with ones.

        if l_a > 0:
            tmp_comp_size, tmp_degree = \
                f90routines.f90sparsecomponents(i_c, a, j_a,
                                                self.number_settlements,
                                                l_ic, l_a)
            self.comp_size, self.degree = list(tmp_comp_size), list(tmp_degree)
        elif l_a == 0:
            self.comp_size, self.degree = [0] * (l_ic - 1), [0] * (l_ic - 1)

        return

    def get_centrality(self):
        # convert adjacency matrix to compressed sparse row format
        adjacency_csr = sparse.csr_matrix(np.absolute(self.adjacency))

        # extract data vector, row index vector and index pointer vector
        a = adjacency_csr.data
        # add one to make indexing compatible to fortran
        # (where indices start counting with 1)
        j_a = adjacency_csr.indices + 1
        i_c = adjacency_csr.indptr + 1

        # determine length of data vectors
        l_a = np.shape(a)[0]
        l_ic = np.shape(i_c)[0]
        # print('number of trade links:', sum(a) / 2)

        # if data vector is not empty, pass data to fortran routine.
        # else, just fill the centrality vector with ones.

        if l_a > 0:
            tmp_centrality = f90routines \
                .f90sparsecentrality(i_c, a, j_a,
                                     self.number_settlements,
                                     l_ic, l_a)
            self.centrality = list(tmp_centrality)
        elif l_a == 0:
            self.centrality = [1] * (l_ic - 1)

        return

    def get_crop_income(self, bca):
        # agricultural benefit of cropping

        for city in self.populated_cities:
            crops = bca[self.cropped_cells[city][0], self.
                        cropped_cells[city][1]]
            # EQUATION #

            if self.crop_income_mode == "mean":
                self.crop_yield[city] = self.r_bca_mean \
                    * np.nanmean(crops[crops > 0])
            elif self.crop_income_mode == "sum":
                self.crop_yield[city] = self.r_bca_sum \
                    * np.nansum(crops[crops > 0])
        self.crop_yield = [
            0 if np.isnan(self.crop_yield[index]) else self.crop_yield[index]

            for index in range(len(self.crop_yield))
        ]

        return

    def get_eco_income(self, es):
        # benefit from ecosystem services of cells in influence
        # ##EQUATION###################################################################

        for city in self.populated_cities:
            if self.eco_income_mode == "mean":
                self.eco_benefit[city] = self.r_es_mean \
                    * np.nanmean(es[self.cells_in_influence[city]])
            elif self.eco_income_mode == "sum":
                self.eco_benefit[city] = self.r_es_sum \
                    * np.nansum(es[self.cells_in_influence[city]])
                self.s_es_ag[city] = self.r_es_sum \
                    * np.nansum(self.es_ag[self.cells_in_influence[city]])
                self.s_es_wf[city] = self.r_es_sum \
                    * np.nansum(self.es_wf[self.cells_in_influence[city]])
                self.s_es_fs[city] = self.r_es_sum \
                    * np.nansum(self.es_fs[self.cells_in_influence[city]])
                self.s_es_sp[city] = self.r_es_sum \
                    * np.nansum(self.es_sp[self.cells_in_influence[city]])
                self.s_es_pg[city] = self.r_es_sum \
                    * np.nansum(self.es_pg[self.cells_in_influence[city]])
        try:
            self.eco_benefit[self.population == 0] = 0
        except IndexError:
            self.print_variable_lengths()
        # ##EQUATION###################################################################

        return

    def get_trade_income(self):
        # ##EQUATION###################################################################
        self.trade_income = [
            1. / 30. * (1 + self.comp_size[i] / self.centrality[i])**0.9

            for i in range(len(self.centrality))
        ]
        self.trade_income = [
            self.r_trade if value > 1 else 0 if
            (value < 0 or self.degree[index] == 0) else self.r_trade * value

            for index, value in enumerate(self.trade_income)
        ]
        # ##EQUATION###################################################################

        return

    def get_real_income_pc(self):
        # combine agricultural, ecosystem service and trade benefit
        # EQUATION #
        self.real_income_pc = [
            (self.crop_yield[index] + self.eco_benefit[index] +
             self.trade_income[index]) /
            self.population[index] if value > 0 else 0

            for index, value in enumerate(self.population)
        ]

        return

    def migration(self, es):
        # if outmigration rate exceeds threshold, found new settlement
        self.migrants = [0] * self.number_settlements
        new_settlements = 0
        vacant_lands = np.isfinite(es)
        influenced_cells = np.concatenate(self.cells_in_influence, axis=1)
        vacant_lands[influenced_cells[0], influenced_cells[1]] = 0
        vacant_lands = np.asarray(np.where(vacant_lands == 1))
        for city in self.populated_cities:
            rd = np.random.rand()

            if (self.out_mig[city] > 400 and len(vacant_lands[0]) > 0
                    and np.random.rand() <= 0.5):

                mig_pop = self.out_mig[city]
                self.migrants[city] = mig_pop
                self.population[city] -= mig_pop
                self.pioneer_set = \
                    vacant_lands[:, np.random.choice(len(vacant_lands[0]), 75)]

                travel_cost = np.sqrt(
                    self.area *
                    ((self.settlement_positions[0][city] - self.coordinates[0])
                     **2 + (self.settlement_positions[1][city] -
                            self.coordinates[1])**2))

                utility = self.mig_ES_pref * es \
                    + self.mig_TC_pref * travel_cost
                utofpio = utility[self.pioneer_set[0], self.pioneer_set[1]]
                new_loc = self.pioneer_set[:, np.nanargmax(utofpio)]

                neighbours = \
                    (np.sqrt(self.area * ((new_loc[0]
                                           - self.settlement_positions[0]) ** 2 +
                                          (new_loc[1]
                                           - self.settlement_positions[1]) ** 2
                                          ))) <= 7.5
                summe = np.sum(neighbours)

                if summe == 0:
                    self.spawn_city(new_loc[0], new_loc[1], mig_pop)
                    index = (vacant_lands[0, :] == new_loc[0]) \
                        & (vacant_lands[1, :] == new_loc[1])
                    np.delete(vacant_lands, int(np.where(index)[0]), 1)
                    new_settlements += 1

        return new_settlements

    def kill_cities(self):

        # BUG: cities can be added twice,
        # if they have neither population nor cropped cells.
        # this might lead to unexpected consequences. see what happenes,
        # when after adding all cities, only unique ones are kept
        # (not a problem when self.kill_cities_without_crops=False)

        killed_cities = 0

        # kill cities if they have either no crops or no inhabitants:
        dead_city_indices = [
            i for i in range(len(self.population))

            if self.population[i] <= self.min_city_size
        ]

        if self.kill_cities_without_crops:
            dead_city_indices += [
                i for i in range(len(self.population))

                if (len(self.cropped_cells[i][0]) <= 0)
            ]

        # the following expression only keeps the unique entries.
        # might solve the problem.
        dead_city_indices = list(set(dead_city_indices))

        # remove entries from variables
        # simple lists that can be deleted elementwise

        for index in sorted(dead_city_indices, reverse=True):
            self.number_settlements -= 1
            self.failed += 1
            del self.age[index]
            del self.birth_rate[index]
            del self.death_rate[index]
            del self.population[index]
            del self.mig_rate[index]
            del self.out_mig[index]
            del self.number_cells_in_influence[index]
            del self.area_of_influence[index]
            del self.number_cropped_cells[index]
            del self.crop_yield[index]
            del self.eco_benefit[index]
            del self.rank[index]
            del self.degree[index]
            del self.comp_size[index]
            del self.centrality[index]
            del self.trade_income[index]
            del self.real_income_pc[index]
            del self.cells_in_influence[index]
            del self.cropped_cells[index]
            del self.s_es_ag[index]
            del self.s_es_wf[index]
            del self.s_es_fs[index]
            del self.s_es_sp[index]
            del self.s_es_pg[index]
            del self.migrants[index]

            killed_cities += 1

        # special cases:
        self.settlement_positions = \
            np.delete(self.settlement_positions,
                      dead_city_indices, axis=1)
        self.adjacency = \
            np.delete(np.delete(self.adjacency,
                                dead_city_indices, axis=0),
                      dead_city_indices, axis=1)

        # update list of indices for populated and dead cities

        # a) update list of populated cities
        self.populated_cities = [
            index for index, value in enumerate(self.population) if value > 0
        ]

        # b) update list of dead cities
        self.dead_cities = [
            index for index, value in enumerate(self.population) if value == 0
        ]

        return killed_cities

    def spawn_city(self, x, y, mig_pop):
        """
        Spawn a new city at given location with
        given population and append it to all necessary lists.

        Parameters
        ----------
        x: int
            x location of new city on map
        y: int
            y location of new city on map
        mig_pop: int
            initial population of new city
        """

        # extend all variables to include new city
        self.number_settlements += 1
        self.settlement_positions = np.append(self.settlement_positions,
                                              [[x], [y]], 1)
        self.cells_in_influence.append([[x], [y]])
        self.cropped_cells.append([[x], [y]])

        n = len(self.adjacency)
        self.adjacency = np.append(self.adjacency, [[0] * n], 0)
        self.adjacency = np.append(self.adjacency, [[0]] * (n + 1), 1)

        self.age.append(0)
        self.birth_rate.append(self.birth_rate_parameter)
        self.death_rate.append(0.1 + 0.05 * np.random.rand())
        self.population.append(mig_pop)
        self.mig_rate.append(0)
        self.out_mig.append(0)
        self.number_cells_in_influence.append(0)
        self.area_of_influence.append(0)
        self.number_cropped_cells.append(1)
        self.crop_yield.append(0)
        self.eco_benefit.append(0)
        self.rank.append(0)
        self.degree.append(0)
        self.trade_income.append(0)
        self.real_income_pc.append(0)
        self.s_es_ag.append(0)
        self.s_es_wf.append(0)
        self.s_es_fs.append(0)
        self.s_es_sp.append(0)
        self.s_es_pg.append(0)
        self.migrants.append(0)

    def run(self, t_max=1):
        """
        Run the model for a given number of steps.
        If no number of steps is given, the model is integrated for one step

        Parameters
        ----------
        t_max: int
            number of steps to integrate the model
        """

        # print update about output state
        #if self.debug:
        #    print('output of settlement and geodata is {} and {}'.format(
        #        self.output_settlement_data, self.output_geographic_data))

        # initialize variables
        # net primary productivity
        npp = np.zeros((self.rows, self.columns))
        # water flow
        wf = np.zeros((self.rows, self.columns))
        # agricultural productivity
        ag = np.zeros((self.rows, self.columns))
        # ecosystem services
        es = np.zeros((self.rows, self.columns))
        # benefit cost map for agriculture
        bca = np.zeros((self.rows, self.columns))

        self.init_output()
        
        for t in tqdm(range(1,t_max+1)):

            #if self.debug:
            #    print(f"time = {t}, population = {sum(self.population)}")

            # evolve subselfs
            # ecosystem
            self.update_precipitation(t)
            npp = self.net_primary_prod()
            self.forest_evolve(npp)
            # this is curious: only waterflow is used,
            # water level is abandoned.
            wf = self.get_waterflow()[1]
            ag = self.get_ag(npp, wf)
            es = self.get_ecoserv(ag, wf)
            bca = self.benefit_cost(ag)

            # society

            if len(self.population) > 0:
                self.get_cells_in_influence()
                abandoned, sown = self.get_cropped_cells(bca)
                self.get_crop_income(bca)
                self.get_eco_income(es)
                self.evolve_soil_deg()
                self.update_pop_gradient()
                self.get_rank()
                (built, lost) = self.build_routes
                self.get_comps()
                self.get_centrality()
                self.get_trade_income()
                self.get_real_income_pc()
                self.get_pop_mig()
                new_settlements = self.migration(es)
                killed_settlements = self.kill_cities()
            else:
                abandoned = sown = cl = 0

            self.step_output(t, npp, wf, ag, es, bca, abandoned, sown, built,
                             lost, new_settlements, killed_settlements)


    def init_output(self):
        """initializes data output for trajectory, settlements and geography depending on settings"""

        if self.output_trajectory:
            self.init_trajectory_output()
            self.init_traders_trajectory_output()

        if self.output_geographic_data or self.output_settlement_data:

            # If output data location is needed and does not exist, create it.

            if not os.path.exists(self.output_data_location):
                os.makedirs(self.output_data_location)

            if not self.output_data_location.endswith('/'):
                self.output_data_location += '/'

        if self.output_settlement_data:
            settlement_init_data = {'shape': (self.rows, self.columns)}
            with open(self.settlement_output_path(0), 'wb') as f:
                pkl.dump(settlement_init_data, f)

        if self.output_geographic_data:
            pass

    def step_output(self, t, npp, wf, ag, es, bca, abandoned, sown, built,
                    lost, new_settlements, killed_settlements):
        """
        call different data saving routines depending on settings.

        Parameters
        ----------
        t: int
            Timestep number to append to save file path
        npp: numpy array
            Net Primary Productivity on cell basis
        wf: numpy array
            Water flow through cell
        ag: numpy array
            Agricultural productivity of cell
        es: numpy array
            Ecosystem services of cell (that are summed and weighted to
            calculate ecosystems service income)
        bca: numpy array
            Benefit cost analysis of agriculture on cell.
        abandoned: int
            Number of cells that was abandoned in the previous time step
        sown: int
            Number of cells that was newly cropped in the previous time step
        built : int
            number of trade links built in this timestep
        lost : int
            number of trade links lost in this timestep
        new_settlements : int
            number of new settlements that were spawned during the preceeding
            timestep
        killed_settlements : int
            number of settlements that were killed during the preceeding
            timestep
        """

        # append stuff to trajectory

        if self.output_trajectory:
            self.update_trajectory_output(t, [npp, wf, ag, es, bca], built,
                                          lost, new_settlements,
                                          killed_settlements)
            self.update_traders_trajectory_output(t)

        # save maps of spatial data

        if self.output_geographic_data:
            self.save_geographic_output(t, npp, wf, ag, es, bca, abandoned,
                                        sown)

        # save data on settlement basis

        if self.output_settlement_data:
            self.save_settlement_output(t)

    def save_settlement_output(self, t):
        """
        Organize settlement based data in Pandas Dataframe
        and save to file.

        Parameters
        ----------
        t: int
            Timestep number to append to save file path

        """
        colums = [
            'population', 'real income', 'ag income', 'es income',
            'trade income', 'x position', 'y position', 'out migration',
            'degree'
        ]
        data = [
            self.population, self.real_income_pc, self.crop_yield,
            self.eco_benefit, self.trade_income,
            list(self.settlement_positions[0]),
            list(self.settlement_positions[1]), self.migrants,
            [self.degree[city] for city in self.populated_cities]
        ]

        data = list(map(list, zip(*data)))

        data_frame = pandas.DataFrame(columns=colums, data=data)

        with open(self.settlement_output_path(t), 'wb') as f:
            pkl.dump(data_frame, f)

    def save_geographic_output(self, t, npp, wf, ag, es, bca, abandoned, sown):
        """
        Organize Geographic data in dictionary (for separate layers
        of data) and save to file.

        Parameters
        ----------
        t: int
            Timestep number to append to save file path
        npp: numpy array
            Net Primary Productivity on cell basis
        wf: numpy array
            Water flow through cell
        ag: numpy array
            Agricultural productivity of cell
        es: numpy array
            Ecosystem services of cell (that are summed and weighted to
            calculate ecosystems service income)
        bca: numpy array
            Benefit cost analysis of agriculture on cell.
        abandoned: int
            Number of cells that was abandoned in the previous time step
        sown: int
            Number of cells that was newly cropped in the previous time step
        """

        tmpforest = self.forest_state.copy()
        tmpforest[np.isnan(self.elev)] = 0
        data = {
            'forest': tmpforest,
            'waterflow': wf,
            'cells in influence': self.cells_in_influence,
            'number of cells in influence': self.number_cells_in_influence,
            'cropped cells': self.cropped_cells,
            'number of cropped cells': self.number_cropped_cells,
            'abandoned sown': np.array([abandoned, sown]),
            'soil degradation': self.soil_deg,
            'population gradient': self.pop_gradient,
            'adjacency': self.adjacency,
            'x positions': list(self.settlement_positions[0]),
            'y positions': list(self.settlement_positions[1]),
            'population': self.population,
            'elev': self.elev,
            'rank': self.rank
        }

        with open(self.geographic_output_path(t), 'wb') as f:
            pkl.dump(data, f)

    def init_trajectory_output(self):
        self.trajectory.append([
            'time', 'total_population', 'max_settlement_population',
            'total_migrants', 'total_settlements', 'total_agriculture_cells',
            'total_cells_in_influence', 'total_trade_links',
            'mean_cluster_size', 'max_cluster_size', 'new_settlements',
            'killed_settlements', 'built_trade_links', 'lost_trade_links',
            'total_income_agriculture', 'total_income_ecosystem',
            'total_income_trade', 'mean_soil_degradation',
            'forest_state_3_cells', 'forest_state_2_cells',
            'forest_state_1_cells', 'es_income_forest', 'es_income_waterflow',
            'es_income_agricultural_productivity', 'es_income_precipitation',
            'es_income_pop_density', 'MAP', 'max_npp', 'mean_waterflow',
            'max_AG', 'max_ES', 'max_bca', 'max_soil_deg', 'max_pop_grad'
        ])

    def init_traders_trajectory_output(self):
        self.traders_trajectory.append([
            'time', 'total_population', 'total_migrants', 'total_traders',
            'total_settlements', 'total_agriculture_cells',
            'total_cells_in_influence', 'total_trade_links',
            'total_income_agriculture', 'total_income_ecosystem',
            'total_income_trade', 'es_income_forest', 'es_income_waterflow',
            'es_income_agricultural_productivity', 'es_income_precipitation',
            'es_income_pop_density'
        ])

    def update_trajectory_output(self, time, args, built, lost,
                                 new_settlements, killed_settlements):
        # args = [npp, wf, ag, es, bca]

        total_population = sum(self.population)
        try:
            max_population = np.nanmax(self.population)
        except:
            max_population = float('nan')
        total_migrangs = sum(self.migrants)
        total_settlements = len(self.population)
        total_trade_links = sum(self.degree) / 2
        income_agriculture = sum(self.crop_yield)
        income_ecosystem = sum(self.eco_benefit)
        income_trade = sum(self.trade_income)
        number_of_components = float(
            sum([1 if value > 0 else 0 for value in self.comp_size]))
        mean_cluster_size = float(sum(self.comp_size)) / number_of_components \
            if number_of_components > 0 else 0
        try:
            max_cluster_size = max(self.comp_size)
        except:
            max_cluster_size = 0
        self.max_cluster_size = max_cluster_size
        total_agriculture_cells = sum(self.number_cropped_cells)
        total_cells_in_influence = sum(self.number_cells_in_influence)

        self.trajectory.append([
            time, total_population, max_population, total_migrangs,
            total_settlements, total_agriculture_cells,
            total_cells_in_influence, total_trade_links, mean_cluster_size,
            max_cluster_size, new_settlements, killed_settlements, built, lost,
            income_agriculture, income_ecosystem, income_trade,
            np.nanmean(self.soil_deg),
            np.sum(self.forest_state == 3),
            np.sum(self.forest_state == 2),
            np.sum(self.forest_state == 1),
            np.sum(self.s_es_fs),
            np.sum(self.s_es_wf),
            np.sum(self.s_es_ag),
            np.sum(self.s_es_sp),
            np.sum(self.s_es_pg),
            np.nanmean(self.spaciotemporal_precipitation),
            np.nanmax(args[0]),
            np.nanmean(args[1]),
            np.nanmax(args[2]),
            np.nanmax(args[3]),
            np.nanmax(args[4]),
            np.nanmax(self.soil_deg),
            np.nanmax(self.pop_gradient)
        ])

    def update_traders_trajectory_output(self, time):

        traders = np.where(np.array(self.degree) > 0)[0]

        total_population = sum([self.population[c] for c in traders])
        total_migrants = sum([self.migrants[c] for c in traders])
        total_settlements = len(self.population)
        total_traders = len(traders)
        total_trade_links = sum(self.degree) / 2
        income_agriculture = sum([self.crop_yield[c] for c in traders])
        income_ecosystem = sum([self.eco_benefit[c] for c in traders])
        income_trade = sum([self.trade_income[c] for c in traders])
        income_es_fs = sum([self.s_es_fs[c] for c in traders])
        income_es_wf = sum([self.s_es_wf[c] for c in traders])
        income_es_ag = sum([self.s_es_ag[c] for c in traders])
        income_es_sp = sum([self.s_es_sp[c] for c in traders])
        income_es_pg = sum([self.s_es_pg[c] for c in traders])
        number_of_components = float(
            sum([1 if value > 0 else 0 for value in self.comp_size]))
        mean_cluster_size = (float(sum(self.comp_size)) / number_of_components
                             if number_of_components > 0 else 0)
        try:
            max_cluster_size = max(self.comp_size)
        except:
            max_cluster_size = 0
        total_agriculture_cells = \
            sum([self.number_cropped_cells[c] for c in traders])
        total_cells_in_influence = \
            sum([self.number_cells_in_influence[c] for c in traders])

        self.traders_trajectory.append([
            time, total_population, total_migrants, total_traders,
            total_settlements, total_agriculture_cells,
            total_cells_in_influence, total_trade_links, income_agriculture,
            income_ecosystem, income_trade, income_es_fs, income_es_wf,
            income_es_ag, income_es_sp, income_es_pg
        ])

    def get_trajectory(self):
        try:
            trj = np.array(self.trajectory)
            columns = trj[0, :]
            df = pandas.DataFrame(trj[1:, :], columns=columns, dtype='float')
        except IOError:
            print('trajectory mode must be turned on')

        return df

    def get_traders_trajectory(self):
        try:
            trj = self.traders_trajectory
            columns = trj.pop(0)
            df = pandas.DataFrame(trj, columns=columns)
        except IOError:
            print('trajectory mode must be turned on')

        return df

    def run_test(self, timesteps=5):
        import shutil

        N = 50

        # define saving location
        comment = "testing_version"
        now = datetime.datetime.now()
        location = "output_data/" \
                   + "Output_" + comment + '/'

        if os.path.exists(location):
            shutil.rmtree(location)
        os.makedirs(location)

        # initialize Model
        model = ModelCore(n=N,
                          debug=True,
                          output_trajectory=True,
                          output_settlement_data=True,
                          output_geographic_data=True,
                          output_data_location=location)
        # run Model

        model.crop_income_mode = 'sum'
        model.r_es_sum = 0.0001
        model.r_bca_sum = 0.1
        model.population_control = 'False'
        model.run(timesteps)

        trj = model.get_trajectory()
        plot = trj[[
            'total_population', 'total_settlements', 'total_migrants'
            ]].plot()


        return 1

    def print_variable_lengths(self):
        for var in dir(self):
            if not var.startswith('__') and not callable(getattr(self, var)):
                try:
                    if len(getattr(self, var)) != 432:
                        print(var, len(getattr(self, var)))
                except:
                    pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import shutil

    N = 10

    # define saving location
    comment = "testing_version"
    now = datetime.datetime.now()
    location = "output_data/" \
               + "Output_" + comment + '/'

    if os.path.exists(location):
        shutil.rmtree(location)
    # os.makedirs(location)

    # initialize Model
    model = ModelCore(n=N,
                      debug=True,
                      output_trajectory=True,
                      output_settlement_data=True,
                      output_geographic_data=True,
                      output_data_location=location)
    # run Model
    timesteps = 30
    model.crop_income_mode = 'sum'
    model.r_es_sum = 0.0001
    model.r_bca_sum = 0.25
    model.population_control = 'False'
    model.run(timesteps)

    # plot trajectory
    trj = model.get_trajectory()
    ax = trj[[
        'total_population', 'total_settlements', 'total_migrants'
        ]].plot()
    plt.show()
    ax.figure.savefig(location + 'plot.png')
