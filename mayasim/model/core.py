import operator
import os
import sys
from itertools import compress, chain

import pickle as pkl
import networkx as nx
import numpy as np
import pandas
import pkg_resources
from scipy import ndimage, sparse
from tqdm.auto import trange

from .parameters import Parameters
from ._ext.f90routines import f90routines

class Core(Parameters):
    """
    This class represents the core of the MayaSim model
    and contains all methods that are needed to run it.
    Parameter-attributes are inherited from the Parameters class.

    MayaSim consists of settlement-agents representing a human society
    which is acting on a cell-grid representing their natural environment.

    Cell attribute names start with 'cel_'.
    Settlement attribute names start with 'stm_'.

    **Example:**

    Initialize and run an instance of the model like so:

    >>> model = mayasim.model.core.Core()
    ... model.run(t_max=350)
    """
    # pylint: disable=too-many-statements
    def __init__(self,
                 n=30,
                 calc_aggregates=True,
                 output_path=None):
        """
        Returns an instance of the MayaSim model.

        Parameters
        ----------
        n: int
            number of settlements to initialize,
        calc_aggregates: bool
            switch to calculate aggregate data in every
            timestep and to store it in self.aggregates,
        output_path: string
            if set, spatial data output will
            be written to this path.
        """

        # Input/Output settings:

        # Set path to static input files

        input_data_path = pkg_resources. \
            resource_filename('mayasim', 'input/')

        # *********************************************************************
        # MODEL PARAMETERS (to be varied)
        # *********************************************************************
        self.calc_aggregates = calc_aggregates

        # Settlement and geographic data will be
        # written to files in each time step.
        # Aggregate data will be kept in one data structure
        # to be read out, when the model run finished.

        if output_path is not None:

            # remove file ending
            self.output_path = output_path.rsplit('.', 1)[0]

            # create callable output paths
            self.settlement_output_path = \
                lambda i: self.output_path + \
                f'settlement_data_{i:03d}.pkl'
            self.geographic_output_path = \
                lambda i: self.output_path + \
                f'geographic_data_{i:03d}.pkl'

            # set switches for output generation
            self.output_geographic_data = True
            self.output_settlement_data = True
        else:
            self.output_geographic_data = False
            self.output_settlement_data = False

        self.aggregates = []
        self.traders_aggregates = []

        # *********************************************************************
        # MODEL DATA SOURCES
        # *********************************************************************

        # documentation for TEMPERATURE and PRECIPITATION data can be found
        # here: http://www.worldclim.org/formats
        # apparently temperature data is given in x*10 format to allow for
        # smaller file sizes.
        # original version of mayasim divides temperature by 12 though
        self.cel_temp = np.load(input_data_path
                                + '0_RES_432x400_temp.npy') / 12.

        # precipitation in mm or liters per square meter
        # (comparing the numbers to numbers from Wikipedia suggests
        # that it is given per year)
        self.cel_precip_mean = np.load(input_data_path + '0_RES_432x400_precip.npy')

        # in meters above sea level
        self.cel_elev = np.load(input_data_path + '0_RES_432x400_elev.npy')
        self.cel_slope = np.load(input_data_path + '0_RES_432x400_slope.npy')

        # documentation for SOIL PRODUCTIVITY is given at:
        # http://www.fao.org/geonetwork/srv/en/
        # main.home?uuid=f7a2b3c0-bdbf-11db-a0f6-000d939bc5d8
        # The soil production index considers the suitability
        # of the best adapted crop to each soils
        # condition in an area and makes a weighted average for
        #  all soils present in a pixel based
        # on the formula: 0.9 * VS + 0.6 * S + 0.3 * MS + 0 * NS.
        # Values range from 0 (bad) to 6 (good)
        self.cel_soilprod = np.load(input_data_path + '0_RES_432x400_soil.npy')
        # it also sets soil productivity to 1.5 where the elevation is <= 1
        self.cel_soilprod[
            (self.cel_elev <= 1) & (np.logical_not(np.isnan(self.cel_elev)))
            ] = 1.5
        # smoothen soil productivity dataset
        self.cel_soilprod = \
            ndimage.gaussian_filter(self.cel_soilprod, sigma=(2, 2), order=0)
        # and set to zero for non land cells
        self.cel_soilprod[np.isnan(self.cel_elev)] = 0

        # *********************************************************************
        # MODEL MAP INITIALIZATION
        # *********************************************************************

        # dimensions of the map
        self.rows, self.columns = self.cel_precip_mean.shape
        self.height, self.width = 914., 840.  # height and width in km
        self.width_cell = self.width / self.columns
        self.height_cell = self.height / self.rows
        self.land_cell_ind = np.asarray(np.where(np.isfinite(self.cel_elev)))
        self.n_land_cells = self.land_cell_ind.shape[1]

        # lengh unit - total map is about 500 km wide
        self.area = 516484. / self.n_land_cells
        self.cel_elev[:, 0] = np.inf
        self.cel_elev[:, -1] = np.inf
        self.cel_elev[0, :] = np.inf
        self.cel_elev[-1, :] = np.inf
        # create a list of the index values i = (x, y) of the land
        # patches with finite elevation h
        self.land_cell_ind_list = [
            i for i, h in np.ndenumerate(self.cel_elev)

            if np.isfinite(self.cel_elev[i])
        ]

        # initialize soil degradation and population
        # gradient (influencing the forest)

        # *********************************************************************
        # INITIALIZE ECOSYSTEM
        # *********************************************************************

        # Soil (influencing primary production and agricultural productivity)
        self.cel_soil_deg = np.zeros((self.rows, self.columns))

        # Forest
        self.cel_forest_state = np.ones((self.rows, self.columns), dtype=int)
        self.cel_forest_state[np.isnan(self.cel_elev)] = 0

        self.cel_forest_memory = np.zeros((self.rows, self.columns), dtype=int)
        self.cel_cleared_neighs = np.zeros((self.rows, self.columns),
                                           dtype=int)
        # The forest has three states: 3=climax forest,
        # 2=secondary regrowth, 1=cleared land.

        for i in self.land_cell_ind_list:
            self.cel_forest_state[i] = 3

        # Variables describing total amount of water and water flow
        self.cel_water = np.zeros((self.rows, self.columns))
        self.cel_flow = np.zeros((self.rows, self.columns))
        self.cel_precip = np.zeros((self.rows, self.columns))

        # initialize the trajectories of the water drops
        self.x = np.zeros((self.rows, self.columns), dtype="int")
        self.y = np.zeros((self.rows, self.columns), dtype="int")

        # define relative coordinates of the neighbourhood of a cell
        self.neighbourhood = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
        self.f90neighbourhood = np.asarray(self.neighbourhood).T

        # *********************************************************************
        # INITIALIZE SOCIETY
        # *********************************************************************

        # Population gradient (influencing the forest)
        self.cel_pop_gradient = np.zeros((self.rows, self.columns))

        self.n_settlements = n
        # distribute specified number of settlements on the map
        self.stm_positions = self.land_cell_ind[
            :, np.random.choice(len(self.land_cell_ind[1]),n).astype('int')
            ]

        self.stm_age = [0] * n

        # demographic variables
        self.stm_birth_rate = [self.birth_rate_parameter] * n
        self.stm_death_rate = [0.1 + 0.05 * r for r in list(np.random.random(n))]
        self.stm_population = list(
            np.random.randint(self.min_init_inhabitants,
                              self.max_init_inhabitants,
                              n).astype(float))
        self.stm_mig_rate = [0.] * n
        self.stm_out_mig = [0] * n
        self.stm_migrants = [0] * n
        self.stm_pioneer_set = []
        self.n_failed_stm = 0

        # agricultural influence
        self.stm_influenced_cells_n = [0] * n
        self.stm_influenced_area = [0.] * n
        self.cel_ind = np.indices((self.rows, self.columns))
        self.stm_influenced_cells_ind = [None] * n  # will be a list of lists

        self.stm_cropped_cells_ind = [None] * n

        # for now, cropped cells are only the settlement positions.
        # first cropped cells are added at the first call of
        # get_cropped_cells()
        for stm, (x, y) in enumerate(zip(*self.stm_positions)):
            self.stm_cropped_cells_ind[stm] = [(x, y)]

        self.cel_is_cropped = np.zeros((self.rows, self.columns))
        self.stm_cropped_cells_n = [0] * n
        self.stm_crop_yield = [0.] * n
        self.stm_eco_benefit = [0.] * n

        # details of income from ecosystems services
        self.stm_es_ag = [0.] * n
        self.stm_es_wf = [0.] * n
        self.stm_es_fs = [0.] * n
        self.stm_es_sp = [0.] * n
        self.stm_es_pg = [0.] * n
        self.cel_es_ag = np.zeros((self.rows, self.columns), dtype=float)
        self.cel_es_wf = np.zeros((self.rows, self.columns), dtype=float)
        self.cel_es_fs = np.zeros((self.rows, self.columns), dtype=float)
        self.cel_es_sp = np.zeros((self.rows, self.columns), dtype=float)
        self.cel_es_pg = np.zeros((self.rows, self.columns), dtype=float)

        # Trade Variables
        self.stm_adjacency = np.zeros((n, n))
        self.stm_rank = [0] * n
        self.stm_degree = [0] * n
        self.stm_comp_size = [0] * n
        self.stm_centrality = [0] * n
        self.stm_trade_income = [0] * n
        self.max_cluster_size = 0

        # total real income per capita
        self.stm_real_income_pc = [0] * n

    def _get_run_variables(self):
        """
        Returns a dictionary containing all attributes of 'self' 
        and their current values.
        """

        run_variables = {
            attr: getattr(self, attr)

            for attr in dir(self)

            if not attr.startswith('__') and not callable(getattr(self, attr))
        }

        return run_variables

    def update_precipitation(self, t):
        """
        Modulates the initial precip dataset with a 24 timestep period.
        Returns a field of rainfall values for each cell.
        If veg_rainfall > 0, cel_cleared_neighs decreases rain.

        TO DO: The original Model increases specialization every time
        rainfall decreases, assuming that trade gets more important to
        compensate for agriculture decline
        """

        if self.precip_modulation:
            self.cel_precip = \
                self.cel_precip_mean * (
                    1 + self.precip_amplitude * self.precip_variation[
                        (np.ceil(t / self.climate_var) % 8).astype(int)]) \
                - self.veg_rainfall * self.cel_cleared_neighs
        else:
            self.cel_precip = \
                self.cel_precip_mean * (
                    1 - self.veg_rainfall * self.cel_cleared_neighs)

        # check if system time is in drought period
        drought = False
        for drought_time in self.drought_times:
            if drought_time[0] < t <= drought_time[1]:
                drought = True

        # if so, decrease precipitation by factor percentage given by
        # drought severity
        if drought:
            self.cel_precip *= (1. - self.drought_severity / 100.)

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
        rain_volume = np.nan_to_num(self.cel_precip * 1e-5)
        max_x, max_y = self.rows, self.columns
        # pylint: disable-next=unused-variable
        err, self.cel_flow, self.cel_water = \
            f90routines.f90waterflow(self.land_cell_ind,
                                     self.cel_elev,
                                     rain_volume,
                                     self.f90neighbourhood,
                                     max_x,
                                     max_y,
                                     self.n_land_cells)

        return self.cel_water, self.cel_flow

    def evolve_forest(self, cel_npp):
        """
        TODO: add docstring
        """

        # Forest regenerates faster [slower] (linearly),
        # if net primary productivity on the patch
        # is above [below] average.
        threshold = np.nanmean(cel_npp) / cel_npp

        # Iterate over all cells repeatedly and regenerate or degenerate
        for _ in range(4):

            # vectorized random number generation for use in 'Degradation'
            degradation_fortune = \
                np.random.random(len(self.land_cell_ind_list))
            probdec = self.natprobdec * (2 * self.cel_pop_gradient + 1)

            for n, i in enumerate(self.land_cell_ind_list):
                if not np.isnan(self.cel_elev[i]):

                    # Degradation:
                    # Decrement with probability 0.003
                    # if there is a settlement around,
                    # degrade with higher probability

                    if degradation_fortune[n] <= probdec[i]:
                        if self.cel_forest_state[i] == 3:
                            self.cel_forest_state[i] = 2
                            self.cel_forest_memory[i] = self.state_change_s2
                        elif self.cel_forest_state[i] == 2:
                            self.cel_forest_state[i] = 1
                            self.cel_forest_memory[i] = 0

                    # Regeneration
                    # recover if tree = 1 and memory > threshold 1
                    if (self.cel_forest_state[i] == 1 and self.cel_forest_memory[i] >
                            self.state_change_s2 * threshold[i]):
                        self.cel_forest_state[i] = 2
                        self.cel_forest_memory[i] = self.state_change_s2
                    # recover if tree = 2 and memory > threshold 2
                    # and certain number of neighbours are
                    # climax forest as well
                    if (self.cel_forest_state[i] == 2
                        and self.cel_forest_memory[i]
                        > self.state_change_s3 * threshold[i]):
                        state_3_neighbours = \
                            np.sum(self.cel_forest_state[
                                i[0] - 1:i[0] + 2,
                                i[1] - 1:i[1] + 2] == 3)
                        if state_3_neighbours > self.min_s3_neighbours:
                            self.cel_forest_state[i] = 3

                    # finally, increase memory by one
                    self.cel_forest_memory[i] += 1

        # calculate cleared land neighbours for output:
        if self.veg_rainfall > 0:
            for i in self.land_cell_ind_list:
                self.cel_cleared_neighs[i] = \
                    np.sum(self.cel_forest_state[
                        i[0] - 1:i[0] + 2,
                        i[1] - 1:i[1] + 2] == 1)

        # make sure all land cell have forest states 1-3
        assert not np.any(self.cel_forest_state[~np.isnan(self.cel_elev)]
                          < 1), 'forest state is smaller than 1 somewhere'

    def get_npp(self):
        """
        net primary productivity is the minimum of a quantity
        derived from local temperature and rain
        NOTE: Why is it rain and not 'surface water'
        according to the waterflow model?
        """
        # EQUATION ############################################################
        cel_npp = 3000 * np.minimum(
            1 - np.exp(-6.64e-4 * self.cel_precip),
            1. / (1 + np.exp(1.315 - (0.119 * self.cel_temp))))
        # EQUATION ############################################################

        return cel_npp

    def get_ag(self, cel_npp, cel_wf):
        """
        agricultural productivity is calculated via a
        linear additive model from
        net primary productivity, soil productivity,
        slope, waterflow and soil degradation
        of each patch.
        """
        # EQUATION ############################################################
        return (self.a_npp * cel_npp
                + self.a_sp * self.cel_soilprod
                - self.a_s * self.cel_slope
                - self.a_wf * cel_wf
                - self.cel_soil_deg)
        # EQUATION ############################################################

    def get_ecoserv(self, cel_ag, cel_wf):
        """
        Ecosystem Services are calculated via a linear
        additive model from agricultural productivity (cel_ag),
        waterflow through the cell (cel_wf) and forest
        state on the cell (forest) in [1,3],
        The recent version of mayasim limits value of
        ecosystem services to 1 < ecoserv < 250, it also proposes
        to include population density (cel_pop_gradient) and precipitation (rain)
        """
        # EQUATION ############################################################

        if not self.better_ess:
            self.cel_es_ag = self.e_ag * cel_ag
            self.cel_es_wf = self.e_wf * cel_wf
            self.cel_es_fs = self.e_f * (self.cel_forest_state - 1.)
            self.cel_es_sp = self.e_r * self.cel_precip
            self.cel_es_pg = self.e_deg * self.cel_pop_gradient
        else:
            # change to use forest as proxy for income from agricultural
            # productivity. Multiply by 2 to get same per cell levels as
            # before
            self.cel_es_ag = np.zeros(np.shape(cel_ag))
            self.cel_es_wf = self.e_wf * cel_wf
            self.cel_es_fs = 2. * self.e_ag * (self.cel_forest_state - 1.) * cel_ag
            self.cel_es_sp = self.e_r * self.cel_precip
            self.cel_es_pg = self.e_deg * self.cel_pop_gradient

        return (self.cel_es_ag
                + self.cel_es_wf
                + self.cel_es_fs
                + self.cel_es_sp
                - self.cel_es_pg)

        # EQUATION ############################################################

    ###########################################################################
    # The Society
    ###########################################################################

    def benefit_cost(self, cel_ag):
        # Benefit cost assessment

        return self.max_yield * (1 - self.origin_shift * np.exp(
            np.float128(-self.slope_yield * cel_ag)))

    def get_influenced_cells(self):
        """
        creates a list of cells for each stm that are under its influence.
        these are the cells that are closer than population^0.8/60 (which is
        not explained any further... change denominator to 80 and max value to
        30 from eyeballing the results
        """
        # EQUATION ############################################################
        self.stm_influenced_area = [
            (x**0.8) / 60. for x in self.stm_population
            ]
        self.stm_influenced_area = [
            value if value < 40. else 40. for value in self.stm_influenced_area
            ]
        # EQUATION ############################################################

        for stm, (x, y) in enumerate(zip(*self.stm_positions)):
            distance = np.sqrt(
                (self.width_cell * (x - self.cel_ind[0]))**2
                + (self.height_cell * (y - self.cel_ind[1]))**2)
            stencil = distance <= self.stm_influenced_area[stm]
            self.stm_influenced_cells_ind[stm] = \
                list(zip(*self.cel_ind[:, stencil]))

        self.stm_influenced_cells_n = \
            [len(ifd) for ifd in self.stm_influenced_cells_ind]

    # pylint: disable=too-many-locals
    def get_cropped_cells(self, cel_bca):
        """
        Updates the cropped cells for each stm with positive population.
        Calculates the utility for each cell (depending on distance from
        the respective stm) If population per cropped cell is lower then
        min_people_per_cropped_cell, cells are abandoned.
        Cells with negative utility are also abandoned.
        If population per cropped cell is higher than
        max_people_per_cropped_cell, new cells are cropped.
        Newly cropped cells are chosen such that they have highest utility
        """
        abandoned = 0
        sown = 0

        # number of currently cropped cells for each settlement
        self.stm_cropped_cells_n = \
            [len(crp) for crp in self.stm_cropped_cells_ind]

        # agricultural population density (people per cropped land)
        # determines the number of cells that can be cropped.
        ag_pop_density = [
            pop / (self.stm_cropped_cells_n[stm] * self.area)

            if self.stm_cropped_cells_n[stm] > 0 else 0.

            for stm, pop in enumerate(self.stm_population)
            ]

        # cel_is_occupied is a mask of all cropped cells of all settlements
        for cel in chain(*self.stm_cropped_cells_ind):
            self.cel_is_cropped[cel] = 1

        # the age of settlements is increased here.
        self.stm_age = [x + 1 for x in self.stm_age]

        # for each settlement: which cells to crop ?
        # calculate utility first! This can be accelerated, if calculations
        # are only done in 40 km radius.

        for stm, (x, y) in enumerate(zip(*self.stm_positions)):

            # get arrays for vectorized utility calculation
            infd_cells = np.transpose(np.array(
                self.stm_influenced_cells_ind[stm]))
            distance = np.sqrt(
                (self.width_cell * (x - infd_cells[0]))**2
                + (self.height_cell * (y - infd_cells[1]))**2)

            # EQUATION ########################################################
            utility = (
                cel_bca[infd_cells[0], infd_cells[1]] - self.estab_cost
                - self.ag_travel_cost * distance / np.sqrt(
                    self.stm_population[stm])
                ).tolist()
            # EQUATION ########################################################

            # do rest of operations using tuple-lists and list-comps
            infd_cells = self.stm_influenced_cells_ind[stm]
            available = [self.cel_is_cropped[cel] == 0 for cel in infd_cells]

            # jointly sort utilities, availability and cells such that cells
            # with highest utility are first.
            sorted_utility, sorted_available, sorted_cells = \
                list(zip(*sorted(
                    list(zip(utility, available, infd_cells)), reverse=True)))
            # of these sorted lists, sort filter only available cells
            available_util = list(
                compress(list(sorted_utility), list(sorted_available)))
            available_cells = list(
                compress(list(sorted_cells), list(sorted_available)))

            # save local copy of all cropped cells
            cropped_cells = self.stm_cropped_cells_ind[stm]
            # select utilities for these cropped cells
            cropped_utils = [
                utility[infd_cells.index(cel)] if cel in infd_cells else -1

                for cel in cropped_cells
                ]

            # sort utilitites and cropped cells to lowest utilities first
            settlement_has_crops = len(cropped_cells) > 0

            if settlement_has_crops:
                occupied_util, occupied_cells = \
                    zip(*sorted(list(zip(cropped_utils, cropped_cells))))

            # 1.) include new cells if population exceeds a threshold

            # calculate number of new cells to crop
            number_of_new_cells = np.floor(
                ag_pop_density[stm]/self.max_people_per_cropped_cell
                ).astype('int')
            # and crop them by selecting cells with positive utility from the
            # beginning of the list

            for n in range(min([number_of_new_cells, len(available_util)])):
                if available_util[n] > 0:
                    self.stm_cropped_cells_ind[stm].append(available_cells[n])
                    self.cel_is_cropped[available_cells[n]] = 1

                    sown += 1

            if settlement_has_crops:

                # 2.) abandon cells if population too low
                # after settlement's age > 5 years

                if (ag_pop_density[stm] < self.min_people_per_cropped_cell
                        and self.stm_age[stm] > 5):

                    # There are some inconsistencies here. Cells are abandoned,
                    # if the 'people per cropped land' is lower then a
                    # threshold for 'people per cropped cells. Then the
                    # number of cells to abandon is calculated as 30/people
                    # per cropped land. Why?! (check the original version!)

                    number_of_lost_cells = np.ceil(
                        30 / ag_pop_density[stm]).astype('int')

                    # TO DO: recycle utility and cell list to do this faster.
                    # therefore, filter cropped cells from utility list
                    # and delete last n cells.

                    for n in range(min([number_of_lost_cells,
                                        len(occupied_cells)])):

                        self.stm_cropped_cells_ind[stm] \
                            .remove(occupied_cells[n])
                        self.cel_is_cropped[occupied_cells[n]] = 0

                        abandoned += 1

                # 3.) abandon cells with utility <= 0

                # find cells that have negative utility and belong
                # to stm under consideration,
                useless_cropped_cells = [
                    occupied_cells[i] for i in range(len(occupied_cells))

                    if occupied_util[i] < 0
                    and occupied_cells[i] in self.stm_cropped_cells_ind[stm]
                    ]

                # and release them.
                for cel in useless_cropped_cells:
                    try:
                        self.stm_cropped_cells_ind[stm] \
                            .remove(cel)
                    except ValueError:
                        print('ERROR: Useless cell gone already')
                    self.cel_is_cropped[cel] = 0

                    abandoned += 1

        # Finally, update list of lists containing cropped cells for each stm
        # with positive population.
        self.stm_cropped_cells_n = [
            len(crp) for crp in self.stm_cropped_cells_ind]

        return abandoned, sown

    def get_pop_mig(self):
        # gives population and out-migration
        # print("number of settlements", len(self.stm_population))

        # death rate correlates inversely with real income per capita
        death_rate_diff = self.max_death_rate - self.min_death_rate

        self.stm_death_rate = [
            -death_rate_diff * self.stm_real_income_pc[i] + self.max_death_rate

            for i in range(len(self.stm_real_income_pc))
            ]

        self.stm_death_rate = list(np.clip(
            self.stm_death_rate, self.min_death_rate, self.max_death_rate))

        # if population control,
        # birth rate negatively correlates with population size

        if self.population_control:
            birth_rate_diff = self.max_birth_rate - self.min_birth_rate

            self.stm_birth_rate = [
                -birth_rate_diff / 10000. * value +
                self.shift if value > 5000 else self.birth_rate_parameter

                for value in self.stm_population
                ]

        # population grows according to effective growth rate
        self.stm_population = [
            int((1. + self.stm_birth_rate[i] - self.stm_death_rate[i]) * value)

            for i, value in enumerate(self.stm_population)
            ]

        self.stm_population = [
            value if value > 0 else 0 for value in self.stm_population
            ]

        mig_rate_diffe = self.max_mig_rate - self.min_mig_rate

        # outmigration rate also correlates
        # inversely with real income per capita
        self.stm_mig_rate = [
            -mig_rate_diffe * self.stm_real_income_pc[i] + self.max_mig_rate

            for i in range(len(self.stm_real_income_pc))
            ]
        self.stm_mig_rate = list(
            np.clip(self.stm_mig_rate, self.min_mig_rate, self.max_mig_rate))
        self.stm_out_mig = [
            int(self.stm_mig_rate[i] * self.stm_population[i])

            for i in range(len(self.stm_population))
            ]
        self.stm_out_mig = [
            value if value > 0 else 0 for value in self.stm_out_mig]

    # impact of sociosphere on ecosphere
    def update_pop_gradient(self):
        # pop gradient quantifies the disturbance of the forest by population
        self.cel_pop_gradient = np.zeros((self.rows, self.columns))

        for stm, (x, y) in enumerate(zip(*self.stm_positions)):
            infd_cells = np.transpose(np.array(
                self.stm_influenced_cells_ind[stm]))
            distance = np.sqrt(self.area * (
                (x - infd_cells[0])**2
                + (y - infd_cells[1])**2))

            # EQUATION ########################################################
            self.cel_pop_gradient[infd_cells[0], infd_cells[1]] += \
                self.stm_population[stm] / (300 * (1 + distance))
            # EQUATION ########################################################
            self.cel_pop_gradient[self.cel_pop_gradient > 15] = 15

    def evolve_soil_deg(self):
        # soil degrades for cropped cells
        for cel in chain(*self.stm_cropped_cells_ind):
            self.cel_soil_deg[cel] += self.deg_rate
        # soil regenerates for climax-forest cells
        self.cel_soil_deg[self.cel_forest_state == 3] -= self.reg_rate
        self.cel_soil_deg[self.cel_soil_deg < 0] = 0

    def get_rank(self):
        # depending on population ranks are assigned
        # attention: ranks are reverted with respect to Netlogo MayaSim !
        # 1 => 3 ; 2 => 2 ; 3 => 1
        self.stm_rank = [
            3

            if value > self.thresh_rank_3 else 2 if value > self.thresh_rank_2
            else 1 if value > self.thresh_rank_1 else 0

            for index, value in enumerate(self.stm_population)
            ]


    @property
    def build_routes(self):

        adj = self.stm_adjacency.copy()
        adj[adj == -1] = 0
        built_links = 0
        lost_links = 0
        g = nx.from_numpy_array(adj, create_using=nx.DiGraph())
        self.stm_degree = g.out_degree()
        # settlements with rank>0 are traders and establish links to neighbours

        for stm, (x, y) in enumerate(zip(*self.stm_positions)):
            if self.stm_degree[stm] < self.stm_rank[stm]:
                distances = np.sqrt(self.area * (
                    (x - self.stm_positions[0]) ** 2
                    + (y - self.stm_positions[1]) ** 2))

                if self.stm_rank[stm] == 3:
                    threshold = 31. * (
                        self.thresh_rank_3 / self.thresh_rank_3 * 0.5 + 1.)
                elif self.stm_rank[stm] == 2:
                    threshold = 31. * (
                        self.thresh_rank_2 / self.thresh_rank_3 * 0.5 + 1.)
                elif self.stm_rank[stm] == 1:
                    threshold = 31. * (
                        self.thresh_rank_1 / self.thresh_rank_3 * 0.5 + 1.)
                else:
                    threshold = 0

                # don't chose yourself as nearest neighbor
                distances[stm] = 2 * threshold

                # collect close enough neighbors and omit those that are
                # already connected.
                a = distances <= threshold
                b = self.stm_adjacency[stm] == 0
                nearby = np.array(list(map(operator.and_, a, b)))

                # if there are traders nearby,
                # connect to the one with highest population

                if sum(nearby) != 0:
                    try:
                        new_partner = np.nanargmax(
                            self.stm_population * nearby)
                        self.stm_adjacency[stm, new_partner] = 1
                        self.stm_adjacency[new_partner, stm] = -1
                        built_links += 1
                    except ValueError:
                        print('ERROR in new partner')
                        print(np.shape(self.stm_population),
                              np.shape(self.stm_positions[0]))
                        sys.exit(-1)

            # settlements that cannot maintain their trade links lose them:
            elif self.stm_degree[stm] > self.stm_rank[stm]:
                # get neighbors of node
                neighbors = g.successors(stm)
                # find smallest of neighbors
                smallest_neighbor = self.stm_population.index(
                    min(self.stm_population[nb] for nb in neighbors))
                # cut corresponding link
                self.stm_adjacency[stm, smallest_neighbor] = 0
                self.stm_adjacency[smallest_neighbor, stm] = 0
                lost_links += 1

        return (built_links, lost_links)

    def get_comps(self):
        # convert adjacency matrix to compressed sparse row format
        adjacency_csr = sparse.csr_matrix(np.absolute(self.stm_adjacency))

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
                f90routines.f90sparsecomponents(
                    i_c, a, j_a,self.n_settlements,l_ic, l_a
                    )
            self.stm_comp_size, self.stm_degree = \
                list(tmp_comp_size), list(tmp_degree)
        elif l_a == 0:
            self.stm_comp_size, self.stm_degree = \
                [0] * (l_ic - 1), [0] * (l_ic - 1)

    def get_centrality(self):
        # convert adjacency matrix to compressed sparse row format
        adjacency_csr = sparse.csr_matrix(np.absolute(self.stm_adjacency))

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
            tmp_centrality = \
                f90routines.f90sparsecentrality(
                    i_c, a, j_a, self.n_settlements, l_ic, l_a
                    )
            self.stm_centrality = list(tmp_centrality)
        elif l_a == 0:
            self.stm_centrality = [1] * (l_ic - 1)

    def get_crop_income(self, cel_bca):
        # agricultural benefit of cropping
        for stm, crpd_cells in enumerate(self.stm_cropped_cells_ind):
            # get bca of settlement's cropped cells
            bca = np.array([cel_bca[cel] for cel in crpd_cells])

            # EQUATION #
            if self.crop_income_mode == "mean":
                self.stm_crop_yield[stm] = self.r_bca_mean \
                    * np.nanmean(bca[bca > 0])
            elif self.crop_income_mode == "sum":
                self.stm_crop_yield[stm] = self.r_bca_sum \
                    * np.nansum(bca[bca > 0])

        self.stm_crop_yield = [
            0 if np.isnan(self.stm_crop_yield[index])
            else self.stm_crop_yield[index]

            for index in range(len(self.stm_crop_yield))
            ]

    def get_eco_income(self, cel_es):
        # benefit from ecosystem services of cells in influence
        # ##EQUATION###########################################################
        if self.eco_income_mode == "mean":
            for stm, infd_cells in enumerate(self.stm_influenced_cells_ind):
                infd_index = np.transpose(np.array(infd_cells))
                self.stm_eco_benefit[stm] = self.r_es_mean \
                    * np.nanmean(cel_es[infd_index])

        elif self.eco_income_mode == "sum":
            for stm, infd_cells in enumerate(self.stm_influenced_cells_ind):
                r = self.r_es_sum
                infd_index = np.transpose(np.array(infd_cells))
                self.stm_eco_benefit[stm] = r * np.nansum(cel_es[infd_index])
                self.stm_es_ag[stm] = r * np.nansum(self.cel_es_ag[infd_index])
                self.stm_es_wf[stm] = r * np.nansum(self.cel_es_wf[infd_index])
                self.stm_es_fs[stm] = r * np.nansum(self.cel_es_fs[infd_index])
                self.stm_es_sp[stm] = r * np.nansum(self.cel_es_sp[infd_index])
                self.stm_es_pg[stm] = r * np.nansum(self.cel_es_pg[infd_index])

        self.stm_eco_benefit[self.stm_population == 0] = 0
        # ##EQUATION###########################################################

    def get_trade_income(self):
        # ##EQUATION###########################################################
        self.stm_trade_income = [
            1. / 30.
            * (1 + self.stm_comp_size[i] / self.stm_centrality[i])**0.9

            for i in range(len(self.stm_centrality))
            ]
        self.stm_trade_income = [
            self.r_trade
            if value > 1 else 0
            if (value < 0 or self.stm_degree[index] == 0)
            else self.r_trade * value

            for index, value in enumerate(self.stm_trade_income)
            ]
        # ##EQUATION###########################################################

    def get_real_income_pc(self):
        # combine agricultural, ecosystem service and trade benefit
        # EQUATION #
        self.stm_real_income_pc = [
            (self.stm_crop_yield[index] + self.stm_eco_benefit[index] +
             self.stm_trade_income[index]) /
            self.stm_population[index] if value > 0 else 0

            for index, value in enumerate(self.stm_population)
        ]

    def migration(self, cel_es):
        # if outmigration rate exceeds threshold, found new settlement
        self.stm_migrants = [0] * self.n_settlements
        new_settlements = 0

        # create mask of cells that are not influenced by any settlement
        uninfluenced = np.isfinite(cel_es)
        for cel in chain(*self.stm_influenced_cells_ind):
            uninfluenced[cel] = 0
        # create index-array of uninfluenced cells
        uninfd_index = np.asarray(np.where(uninfluenced == 1))

        for stm, (x,y) in enumerate(zip(*self.stm_positions)):

            if (self.stm_out_mig[stm] > 400 and len(uninfd_index[0]) > 0
                    and np.random.rand() <= 0.5):

                mig_pop = self.stm_out_mig[stm]
                self.stm_migrants[stm] = mig_pop
                self.stm_population[stm] -= mig_pop
                self.stm_pioneer_set = \
                    uninfd_index[:, np.random.choice(len(uninfd_index[0]), 75)]

                travel_cost = np.sqrt(self.area * (
                    (x - self.cel_ind[0])**2 + (y -self.cel_ind[1])**2))

                utility = self.mig_ES_pref * cel_es \
                    + self.mig_TC_pref * travel_cost
                utofpio = \
                    utility[self.stm_pioneer_set[0], self.stm_pioneer_set[1]]
                new_loc = self.stm_pioneer_set[:, np.nanargmax(utofpio)]

                neighbours = np.sqrt(self.area * (
                    (new_loc[0] - self.stm_positions[0]) ** 2
                    + (new_loc[1] - self.stm_positions[1]) ** 2
                    )) <= 7.5

                if np.sum(neighbours) == 0:
                    self.spawn_settlement(new_loc[0], new_loc[1], mig_pop)
                    index = ((uninfd_index[0, :] == new_loc[0])
                             & (uninfd_index[1, :] == new_loc[1]))
                    np.delete(uninfd_index, int(np.where(index)[0]), 1)
                    new_settlements += 1

        return new_settlements

    def kill_settlements(self):

        killed_stm = 0

        # kill settlements if they have either no crops or no inhabitants:
        dead_stm_ind = [
            stm for stm in range(self.n_settlements)

            if self.stm_population[stm] <= self.min_stm_size
            ]

        if self.kill_stm_without_crops:
            dead_stm_ind += [
                stm for stm in range(self.n_settlements)

                if not self.stm_cropped_cells_ind[stm]
                ]

        # only keep unique entries
        dead_stm_ind = list(set(dead_stm_ind))

        # remove settlement attributes from attribute-lists
        for index in sorted(dead_stm_ind, reverse=True):
            self.n_settlements -= 1
            self.n_failed_stm += 1
            del self.stm_age[index]
            del self.stm_birth_rate[index]
            del self.stm_death_rate[index]
            del self.stm_population[index]
            del self.stm_mig_rate[index]
            del self.stm_out_mig[index]
            del self.stm_influenced_cells_n[index]
            del self.stm_influenced_area[index]
            del self.stm_cropped_cells_n[index]
            del self.stm_crop_yield[index]
            del self.stm_eco_benefit[index]
            del self.stm_rank[index]
            del self.stm_degree[index]
            del self.stm_comp_size[index]
            del self.stm_centrality[index]
            del self.stm_trade_income[index]
            del self.stm_real_income_pc[index]
            del self.stm_influenced_cells_ind[index]
            del self.stm_cropped_cells_ind[index]
            del self.stm_es_ag[index]
            del self.stm_es_wf[index]
            del self.stm_es_fs[index]
            del self.stm_es_sp[index]
            del self.stm_es_pg[index]
            del self.stm_migrants[index]

            killed_stm += 1

        # special cases:
        self.stm_positions = \
            np.delete(self.stm_positions,
                      dead_stm_ind, axis=1)
        self.stm_adjacency = \
            np.delete(np.delete(self.stm_adjacency,dead_stm_ind, axis=0),
                      dead_stm_ind, axis=1)

        return killed_stm

    def spawn_settlement(self, x, y, mig_pop):
        """
        Spawn a new stm at given location with
        given population and append it to all necessary lists.

        Parameters
        ----------
        x: int
            x location of new stm on map
        y: int
            y location of new stm on map
        mig_pop: int
            initial population of new stm
        """

        # extend all variables to include new stm
        self.n_settlements += 1
        self.stm_positions = np.append(self.stm_positions,
                                              [[x], [y]], 1)
        self.stm_influenced_cells_ind.append([(x, y)])
        self.stm_cropped_cells_ind.append([(x, y)])

        n = len(self.stm_adjacency)
        self.stm_adjacency = np.append(self.stm_adjacency, [[0] * n], 0)
        self.stm_adjacency = np.append(self.stm_adjacency, [[0]] * (n + 1), 1)

        self.stm_age.append(0)
        self.stm_birth_rate.append(self.birth_rate_parameter)
        self.stm_death_rate.append(0.1 + 0.05 * np.random.rand())
        self.stm_population.append(mig_pop)
        self.stm_mig_rate.append(0)
        self.stm_out_mig.append(0)
        self.stm_influenced_cells_n.append(0)
        self.stm_influenced_area.append(0)
        self.stm_cropped_cells_n.append(1)
        self.stm_crop_yield.append(0)
        self.stm_eco_benefit.append(0)
        self.stm_rank.append(0)
        self.stm_degree.append(0)
        self.stm_trade_income.append(0)
        self.stm_real_income_pc.append(0)
        self.stm_es_ag.append(0)
        self.stm_es_wf.append(0)
        self.stm_es_fs.append(0)
        self.stm_es_sp.append(0)
        self.stm_es_pg.append(0)
        self.stm_migrants.append(0)

    def run(self, t_max=1):
        """
        Run the model for a given number of steps.
        If no number of steps is given, the model is integrated for one step

        Parameters
        ----------
        t_max: int
            number of steps to integrate the model
        """

        self.init_output()

        # initialize progress bar
        t_range = trange(1, t_max+1,
                         desc='running MayaSim',
                         postfix={'population': sum(self.stm_population)})

        for t in t_range:
            # evolve subselfs

            # ecosystem
            self.update_precipitation(t)
            # net primary productivity
            cel_npp = self.get_npp()
            self.evolve_forest(cel_npp)

            # water flow
            # NOTE: this is curious, only waterflow
            # is used, water level is abandoned.
            cel_wf = self.get_waterflow()[1]
            # agricultural productivity
            cel_ag = self.get_ag(cel_npp, cel_wf)
            # ecosystem services
            cel_es = self.get_ecoserv(cel_ag, cel_wf)
            # benefit cost map for agriculture
            cel_bca = self.benefit_cost(cel_ag)

            # society
            if len(self.stm_population) > 0:
                # ag income
                self.get_influenced_cells()
                abandoned, sown = self.get_cropped_cells(cel_bca)
                self.get_crop_income(cel_bca)
                # es income
                self.get_eco_income(cel_es)
                self.evolve_soil_deg()
                # tr income
                self.update_pop_gradient()
                self.get_rank()
                (built, lost) = self.build_routes
                self.get_comps()
                self.get_centrality()
                self.get_trade_income()
                # total income
                self.get_real_income_pc()
                # migration
                self.get_pop_mig()
                new_settlements = self.migration(cel_es)
                killed_settlements = self.kill_settlements()
            else:
                abandoned = sown = 0

            # update population counter on progress bar
            t_range.set_postfix({'population': sum(self.stm_population)})

            self.step_output(t, cel_npp, cel_wf, cel_ag, cel_es, cel_bca, abandoned, sown, built,
                             lost, new_settlements, killed_settlements)


    def init_output(self):
        """
        initializes data output for aggregates, settlements
        and geography depending on settings
        """

        if self.calc_aggregates:
            self.init_aggregates()
            self.init_traders_aggregates()

        if self.output_geographic_data or self.output_settlement_data:
            # If output data location is needed and does not exist, create it.
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not self.output_path.endswith('/'):
                self.output_path += '/'

        if self.output_settlement_data:
            settlement_init_data = {'shape': (self.rows, self.columns)}
            with open(self.settlement_output_path(0), 'wb') as f:
                pkl.dump(settlement_init_data, f)

        if self.output_geographic_data:
            pass

    def step_output(self, t, cel_npp, cel_wf, cel_ag, cel_es, cel_bca, abandoned, sown, built,
                    lost, new_settlements, killed_settlements):
        """
        call different data saving routines depending on settings.

        Parameters
        ----------
        t: int
            Timestep number to append to save file path
        cel_npp: numpy array
            Net Primary Productivity on cell basis
        cel_wf: numpy array
            Water flow through cell
        cel_ag: numpy array
            Agricultural productivity of cell
        cel_es: numpy array
            Ecosystem services of cell (that are summed and weighted to
            calculate ecosystems service income)
        cel_bca: numpy array
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

        # append stuff to aggregates

        if self.calc_aggregates:
            self.update_aggregates(
                t, [cel_npp, cel_wf, cel_ag, cel_es, cel_bca], built, lost,
                new_settlements, killed_settlements
                )
            self.update_traders_aggregates(t)

        # save maps of spatial data

        if self.output_geographic_data:
            self.save_geographic_output(t, cel_wf, abandoned, sown)

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
            self.stm_population, self.stm_real_income_pc, self.stm_crop_yield,
            self.stm_eco_benefit, self.stm_trade_income,
            list(self.stm_positions[0]), list(self.stm_positions[1]),
            self.stm_migrants, self.stm_degree
            ]

        data = list(map(list, zip(*data)))

        data_frame = pandas.DataFrame(columns=colums, data=data)

        with open(self.settlement_output_path(t), 'wb') as f:
            pkl.dump(data_frame, f)

    def save_geographic_output(self, t, cel_wf, abandoned, sown):
        """
        Organize Geographic data in dictionary (for separate layers
        of data) and save to file.

        Parameters
        ----------
        t: int
            Timestep number to append to save file path
        cel_wf: numpy array
            Water flow through cell
        abandoned: int
            Number of cells that was abandoned in the previous time step
        sown: int
            Number of cells that was newly cropped in the previous time step
        """

        tmpforest = self.cel_forest_state.copy()
        tmpforest[np.isnan(self.cel_elev)] = 0
        data = {
            'forest': tmpforest,
            'waterflow': cel_wf,
            'cells in influence': self.stm_influenced_cells_ind,
            'number of cells in influence': self.stm_influenced_cells_n,
            'cropped cells': self.stm_cropped_cells_ind,
            'number of cropped cells': self.stm_cropped_cells_n,
            'abandoned sown': np.array([abandoned, sown]),
            'soil degradation': self.cel_soil_deg,
            'population gradient': self.cel_pop_gradient,
            'adjacency': self.stm_adjacency,
            'x positions': list(self.stm_positions[0]),
            'y positions': list(self.stm_positions[1]),
            'population': self.stm_population,
            'elev': self.cel_elev,
            'rank': self.stm_rank
            }

        with open(self.geographic_output_path(t), 'wb') as f:
            pkl.dump(data, f)

    def init_aggregates(self):
        self.aggregates.append([
            'time', 'total_population', 'max_settlement_population',
            'total_migrants', 'total_settlements', 'total_cropped_cells',
            'total_influenced_cells', 'total_trade_links',
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

    def init_traders_aggregates(self):
        self.traders_aggregates.append([
            'time', 'total_population', 'total_migrants', 'total_traders',
            'total_settlements', 'total_cropped_cells',
            'total_influenced_cells', 'total_trade_links',
            'total_income_agriculture', 'total_income_ecosystem',
            'total_income_trade', 'es_income_forest', 'es_income_waterflow',
            'es_income_agricultural_productivity', 'es_income_precipitation',
            'es_income_pop_density'
            ])

    def update_aggregates(self, time, args, built, lost,
                                 new_settlements, killed_settlements):
        # args = [cel_npp, cel_wf, cel_ag, cel_es, cel_bca]

        total_population = sum(self.stm_population)
        try:
            max_population = np.nanmax(self.stm_population)
        except: # pylint: disable=bare-except
            max_population = float('nan')
        total_migrangs = sum(self.stm_migrants)
        total_settlements = len(self.stm_population)
        total_trade_links = sum(self.stm_degree) / 2
        income_agriculture = sum(self.stm_crop_yield)
        income_ecosystem = sum(self.stm_eco_benefit)
        income_trade = sum(self.stm_trade_income)
        number_of_components = float(
            sum(1 if value > 0 else 0 for value in self.stm_comp_size))
        mean_cluster_size = (
            float(sum(self.stm_comp_size)) / number_of_components
            if number_of_components > 0 else 0
            )
        try:
            max_cluster_size = max(self.stm_comp_size)
        except: # pylint: disable=bare-except
            max_cluster_size = 0
        self.max_cluster_size = max_cluster_size
        total_cropped_cells = sum(self.stm_cropped_cells_n)
        total_influenced_cells = sum(self.stm_influenced_cells_n)

        self.aggregates.append([
            time, total_population, max_population, total_migrangs,
            total_settlements, total_cropped_cells,
            total_influenced_cells, total_trade_links, mean_cluster_size,
            max_cluster_size, new_settlements, killed_settlements, built, lost,
            income_agriculture, income_ecosystem, income_trade,
            np.nanmean(self.cel_soil_deg),
            np.sum(self.cel_forest_state == 3),
            np.sum(self.cel_forest_state == 2),
            np.sum(self.cel_forest_state == 1),
            np.sum(self.stm_es_fs),
            np.sum(self.stm_es_wf),
            np.sum(self.stm_es_ag),
            np.sum(self.stm_es_sp),
            np.sum(self.stm_es_pg),
            np.nanmean(self.cel_precip),
            np.nanmax(args[0]),
            np.nanmean(args[1]),
            np.nanmax(args[2]),
            np.nanmax(args[3]),
            np.nanmax(args[4]),
            np.nanmax(self.cel_soil_deg),
            np.nanmax(self.cel_pop_gradient)
            ])

    def update_traders_aggregates(self, time):

        traders = np.where(np.array(self.stm_degree) > 0)[0]

        total_population = sum(self.stm_population[c] for c in traders)
        total_migrants = sum(self.stm_migrants[c] for c in traders)
        total_settlements = len(self.stm_population)
        total_traders = len(traders)
        total_trade_links = sum(self.stm_degree) / 2
        income_agriculture = sum(self.stm_crop_yield[c] for c in traders)
        income_ecosystem = sum(self.stm_eco_benefit[c] for c in traders)
        income_trade = sum(self.stm_trade_income[c] for c in traders)
        income_es_fs = sum(self.stm_es_fs[c] for c in traders)
        income_es_wf = sum(self.stm_es_wf[c] for c in traders)
        income_es_ag = sum(self.stm_es_ag[c] for c in traders)
        income_es_sp = sum(self.stm_es_sp[c] for c in traders)
        income_es_pg = sum(self.stm_es_pg[c] for c in traders)

        total_cropped_cells = \
            sum(self.stm_cropped_cells_n[c] for c in traders)
        total_influenced_cells = \
            sum(self.stm_influenced_cells_n[c] for c in traders)

        self.traders_aggregates.append([
            time, total_population, total_migrants, total_traders,
            total_settlements, total_cropped_cells,
            total_influenced_cells, total_trade_links, income_agriculture,
            income_ecosystem, income_trade, income_es_fs, income_es_wf,
            income_es_ag, income_es_sp, income_es_pg
            ])

    def get_aggregates(self):
        if self.calc_aggregates:
            trj = np.array(self.aggregates)
            columns = trj[0, :]
            df = pandas.DataFrame(trj[1:, :], columns=columns, dtype='float')
        else:
            print("Error: 'calc_aggregates' was set to 'False'")

        return df

    def get_traders_aggregates(self):
        if self.calc_aggregates:
            trj = self.traders_aggregates
            columns = trj.pop(0)
            df = pandas.DataFrame(trj, columns=columns)
        else:
            print("Error: 'calc_aggregates' was set to 'False'")

        return df
