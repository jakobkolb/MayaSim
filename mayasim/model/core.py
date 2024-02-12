# Authors: Jakob J. Kolb, Fritz Kühlein

import os
from itertools import compress, chain
from random import sample

import pickle as pkl
import networkx as nx
import numpy as np
from numpy.random import rand
from numpy.typing import NDArray
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
    ... model.run(steps=350)
    """
    # pylint: disable=too-many-statements
    def __init__(self,
                 n_init: int = 30,
                 calc_aggregates: bool = True,
                 output_path: str = None):
        """
        Returns an instance of the MayaSim model.

        Parameters
        ----------
        n_init: int
            initial number of settlements,
        calc_aggregates: bool
            calculate aggregate data in every
            timestep and store it in self.aggregates,
        output_path: string
            if given, spatial data output will
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

        if output_path:
            # remove file ending
            self.output_path = output_path.rsplit('.', 1)[0]
            # create callable output paths
            self.settlement_output_path = \
                lambda timestep: self.output_path + \
                f'settlement_data_{timestep:03d}.pkl'
            self.geographic_output_path = \
                lambda timestep: self.output_path + \
                f'geographic_data_{timestep:03d}.pkl'
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
        self.cel_temp = np.load(
            input_data_path + '0_RES_432x400_temp.npy') / 12.

        # precipitation in mm or liters per square meter
        # (comparing the numbers to numbers from Wikipedia suggests
        # that it is given per year)
        self.cel_precip_mean = np.load(
            input_data_path + '0_RES_432x400_precip.npy')

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
        self.map_shape = self.cel_temp.shape
        self.height, self.width = 914., 840.  # height and width in km
        self.height_cell = self.height / self.map_shape[0]
        self.width_cell = self.width / self.map_shape[1]

        # find land cells
        self.land_cell_index = np.asarray(np.where(~np.isnan(self.cel_elev)))
        self.land_cells = list(zip(*self.land_cell_index))
        self.n_land_cells = len(self.land_cells)

        # lengh unit - total map is about 500 km wide
        # NOTE: this seems questionable, where does this value come from?
        self.area = 516484. / self.n_land_cells

        # initialize soil degradation and population
        # gradient (influencing the forest)

        # *********************************************************************
        # INITIALIZE ECOSYSTEM
        # *********************************************************************

        # Soil (influencing primary production and agricultural productivity)
        self.cel_soil_deg = np.zeros(self.map_shape)

        # Forest
        self.cel_forest_state = np.zeros(self.map_shape, dtype=int)
        self.cel_forest_memory = np.zeros(self.map_shape, dtype=int)
        self.cel_cleared_neighs = np.zeros(self.map_shape, dtype=int)
        # forest states: 3=climax forest, 2=secondary regrowth, 1=cleared land
        # set all land cells to 'climax forest'
        self.cel_forest_state[*self.land_cell_index] = 3

        # Variables describing total amount of water and water flow
        self.cel_waterlevel = np.zeros(self.map_shape)
        self.cel_waterflow = np.zeros(self.map_shape)
        self.cel_precip = np.zeros(self.map_shape)

        # define relative coordinates of the neighbourhood of a cell
        # NOTE: why is this not happening within the f90 module?
        neighbourhood = [(y, x) for y in [-1, 0, 1] for x in [-1, 0, 1]]
        self.f90neighbourhood = np.asarray(neighbourhood).T

        # *********************************************************************
        # INITIALIZE SOCIETY
        # *********************************************************************

        # Population gradient (influencing the forest)
        self.cel_pop_gradient = np.zeros(self.map_shape)

        self.n_settlements = n_init
        # randomly distribute specified number of settlements on the map
        self.stm_positions = sample(self.land_cells, self.n_settlements)

        self.stm_age = [0] * n_init

        # demographic variables
        self.stm_birth_rate = [self.birth_rate_parameter] * n_init
        self.stm_death_rate = [0.1 + 0.05 * r for r in list(rand(n_init))]
        self.stm_population = list(
            np.random.randint(self.min_init_inhabitants,
                              self.max_init_inhabitants,
                              n_init).astype(float))
        self.stm_mig_rate = [0.] * n_init
        self.stm_out_mig = [0] * n_init
        self.stm_migrants = [0] * n_init
        self.n_failed_stm = 0

        # area of influence
        self.stm_influenced_cells_n = [0] * n_init
        self.stm_influence_rad = [0.] * n_init
        self.stm_influenced_cells = [None] * n_init

        # agriculture/cropping
        self.cel_is_cropped = np.zeros(self.map_shape)
        self.stm_cropped_cells_n = [0] * n_init
        self.stm_cropped_cells = [None] * n_init
        # add settlement positions until get_cropped_cells() is first called
        for stm, pos in enumerate(self.stm_positions):
            self.stm_cropped_cells[stm] = [pos]

        self.stm_crop_yield = [0.] * n_init
        self.stm_eco_benefit = [0.] * n_init

        # components of ecosystem service income
        self.cel_es_ag = np.zeros(self.map_shape, dtype=float)
        self.cel_es_wf = np.zeros(self.map_shape, dtype=float)
        self.cel_es_fs = np.zeros(self.map_shape, dtype=float)
        self.cel_es_sp = np.zeros(self.map_shape, dtype=float)
        self.cel_es_pg = np.zeros(self.map_shape, dtype=float)

        # Trade Variables
        self.stm_adjacency = np.zeros((n_init, n_init))
        self.stm_rank = [0] * n_init
        self.stm_degree = [0] * n_init
        self.stm_comp_size = [0] * n_init
        self.stm_centrality = [0] * n_init
        self.stm_trade_income = [0] * n_init
        self.max_cluster_size = 0

        # total real income per capita
        self.stm_real_income_pc = [0] * n_init

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

    def update_precipitation(self, timestep: int):
        """
        Modulates the initial precip dataset with a 24 timestep period.
        Returns a field of rainfall values for each cell.
        If veg_rainfall > 0, cel_cleared_neighs decreases rain.
        """

        if self.precip_modulation:
            self.cel_precip = \
                self.cel_precip_mean * (
                    1 + self.precip_amplitude * self.precip_variation[
                        (np.ceil(timestep / self.climate_var) % 8)
                        .astype(int)]) \
                - self.veg_rainfall * self.cel_cleared_neighs
        else:
            self.cel_precip = \
                self.cel_precip_mean * (
                    1 - self.veg_rainfall * self.cel_cleared_neighs)

        # check if system time is in drought period
        drought = False
        for drought_time in self.drought_times:
            if drought_time[0] < timestep <= drought_time[1]:
                drought = True

        # if so, decrease precipitation by factor percentage given by
        # drought severity
        if drought:
            self.cel_precip *= (1. - self.drought_severity / 100.)

    def get_waterflow(self):
        """
        Wrapper for the Fortran waterflow-routine: returns water flow
        distribution on cell grid calculated from precipitation and elevation.
        """

        # convert precipitation from mm to meters, because
        # rain_volume and elevation must have same units
        rain_volume = np.nan_to_num(self.cel_precip * 1e-3)
        # call Fortran waterflow routine
        self.cel_waterflow, self.cel_waterlevel = \
            f90routines.f90waterflow(
                self.land_cell_index,
                self.cel_elev,
                rain_volume,
                self.f90neighbourhood,
                self.map_shape[0],
                self.map_shape[1],
                self.n_land_cells)

        # only return waterflow, but keep waterlevel as a state attribute
        return self.cel_waterflow

    def evolve_forest(self, cel_npp: NDArray):
        """
        TODO: add docstring
        """

        # Forest regenerates faster [slower] (linearly),
        # if cell net primary productivity is above [below] average.
        cel_npp_relative = np.nanmean(cel_npp) / cel_npp

        # neighbor kernel
        neigh_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Iterate over all cells repeatedly and regenerate or degenerate
        for _ in range(4):

            # Degeneration

            # Probability is increased if there are settlements around.
            cel_deg_prob = self.natprobdec * (1 + 2 * self.cel_pop_gradient)
            cel_deg_fortune = cel_deg_prob >= rand(*self.map_shape)
            # Mask state 3 candidates and degenerate
            cel_deg_s3 = cel_deg_fortune & (self.cel_forest_state == 3)
            self.cel_forest_state[cel_deg_s3] = 2
            self.cel_forest_memory[cel_deg_s3] = self.state_change_s2
            # Mask state 2 candidates and degenerate
            cel_deg_s2 = cel_deg_fortune & (self.cel_forest_state == 2)
            self.cel_forest_state[cel_deg_s2] = 1
            self.cel_forest_memory[cel_deg_s2] = 0

            # Regeneration

            # Mask state 1 candidates
            cel_reg_s1 = ((self.cel_forest_state == 1)
                          & (self.cel_forest_memory
                             > (self.state_change_s2 * cel_npp_relative)))
            # and regenerate.
            self.cel_forest_state[cel_reg_s1] = 2
            self.cel_forest_memory[cel_reg_s1] = self.state_change_s2
            # Mask state 2 candidates,
            cel_reg_s2 = ((self.cel_forest_state == 2)
                          & (self.cel_forest_memory
                             > (self.state_change_s3 * cel_npp_relative)))
            # check for minimum amount of state 3 neighbors
            cel_s3 = (self.cel_forest_state == 3).astype('int')
            cel_s3_neighbors = (
                ndimage.convolve(cel_s3, neigh_kernel, mode='constant', cval=0)
                > self.min_s3_neighbours)
            # and regenerate.
            self.cel_forest_state[cel_reg_s2 & cel_s3_neighbors] = 3

            # Finally, increase memory of all forest cells by one
            self.cel_forest_memory[*self.land_cell_index] += 1

        # calculate cleared land neighbours for output:
        if self.veg_rainfall > 0:
            cel_s1 = (self.cel_forest_state == 1).astype('int')
            self.cel_cleared_neighs = \
                ndimage.convolve(cel_s1, neigh_kernel, mode='constant', cval=0)

        # make sure all land cells have forest states 1-3
        assert not np.any(self.cel_forest_state[*self.land_cell_index]
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

    def get_ag(self, cel_npp: NDArray, cel_wf: NDArray):
        """
        agricultural productivity is calculated via a linear additive
        model from net primary productivity, soil productivity, slope,
        waterflow and soil degradation of each patch.
        """
        # EQUATION ############################################################
        return (self.a_npp * cel_npp
                + self.a_sp * self.cel_soilprod
                - self.a_s * self.cel_slope
                - self.a_wf * cel_wf
                - self.cel_soil_deg)
        # EQUATION ############################################################

    def get_ecoserv(self, cel_ag: NDArray, cel_wf: NDArray):
        """
        Ecosystem Services are calculated via a linear
        additive model from agricultural productivity (cel_ag),
        waterflow through the cell (cel_wf) and forest
        state on the cell (cel_forest_state) in [1,3],
        The recent version of mayasim limits value of
        ecosystem services to 1 < ecoserv < 250, it also
        proposes to include population density
        (cel_pop_gradient) and precipitation (cel_precip)
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
            self.cel_es_fs = 2. * self.e_ag \
                * (self.cel_forest_state - 1.) * cel_ag
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

    def benefit_cost(self, cel_ag: NDArray):
        # Benefit cost assessment
        return self.max_yield * (1 - self.origin_shift * np.exp(
            np.float128(-self.slope_yield * cel_ag)))

    def update_influenced_cells(self):
        """
        For each settlement, calculate influence radius and write a list of
        cells within that radius to ``self.stm_influenced_cells``.

        NOTE: these are the cells that are closer than population^0.8/60
        (which is not explained any further... change denominator to 80 and
        max value to 30 from eyeballing the results
        """
        # calculate influence radius
        # EQUATION ############################################################
        self.stm_influence_rad = [p**0.8 / 60. for p in self.stm_population]
        self.stm_influence_rad = \
            [rad if rad < 40. else 40. for rad in self.stm_influence_rad]
        # EQUATION ############################################################

        # get cells within influence radius
        for stm, (y, x) in enumerate(self.stm_positions):
            rad = np.ceil(self.stm_influence_rad[stm] / np.sqrt(self.area))
            # create square coordinate-grid of settlement's neighboring cells
            coords = np.mgrid[
                max(y - rad, 0):min(y + rad, self.map_shape[0]),
                max(x - rad, 0):min(x + rad, self.map_shape[1])
            ].reshape(2, -1).astype('int')
            # mask those that are within influence radius
            infd_mask = ((coords[0] - y)**2 + (coords[1] - x)**2) \
                <= (self.stm_influence_rad[stm]**2 / self.area)
            # select coordinates and save to list of tuples
            self.stm_influenced_cells[stm] = list(zip(*coords[:, infd_mask]))

        # count influenced cells
        self.stm_influenced_cells_n = \
            [len(ifd) for ifd in self.stm_influenced_cells]

    # pylint: disable=too-many-locals
    def update_cropped_cells(self, cel_bca: NDArray):
        """
        Update cropped cells for each settlement.
        Cell utilities are calculated depending on cell distance from
        the respective settlement.
        New cells are cropped if population per cropped cell is higher than
        max_people_per_cropped_cell, choosing highest utility cells first.
        Cells are abandoned if population per cropped cell is lower than
        min_people_per_cropped_cell or if cells have negative utility.
        """
        abandoned = 0
        sown = 0

        # number of currently cropped cells for each settlement
        self.stm_cropped_cells_n = \
            [len(crp) for crp in self.stm_cropped_cells]

        # agricultural population density (inhabitants per cropped cell)
        # determines the number of cells that can be cropped.
        ag_pop_density = [
            pop / (cropped_n * self.area) if cropped_n > 0 else 0.

            for pop, cropped_n in zip(
                self.stm_population, self.stm_cropped_cells_n, strict=True)
        ]

        # cel_is_cropped is a mask of all cropped cells of all settlements
        for cel in chain(*self.stm_cropped_cells):
            self.cel_is_cropped[cel] = 1

        # the age of settlements is increased here.
        self.stm_age = [age + 1 for age in self.stm_age]

        # update cropped cells for each settlement
        for stm, (y, x) in enumerate(self.stm_positions):

            # CALCULATE UTILITIES of all influenced cells
            # get influenced cells as array
            infd_index = np.array(self.stm_influenced_cells[stm]).T
            distance = np.sqrt(
                (self.height_cell * (y - infd_index[0]))**2
                + (self.width_cell * (x - infd_index[1]))**2)
            # EQUATION ########################################################
            utility = (
                cel_bca[*infd_index] - self.estab_cost
                - self.ag_travel_cost * distance / np.sqrt(
                    self.stm_population[stm])
            ).tolist()
            # EQUATION ########################################################

            # do rest of operations using tuple-lists and list-comps
            infd_cells = self.stm_influenced_cells[stm]
            available = [self.cel_is_cropped[cel] == 0 for cel in infd_cells]

            # jointly sort utilities, availability and cells such that cells
            # with highest utility are first.
            sorted_utility, sorted_available, sorted_cells = \
                list(zip(*sorted(
                    list(zip(utility, available, infd_cells)), reverse=True)))
            # select utilities and cell coordinates of only available cells
            available_util = list(
                compress(list(sorted_utility), list(sorted_available)))
            available_cells = list(
                compress(list(sorted_cells), list(sorted_available)))

            # 1.) CROP NEW CELLS if population exceeds a threshold
            # -----------------------------------------------------------------

            # calculate number of new cells to crop
            number_of_new_cells = np.floor(
                ag_pop_density[stm] / self.max_people_per_cropped_cell
            ).astype('int')
            # and crop them by selecting cells with positive utility from the
            # beginning of the list
            for n in range(min([number_of_new_cells, len(available_util)])):
                if available_util[n] > 0:
                    self.stm_cropped_cells[stm].append(available_cells[n])
                    self.cel_is_cropped[available_cells[n]] = 1

                    sown += 1

            # END CURRENT LOOP HERE if settlement had no cropped cells before
            if self.stm_cropped_cells_n[stm] == 0:
                continue

            # select utilities of actually cropped cells
            cropped_utils = [
                utility[infd_cells.index(cel)] if cel in infd_cells else -1
                for cel in self.stm_cropped_cells[stm]
            ]

            # sort utilitites and cropped cells to lowest utilities first
            cropped_utils, cropped_cells = zip(*sorted(list(zip(
                cropped_utils, self.stm_cropped_cells[stm]))))

            # 2.) ABANDON CELLS if population too low
            #     and settlement older than > 5 timesteps
            # -----------------------------------------------------------------
            if (ag_pop_density[stm] < self.min_people_per_cropped_cell
                    and self.stm_age[stm] > 5):
                # number of cells to abandon, not further explained, see #6
                lost_cells_n = np.ceil(30 / ag_pop_density[stm]).astype('int')
                for n in range(min([lost_cells_n,
                                    self.stm_cropped_cells_n[stm]])):
                    self.stm_cropped_cells[stm].remove(cropped_cells[n])
                    self.cel_is_cropped[cropped_cells[n]] = 0

                    abandoned += 1

            # 3.) ABANDON CELLS with negative utility
            # -----------------------------------------------------------------
            # find cells that have negative utility and belong to current stm
            useless_cropped_cells = [
                cropped_cells[cel] for cel in range(len(cropped_cells))

                if cropped_utils[cel] < 0
                and cropped_cells[cel] in self.stm_cropped_cells[stm]
            ]
            # and release them.
            for cel in useless_cropped_cells:
                try:
                    self.stm_cropped_cells[stm].remove(cel)
                except ValueError:
                    print('ERROR: Useless cell gone already')
                self.cel_is_cropped[cel] = 0

                abandoned += 1

        # Finally, update number of cropped cells for all settlements
        self.stm_cropped_cells_n = [
            len(crp) for crp in self.stm_cropped_cells]

        return abandoned, sown

    def get_pop_mig(self):
        # gives population and out-migration
        # print("number of settlements", len(self.stm_population))

        # death rate correlates inversely with real income per capita
        death_rate_diff = self.max_death_rate - self.min_death_rate

        self.stm_death_rate = [
            -death_rate_diff * income + self.max_death_rate
            for income in self.stm_real_income_pc
        ]

        self.stm_death_rate = list(np.clip(
            self.stm_death_rate, self.min_death_rate, self.max_death_rate))

        # if population control,
        # birth rate negatively correlates with population size

        if self.population_control:
            birth_rate_diff = self.max_birth_rate - self.min_birth_rate

            self.stm_birth_rate = [
                - birth_rate_diff / 10000. * pop
                + self.shift if pop > 5000 else self.birth_rate_parameter

                for pop in self.stm_population
            ]

        # population grows according to effective growth rate
        self.stm_population = [
            int((1. + b_rate - d_rate) * pop)

            for b_rate, d_rate, pop in zip(
                self.stm_birth_rate, self.stm_death_rate, self.stm_population,
                strict=True)
        ]

        self.stm_population = [
            pop if pop > 0 else 0 for pop in self.stm_population
        ]

        mig_rate_diffe = self.max_mig_rate - self.min_mig_rate

        # outmigration rate also correlates
        # inversely with real income per capita
        self.stm_mig_rate = [
            -mig_rate_diffe * income + self.max_mig_rate
            for income in self.stm_real_income_pc
        ]

        self.stm_mig_rate = list(
            np.clip(self.stm_mig_rate, self.min_mig_rate, self.max_mig_rate))

        self.stm_out_mig = [
            int(m_rate * pop)
            for m_rate, pop in zip(
                self.stm_mig_rate, self.stm_population, strict=True)
        ]

        self.stm_out_mig = [
            value if value > 0 else 0 for value in self.stm_out_mig]

    # impact of sociosphere on ecosphere
    def update_pop_gradient(self):
        # pop gradient quantifies the disturbance of the forest by population
        self.cel_pop_gradient = np.zeros(self.map_shape)

        for stm, (y, x) in enumerate(self.stm_positions):

            infd_index = np.array(self.stm_influenced_cells[stm]).T
            distance = np.sqrt(self.area * (
                (y - infd_index[0])**2 + (x - infd_index[1])**2))

            # EQUATION ########################################################
            self.cel_pop_gradient[*infd_index] += \
                self.stm_population[stm] / (300 * (1 + distance))
            # EQUATION ########################################################
            self.cel_pop_gradient[self.cel_pop_gradient > 15] = 15

    def evolve_soil_deg(self):
        # soil degrades for cropped cells
        for cel in chain(*self.stm_cropped_cells):
            self.cel_soil_deg[cel] += self.deg_rate
        # soil regenerates for climax-forest cells
        self.cel_soil_deg[self.cel_forest_state == 3] -= self.reg_rate
        self.cel_soil_deg[self.cel_soil_deg < 0] = 0

    def get_rank(self):
        # depending on population ranks are assigned
        # attention: ranks are reverted with respect to Netlogo MayaSim !
        # 1 -> 3 ; 2 -> 2 ; 3 -> 1
        self.stm_rank = [
            3 if pop > self.thresh_rank_3
            else 2 if pop > self.thresh_rank_2
            else 1 if pop > self.thresh_rank_1
            else 0

            for pop in self.stm_population
        ]

    @property
    def build_routes(self):

        adj = self.stm_adjacency.copy()
        adj[adj == -1] = 0
        built_links = 0
        lost_links = 0
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph())
        self.stm_degree = graph.out_degree()

        # create index-array of stm positions to vectorize distance calculation
        positions = np.array(self.stm_positions).T

        for stm, (y, x) in enumerate(self.stm_positions):
            # stm with rank > 0 are traders and establish links to neighbours
            # NOTE: only if they do not already have 'rank' trading partners?
            if self.stm_degree[stm] < self.stm_rank[stm]:
                # calculate maximum trading radius depending on trade rank
                trade_radius = 31. * (
                    getattr(self, f'thresh_rank_{self.stm_rank[stm]}', 0)
                    / self.thresh_rank_3 * 0.5 + 1.)

                # calculate distances to other settlements
                distances = (y - positions[0]) ** 2 + (x - positions[1]) ** 2

                # collect neighbors within radius, but not self
                # and omit those that are already connected.
                candidates = (
                    (0.5 < (distances <= trade_radius**2 / self.area))
                    & (self.stm_adjacency[stm] == 0)
                )

                # if trading candidates found, connect to largest
                if sum(candidates) != 0:
                    new_partner = np.argmax(self.stm_population * candidates)
                    self.stm_adjacency[stm, new_partner] = 1
                    self.stm_adjacency[new_partner, stm] = -1
                    built_links += 1

            # settlements that cannot maintain their trade links lose them:
            elif self.stm_degree[stm] > self.stm_rank[stm]:
                # get neighbors of node
                neighbors = graph.successors(stm)
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
            tmp_comp_size, tmp_degree = f90routines.f90sparsecomponents(
                i_c, a, j_a, self.n_settlements, l_ic, l_a)
            self.stm_comp_size = list(tmp_comp_size)
            self.stm_degree = list(tmp_degree)
        elif l_a == 0:
            self.stm_comp_size = [0] * (l_ic - 1)
            self.stm_degree = [0] * (l_ic - 1)

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
            tmp_centrality = f90routines.f90sparsecentrality(
                i_c, a, j_a, self.n_settlements, l_ic, l_a)
            self.stm_centrality = list(tmp_centrality)
        elif l_a == 0:
            self.stm_centrality = [1] * (l_ic - 1)

    def get_crop_income(self, cel_bca: NDArray):
        # agricultural benefit of cropping
        for stm, crpd_cells in enumerate(self.stm_cropped_cells):
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

    def get_eco_income(self, cel_es: NDArray):
        # benefit from ecosystem services of cells in influence
        # ##EQUATION###########################################################
        if self.eco_income_mode == "mean":
            for stm, infd_cells in enumerate(self.stm_influenced_cells):
                infd_index = np.array(infd_cells).T
                self.stm_eco_benefit[stm] = \
                    self.r_es_mean * np.nanmean(cel_es[*infd_index])

        elif self.eco_income_mode == "sum":
            for stm, infd_cells in enumerate(self.stm_influenced_cells):
                infd_index = np.array(infd_cells).T
                self.stm_eco_benefit[stm] = \
                    self.r_es_sum * np.nansum(cel_es[*infd_index])
        # ##EQUATION###########################################################

    def get_trade_income(self):
        # ##EQUATION###########################################################
        self.stm_trade_income = [
            1. / 30. * (1 + comp_size / centrality)**0.9

            for comp_size, centrality in zip(
                self.stm_comp_size, self.stm_centrality, strict=True)
        ]

        self.stm_trade_income = [
            self.r_trade if trade_income > 1
            else 0 if (trade_income < 0 or degree == 0)
            else self.r_trade * trade_income

            for trade_income, degree in zip(
                self.stm_trade_income, self.stm_degree, strict=True)
        ]
        # ##EQUATION###########################################################

    def get_real_income_pc(self):
        # combine agricultural, ecosystem service and trade benefit
        # EQUATION #
        self.stm_real_income_pc = [
            (crop_yield + eco_benefit + trade_income) / pop
            if pop > 0 else 0

            for crop_yield, eco_benefit, trade_income, pop in zip(
                self.stm_crop_yield, self.stm_eco_benefit,
                self.stm_trade_income, self.stm_population,
                strict=True)
        ]

    def migration(self, cel_es):
        # if outmigration rate exceeds threshold, found new settlement
        self.stm_migrants = [0] * self.n_settlements
        new_settlements = 0

        # create mask of land cells that are not influenced by any settlement
        uninfluenced = np.isfinite(cel_es)
        for cel in chain(*self.stm_influenced_cells):
            uninfluenced[cel] = 0
        # create index-array of uninfluenced cells
        uninfd_index = np.asarray(np.where(uninfluenced == 1))
        # create index-array of stm positions to vectorize distance calculation
        positions = np.array(self.stm_positions).T

        for stm, (y, x) in enumerate(self.stm_positions):
            if (self.stm_out_mig[stm] > 400 and len(uninfd_index[0]) > 0
                    and np.random.rand() <= 0.5):

                mig_pop = self.stm_out_mig[stm]
                self.stm_migrants[stm] = mig_pop
                self.stm_population[stm] -= mig_pop
                pioneer_set = \
                    uninfd_index[:, np.random.choice(len(uninfd_index[0]), 75)]

                # calculate pioneer utilities from travel cost and eco-services
                pio_tc = np.sqrt(self.area * (
                    (y - pioneer_set[0])**2 + (x - pioneer_set[1])**2))
                pio_es = cel_es[pioneer_set[0], pioneer_set[1]]
                pio_ut = self.mig_ES_pref * pio_es + self.mig_TC_pref * pio_tc
                # choose pioneer with highest utility as new location
                new_y, new_x = pioneer_set[:, np.argmax(pio_ut)]

                # check if other settlements are near new location
                neighbours = (
                    (new_y - positions[0])**2 + (new_x - positions[1])**2
                ) <= 7.5**2 / self.area
                # if not, spawn settlement
                if np.sum(neighbours) == 0:
                    self.spawn_settlement(new_y, new_x, mig_pop)
                    new_settlements += 1

        return new_settlements

    def kill_settlements(self):
        '''TODO: add docstring'''

        killed_stm = 0
        # kill settlements if they have too few inhabitants
        dead_stm_ind = [
            stm for stm in range(self.n_settlements)

            if self.stm_population[stm] <= self.min_stm_size
        ]
        # optional: kill settlements if they have no crops
        if self.kill_stm_without_crops:
            dead_stm_ind += [
                stm for stm in range(self.n_settlements)

                if not self.stm_cropped_cells[stm]
            ]

        # only keep unique entries
        dead_stm_ind = list(set(dead_stm_ind))

        # remove settlement attributes from attribute-lists
        for dead_stm in sorted(dead_stm_ind, reverse=True):
            self.n_settlements -= 1
            self.n_failed_stm += 1
            del self.stm_positions[dead_stm]
            del self.stm_age[dead_stm]
            del self.stm_birth_rate[dead_stm]
            del self.stm_death_rate[dead_stm]
            del self.stm_population[dead_stm]
            del self.stm_mig_rate[dead_stm]
            del self.stm_out_mig[dead_stm]
            del self.stm_influenced_cells_n[dead_stm]
            del self.stm_influence_rad[dead_stm]
            del self.stm_cropped_cells_n[dead_stm]
            del self.stm_crop_yield[dead_stm]
            del self.stm_eco_benefit[dead_stm]
            del self.stm_rank[dead_stm]
            del self.stm_degree[dead_stm]
            del self.stm_comp_size[dead_stm]
            del self.stm_centrality[dead_stm]
            del self.stm_trade_income[dead_stm]
            del self.stm_real_income_pc[dead_stm]
            del self.stm_influenced_cells[dead_stm]
            del self.stm_cropped_cells[dead_stm]
            del self.stm_migrants[dead_stm]

            killed_stm += 1

        # special cases:
        self.stm_adjacency = \
            np.delete(np.delete(self.stm_adjacency, dead_stm_ind, axis=0),
                      dead_stm_ind, axis=1)

        return killed_stm

    def spawn_settlement(self, y: int, x: int, mig_pop: int):
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
        self.stm_positions.append((y, x))
        self.stm_influenced_cells.append([(y, x)])
        self.stm_cropped_cells.append([(y, x)])

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
        self.stm_influence_rad.append(0)
        self.stm_cropped_cells_n.append(1)
        self.stm_crop_yield.append(0)
        self.stm_eco_benefit.append(0)
        self.stm_rank.append(0)
        self.stm_degree.append(0)
        self.stm_trade_income.append(0)
        self.stm_real_income_pc.append(0)
        self.stm_migrants.append(0)

    def run(self, steps: int = 1):
        """
        Run the model for a given number of steps.
        If no number of steps is given, the model is integrated for one step

        Parameters
        ----------
        steps: int
            number of steps to integrate the model
        """

        self.init_output()

        # initialize progress bar
        t_range = trange(1, steps + 1,
                         desc='running MayaSim',
                         postfix={'population': sum(self.stm_population)})

        for timestep in t_range:
            # evolve subselfs

            # ecosystem
            self.update_precipitation(timestep)
            # net primary productivity
            cel_npp = self.get_npp()
            self.evolve_forest(cel_npp)

            # water flow
            cel_wf = self.get_waterflow()
            # agricultural productivity
            cel_ag = self.get_ag(cel_npp, cel_wf)
            # ecosystem services
            cel_es = self.get_ecoserv(cel_ag, cel_wf)
            # benefit cost map for agriculture
            cel_bca = self.benefit_cost(cel_ag)

            # society
            if len(self.stm_population) > 0:
                # ag income
                self.update_influenced_cells()
                abandoned, sown = self.update_cropped_cells(cel_bca)
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

            self.step_output(
                timestep, cel_npp, cel_wf, cel_ag, cel_es, cel_bca, abandoned,
                sown, built, lost, new_settlements, killed_settlements)

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
            settlement_init_data = {'shape': self.map_shape}
            with open(self.settlement_output_path(0), 'wb') as f:
                pkl.dump(settlement_init_data, f)

        if self.output_geographic_data:
            pass

    def step_output(
            self, timestep: int,
            cel_npp: NDArray, cel_wf: NDArray,
            cel_ag: NDArray, cel_es: NDArray, cel_bca: NDArray,
            abandoned: int, sown: int, built: int, lost: int,
            new_settlements: int, killed_settlements: int):
        """
        call different data saving routines depending on settings.

        Parameters
        ----------
        timestep: int
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
                timestep, [cel_npp, cel_wf, cel_ag, cel_es, cel_bca],
                built, lost, new_settlements, killed_settlements
            )
            self.update_traders_aggregates(timestep)

        # save maps of spatial data

        if self.output_geographic_data:
            self.save_geographic_output(timestep, cel_wf, abandoned, sown)

        # save data on settlement basis

        if self.output_settlement_data:
            self.save_settlement_output(timestep)

    def save_settlement_output(self, timestep: int):
        """
        Organize settlement based data in Pandas Dataframe
        and save to file.

        Parameters
        ----------
        timestep: int
            Timestep number to append to save file path

        """
        columns = [
            'population', 'real income', 'ag income', 'es income',
            'trade income', 'position', 'out migration',
            'degree'
        ]
        data = [
            self.stm_population, self.stm_real_income_pc, self.stm_crop_yield,
            self.stm_eco_benefit, self.stm_trade_income, self.stm_positions,
            self.stm_migrants, self.stm_degree
        ]

        data = list(map(list, zip(*data)))

        data_frame = pandas.DataFrame(columns=columns, data=data)

        with open(self.settlement_output_path(timestep), 'wb') as f:
            pkl.dump(data_frame, f)

    def save_geographic_output(
            self, timestep: int, cel_wf: NDArray, abandoned: int, sown: int):
        """
        Organize Geographic data in dictionary (for separate layers
        of data) and save to file.

        Parameters
        ----------
        timestep: int
            Timestep number to append to save file path
        cel_wf: numpy array
            Water flow through cell
        abandoned: int
            Number of cells that was abandoned in the previous time step
        sown: int
            Number of cells that was newly cropped in the previous time step
        """

        data = {
            'forest': self.cel_forest_state,
            'waterflow': cel_wf,
            'cells in influence': self.stm_influenced_cells,
            'number of cells in influence': self.stm_influenced_cells_n,
            'cropped cells': self.stm_cropped_cells,
            'number of cropped cells': self.stm_cropped_cells_n,
            'abandoned sown': np.array([abandoned, sown]),
            'soil degradation': self.cel_soil_deg,
            'population gradient': self.cel_pop_gradient,
            'adjacency': self.stm_adjacency,
            'position': self.stm_positions,
            'population': self.stm_population,
            'elev': self.cel_elev,
            'rank': self.stm_rank
        }

        with open(self.geographic_output_path(timestep), 'wb') as f:
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
            'forest_state_1_cells', 'MAP', 'max_npp', 'mean_waterflow',
            'max_AG', 'max_ES', 'max_bca', 'max_soil_deg', 'max_pop_grad'
        ])

    def init_traders_aggregates(self):
        self.traders_aggregates.append([
            'time', 'total_population', 'total_migrants', 'total_traders',
            'total_settlements', 'total_cropped_cells',
            'total_influenced_cells', 'total_trade_links',
            'total_income_agriculture', 'total_income_ecosystem',
            'total_income_trade'
        ])

    def update_aggregates(
            self, time: int, args: list[NDArray], built: int, lost: int,
            new_settlements: int, killed_settlements: int):
        # args = [cel_npp, cel_wf, cel_ag, cel_es, cel_bca]

        total_population = sum(self.stm_population)
        max_population = max(self.stm_population)
        total_migrants = sum(self.stm_migrants)
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
        except:  # pylint: disable=bare-except
            max_cluster_size = 0
        self.max_cluster_size = max_cluster_size
        total_cropped_cells = sum(self.stm_cropped_cells_n)
        total_influenced_cells = sum(self.stm_influenced_cells_n)

        self.aggregates.append([
            time, total_population, max_population, total_migrants,
            total_settlements, total_cropped_cells,
            total_influenced_cells, total_trade_links, mean_cluster_size,
            max_cluster_size, new_settlements, killed_settlements, built, lost,
            income_agriculture, income_ecosystem, income_trade,
            np.nanmean(self.cel_soil_deg),
            np.sum(self.cel_forest_state == 3),
            np.sum(self.cel_forest_state == 2),
            np.sum(self.cel_forest_state == 1),
            np.nanmean(self.cel_precip),
            np.nanmax(args[0]),
            np.nanmean(args[1]),
            np.nanmax(args[2]),
            np.nanmax(args[3]),
            np.nanmax(args[4]),
            np.nanmax(self.cel_soil_deg),
            np.nanmax(self.cel_pop_gradient)
        ])

    def update_traders_aggregates(self, time: int):

        traders = np.where(np.array(self.stm_degree) > 0)[0]

        total_population = sum(self.stm_population[stm] for stm in traders)
        total_migrants = sum(self.stm_migrants[stm] for stm in traders)
        total_settlements = len(self.stm_population)
        total_traders = len(traders)
        total_trade_links = sum(self.stm_degree) / 2
        income_agriculture = sum(self.stm_crop_yield[stm] for stm in traders)
        income_ecosystem = sum(self.stm_eco_benefit[stm] for stm in traders)
        income_trade = sum(self.stm_trade_income[stm] for stm in traders)

        total_cropped_cells = \
            sum(self.stm_cropped_cells_n[stm] for stm in traders)
        total_influenced_cells = \
            sum(self.stm_influenced_cells_n[stm] for stm in traders)

        self.traders_aggregates.append([
            time, total_population, total_migrants, total_traders,
            total_settlements, total_cropped_cells,
            total_influenced_cells, total_trade_links, income_agriculture,
            income_ecosystem, income_trade
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
