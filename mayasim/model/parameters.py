import numpy as np


class Parameters:
    # default parameters of NetLogo MayaSim v1.3.0 are noted as comments

    # *************************************************************************
    # RUN PARAMETERS:

    # INITIAL NUMBER OF INHABITANTS PER SETTLEMENT
    min_init_inhabitants = 1000     # 1000
    max_init_inhabitants = 2000     # 2000

    # *************************************************************************
    # MODEL PARAMETERS:
    # *************************************************************************

    # *************************************************************************
    # ECOSYSTEM

    # RAINFALL PARAMETERS

    veg_rainfall = 0.               # 0. (i.e. disabled)
    precip_modulation = False       # True
    # multiplier for precipitation according to climate cycle
    precip_variation = np.array([-0.06, -0.12, -0.18,
                                        -0.12, -0.06, 0., 0.06, 0.])
    precip_amplitude = 1.           # 1. (analogous)
    # length of period with constant rainfall
    climate_var = 3                 # 3

    # DROUGHT PARAMETERS

    # timing of drought events as list of list of
    # start and end dates
    drought_times = []  # e.g. [[0, 2], [150, 170]]
    # percentage of rainfall decrease
    drought_severity = 10

    # FOREST PARAMETERS

    # rate of natural forest depreciation per unit time
    natprobdec = 0.003      # 0.003
    # Threshold for forest regeneration 1 -> 2
    state_change_s2 = 40.   # 40.
    # Threshold for forest regeneration 2 -> 3
    state_change_s3 = 100.  # 100.
    # number of state 3 neighbors required for 2 -> 3 regeneration
    min_s3_neighbours = 2   # 2

    # *************************************************************************
    # AGRICULTURE

    # WEIGHTS FOR AGRICULTURAL PRODUCTIVITY
    a_npp = 0.14  # 0.14 - net primary productivity
    a_sp = 84.    # 84.  - soil productivity
    a_s = 18.     # 18.  - slope
    a_wf = 4.     # 400. - water flow, NOTE: different units?

    # WEIGHTS FOR BENEFIT COST ASSESSMENT TO CROP NEW CELL
    max_yield = 1100        # 1100
    origin_shift = 1.11     # 1.11
    slope_yield = 0.0052    # 0.0052

    # PARAMETERS FOR CROPPING CELLS
    estab_cost = 900                    # 900.
    ag_travel_cost = 950                # 950.
    min_people_per_cropped_cell = 40.   # 40.
    max_people_per_cropped_cell = 125.  # 125.

    # KILL SETTLEMENTS WITHOUT POPULATION ONLY OR ALSO WITHOUT CROPS
    kill_stm_without_crops = False      # True

    # PARAMETERS FOR SOIL DEGRADATION:
    deg_rate = 5.0  # 5.0 - degradation rate for cropped cells
    reg_rate = 2.5  # 2.5 - regeneration rate for state 3 forest cells

    # WEIGHTS FOR ECOSYSTEM SERVICES
    e_ag = 0.05  # 0.06 - agricultural productivity
    e_wf = 1.    # 40.  - water flow, NOTE: different units?
    e_f = 45.    # 45.  - forest
    e_r = 0.     # 0.   - (i.e. excluded) rainfall
    e_deg = 0.   # 0.   - (i.e. excluded) population density

    # SWITCH FOR ECOSYSTEM SERVICES CALCULATION
    better_ess = True    # False

    # *************************************************************************
    # SOCIO-ECONOMY

    # specifiy the way, ag and es income is calculated. ('mean' or 'sum')
    crop_income_mode = 'sum'    # 'mean'
    eco_income_mode = 'sum'     # 'mean'

    # WEIGHTS FOR INCOME CALCULATION
    # 'sum' weights have been calibrated from parameter scans to yield
    # realistic peak population levels.
    r_bca_mean = 1.1  # 1.1   - weight ag income for mean calculation
    r_bca_sum = 0.25  # --    - weight ag income for sum calculation
    r_es_mean = 10.   # 10.   - weight es income for mean calculation
    r_es_sum = .05    # --    - weight es income for sum calculation
    r_trade = 6000.   # 6000. - weight trade income

    # DEMOGRAPHIC PARAMETERS
    birth_rate_parameter = 0.15 # 0.15

    min_stm_size = 360          # 360 = estab_cost * 0.4

    # optionally make birth rate scale inversely with settlement population
    population_control = False
    max_birth_rate = 0.15       # 0.15
    min_birth_rate = 0.         # -0.2
    shift = 0.325               # 0.325

    # death rate correlates inversely with real income p.c.
    min_death_rate = 0.005      # 0.005
    max_death_rate = 0.25       # 0.25

    # MIGRATION PREFERENCE PARAMETERS
    mig_TC_pref = -0.1          # -0.1
    mig_ES_pref = 0.3           # 0.3

    min_mig_rate = 0.           # 0.
    max_mig_rate = 0.15         # 0.15

    # POPULATION THRESHOLDS FOR TRADER RANKS
    thresh_rank_1 = 4000        # 4000.
    thresh_rank_2 = 7000        # 7000.
    thresh_rank_3 = 9500        # 9500.
