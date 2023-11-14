import numpy as np

class Parameters:

    # *************************************************************************
    # RUN PARAMETERS:

    # OUTPUT LEVEL
    output_level = 'spatial'   # either 'spatial' or 'trajectory'

    # INITIAL NUMBER OF INHABITANTS PER SETTLEMENT
    min_init_inhabitants = 1000     # 1000
    max_init_inhabitants = 2000     # 2000

    timesteps = 325			# 325

    # *************************************************************************
    # MODEL PARAMETERS:
    # *************************************************************************

    # *************************************************************************
    # ECOSYSTEM

    # RAINFALL PARAMETERS

    veg_rainfall = 0.
    precipitation_modulation = True
    # multiplier for precipitation according to climate cycle
    precipitation_variation = np.array([-0.06, -0.12, -0.18,
                                        -0.12, -0.06, 0., 0.06, 0.])
    precipitation_amplitude = 1.
    # length of period with constant rainfall
    climate_var = 3

    # DROUGHT PARAMETERS

    # timing of drought events as list of list of
    # start and end dates
    drought_times = [[0, 2], [150, 170]]
    # decrease in rainfall in percent
    drought_severity = 10

    # FOREST PARAMETERS

    # rate of natural forest depreciation per unit time
    natprobdec = 0.003
    # Threshold for forest regeneration 1 -> 2
    state_change_s2 = 40.
    # Threshold for forest regeneration 2 -> 3
    state_change_s3 = 100.
    # number of state 3 neighbors required for 2 -> 3 regeneration
    min_number_of_s3_neighbours = 2

    # *************************************************************************
    # AGRICULTURE

    # WEIGHTS FOR AGRICULTURAL PRODUCTIVITY
    a_npp = 0.14  # weight for net primary productivity
    a_sp = 84.    # for soil productivity
    a_s = 18.     # for slope
    a_wf = 400.   # water flow

    # WEIGHTS FOR BENEFIT COST ASSESSMENT TO CROP NEW CELL
    max_yield = 1100
    origin_shift = 1.11
    slope_yield = 0.0052

    # PARAMETERS FOR CROPPING CELLS
    estab_cost = 900
    ag_travel_cost = 950
    min_people_per_cropped_cell = 40.
    max_people_per_cropped_cell = 125.

    # KILL CITIES WITHOUT POPULATION ONLY OR ALSO WITHOUT CROPS
    kill_cities_without_crops = False

    # PARAMETERS FOR SOIL DEGRADATION:
    deg_rate = 5.0  # 5.0 - degradation rate for cropped cells
    reg_rate = 2.5  # 2.5 - regeneration rate for state 3 forest cells

    # WEIGHTS FOR ECOSYSTEM SERVICES
    e_ag = 0.05  # weight for agricultural productivity
    e_wf = 100.   # water flow
    e_f = 45.    # forest
    e_r = 0.     # rainfall (not included in current netlogo version)
    e_deg = 0.   # population density (not included in current netlogo version)

    # SWITCH FOR ECOSYSTEM SERVICES CALCULATION
    better_ess = False

    # *************************************************************************
    # SOCIO-ECONOMY

    # crop_income specifies the way, crop income is calculated.
    # 'mean' or 'sum', Heckbert: 'mean'
    crop_income_mode = 'sum'
    eco_income_mode = 'sum'

    # WEIGHTS FOR INCOME CALCULATION
    # Have been calibrated from parameter scans to yield
    # realistic peak population levels.
    r_bca_mean = 1.1  # 1.1 - weight agriculture income for mean calculation
    r_bca_sum = 0.25  # 0.25 - weight agr income for sum calculation
    r_es_mean = 10.   # 10. - weight ecosystem services for mean calculation
    r_es_sum = .0001  # 0.0002  - weight ecosys services for sum calculation
    r_trade = 6000.   # 6000. weight trade income

    # DEMOGRAPHIC PARAMETERS
    birth_rate_parameter = 0.15

    min_city_size = 360

    # optionally make birth rate scale inversely with p.c. income
    population_control = False
    max_birth_rate = 0.15
    min_birth_rate = 0.
    shift = 0.325

    # death rate correlates inversely with real income p.c.
    min_death_rate = 0.005
    max_death_rate = 0.25

    # MIGRATION PREFERENCE PARAMETERS
    mig_TC_pref = -0.1
    mig_ES_pref =  0.3

    min_mig_rate = 0.
    max_mig_rate = 0.15

    # POPULATION THRESHOLDS FOR TRADER RANKS
    thresh_rank_1 = 4000
    thresh_rank_2 = 7000
    thresh_rank_3 = 9500
