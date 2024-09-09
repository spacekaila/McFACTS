#!/usr/bin/env python3
import os
from os.path import isfile, isdir
from pathlib import Path

import numpy as np
import scipy.interpolate

from mcfacts.inputs import ReadInputs
from importlib import resources as impresources
from mcfacts.inputs import data as input_data
from mcfacts.mcfacts_random_state import reset_random, rng
from mcfacts.objects.agnobject import AGNBlackHole, AGNFilingCabinet

from mcfacts.setup import setupdiskblackholes, initializediskstars
from mcfacts.physics.migration.type1 import type1
from mcfacts.physics.accretion.eddington import changebhmass, changestarsmass
from mcfacts.physics.accretion.torque import changebh, changestars
from mcfacts.physics.feedback.hankla21 import feedback_hankla21, feedback_hankla21_stars
from mcfacts.physics.dynamics import dynamics
from mcfacts.physics.eccentricity import orbital_ecc
from mcfacts.physics.binary.formation import hillsphere, add_new_binary
from mcfacts.physics.binary.evolve import evolve
from mcfacts.physics.binary.harden import baruteau11
from mcfacts.physics.binary.merge import tichy08, chieff, tgw
from mcfacts.physics.disk_capture import crude_retro_evol
from mcfacts.outputs import mergerfile

binary_field_names = "R1 R2 M1 M2 a1 a2 theta1 theta2 sep com t_gw merger_flag t_mgr  gen_1 gen_2  bin_ang_mom bin_ecc bin_incl bin_orb_ecc nu_gw h_bin"
binary_stars_field_names = "R1 R2 M1 M2 R1_star R2_star a1 a2 theta1 theta2 sep com t_gw merger_flag t_mgr  gen_1 gen_2  bin_ang_mom bin_ecc bin_incl bin_orb_ecc nu_gw h_bin"
merger_field_names = ' '.join(mergerfile.names_rec)

# Do not change this line EVER
DEFAULT_INI = impresources.files(input_data) / "model_choice.ini"
bh_initial_field_names = "disk_location mass spin spin_angle orb_ang_mom orb_ecc orb_incl"
# Feature in testing do not use unless you know what you're doing.

assert DEFAULT_INI.is_file()

FORBIDDEN_ARGS = [
    "disk_outer_radius",
    "max_disk_radius_pc",
    "disk_inner_radius",
    ]


def arg():
    import argparse
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--bin_num_max", default=1000, type=int)
    parser.add_argument("--fname-ini", help="Filename of configuration file",
                        default=DEFAULT_INI, type=str)
    parser.add_argument("--fname-output-mergers", default="output_mergers.dat",
                        help="output merger file (if any)", type=str)
    parser.add_argument("--fname-snapshots-bh",
                        default="output_bh_[single|binary]_$(index).dat",
                        help="output of BH index file ")
    parser.add_argument("--no-snapshots", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-w", "--work-directory",
                        default=Path().parent.resolve(),
                        help="Set the working directory for saving output. Default: current working directory",
                        type=str
                        )
    parser.add_argument("--seed", type=int, default=None,
                        help="Set the random seed. Randomly sets one if not passed. Default: None")
    parser.add_argument("--fname-log", default="mcfacts_sim.log", type=str,
                        help="Specify a file to save the arguments for mcfacts")

    # Add inifile arguments
    # Read default inifile
    _variable_inputs = ReadInputs.ReadInputs_ini(DEFAULT_INI, False)
    # Loop the arguments
    for name in _variable_inputs:
        # Skip CL read of forbidden arguments
        if name in FORBIDDEN_ARGS:
            continue
        _metavar = name
        _opt = "--%s" % (name)
        _default = _variable_inputs[name]
        _dtype = type(_variable_inputs[name])
        parser.add_argument(_opt,
                            default=_default,
                            type=_dtype,
                            metavar=_metavar,
                            )

    # Parse arguments
    opts = parser.parse_args()
    # Check that the inifile exists
    assert isfile(opts.fname_ini)
    # Convert to path objects
    opts.fname_ini = Path(opts.fname_ini)
    assert opts.fname_ini.is_file()
    opts.fname_snapshots_bh = Path(opts.fname_snapshots_bh)
    opts.fname_output_mergers = Path(opts.fname_output_mergers)

    # Parse inifile
    print("opts.fname_ini", opts.fname_ini)
    # Read inifile
    variable_inputs = ReadInputs.ReadInputs_ini(opts.fname_ini, opts.verbose)
    print("variable_inputs", variable_inputs)
    # Hidden variable inputs
    print("_variable_inputs", _variable_inputs)
    # raise Exception
    # Okay, this is important. The priority of input arguments is:
    # command line > specified inifile > default inifile
    for name in variable_inputs:
        # Check for args not in parser. These were generated or changed in ReadInputs.py
        if not hasattr(opts, name):
            setattr(opts, name, variable_inputs[name])
            continue
        # Check for args not in the default_ini file
        if getattr(opts, name) != _variable_inputs[name]:
            # This is the case where the user has set the value of an argument
            # from the command line. We don't want to argue with the user.
            pass
        else:
            # This is the case where the user has not set the value of an
            # argument from the command line.
            # We can overwrite the default value with the inifile value
            setattr(opts, name, variable_inputs[name])
    # Case 3: if an attribute is in the default infile,
    #   and not the specified inifile,
    #   it remains unaltered.

    if opts.verbose:
        for item in opts.__dict__:
            print(item, getattr(opts, item))
    print("variable_inputs", variable_inputs)
    # raise Exception
    # Get the user-defined or default working directory / output location
    opts.work_directory = Path(opts.work_directory).resolve()
    if not isdir(opts.work_directory):
        os.mkdir(opts.work_directory)
    assert opts.work_directory.is_dir()
    try:  # check if working directory for output exists
        os.stat(opts.work_directory)
    except FileNotFoundError as e:
        raise e
    print(f"Output will be saved trno {opts.work_directory}")

    # Get the parent path to this file and cd to that location for runtime
    opts.runtime_directory = Path(__file__).parent.resolve()
    assert opts.runtime_directory.is_dir()
    os.chdir(opts.runtime_directory)

    # set the seed for random number generation and reproducibility if not user-defined
    if opts.seed is None:
        opts.seed = np.random.randint(low=0, high=int(1e18), dtype=np.int_)
        print(f'Random number generator seed set to: {opts.seed}')

    # Write parameters to log file
    with open(opts.work_directory / opts.fname_log, 'w') as F:
        for item in opts.__dict__:
            line = "%s = %s\n" % (item, str(opts.__dict__[item]))
            F.write(line)
    return opts


def main():
    """
    """
    # Setting up automated input parameters
    # see IOdocumentation.txt for documentation of variable names/types/etc.
    opts = arg()
    # Disk surface density (in kg/m^2) is a function of radius, where radius is in r_g
    # Disk aspect ratio is also a function of radius, where radius is in r_g
    disk_surface_density, disk_aspect_ratio = \
        ReadInputs.construct_disk_interp(opts.smbh_mass,
                                         opts.disk_radius_outer,
                                         opts.disk_model_name,
                                         opts.disk_alpha_viscosity,
                                         opts.disk_bh_eddington_ratio,
                                         disk_radius_max_pc=opts.disk_radius_max_pc,
                                         flag_use_pagn=opts.flag_use_pagn,
                                         verbose=opts.verbose
                                         )

    merged_bh_array_pop = []
    surviving_bh_array_pop = []
    emris_array_pop = []

    gw_array_pop = []
    print("opts.__dict__", opts.__dict__)
    print("opts.smbh_mass", opts.smbh_mass)
    print("opts.fraction_bin_retro", opts.fraction_bin_retro)

    for iteration in range(opts.iteration_num):
        print("Iteration", iteration)
        # Set random number generator for this run with incremented seed
        reset_random(opts.seed+iteration)

        # Make subdirectories for each iteration
        # Fills run number with leading zeros to stay sequential
        iteration_zfilled_str = f"{iteration:>0{int(np.log10(opts.iteration_num))+1}}"
        try:  # Make subdir, exit if it exists to avoid clobbering.
            os.makedirs(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}"), exist_ok=False)
        except FileExistsError:
            raise FileExistsError(f"Directory \'run{iteration_zfilled_str}\' exists. Exiting so I don't delete your data.")

        # can index other parameter lists here if needed.
        # galaxy_type = galaxy_models[iteration] # e.g. star forming/spiral vs. elliptical
        # NSC mass
        # SMBH mass
        # Housekeeping for array initialization
        temp_emri_array = np.zeros(7)
        emri_array = np.zeros(7)
        temp_bbh_gw_array = np.zeros(7)
        bbh_gw_array = np.zeros(7)

        # Fractional rate of mass growth per year set to 
        # the Eddington rate(2.3e-8/yr)
        disk_bh_eddington_mass_growth_rate = 2.3e-8
        # minimum spin angle resolution 
        # (ie less than this value gets fixed to zero) 
        # e.g 0.02 rad=1deg
        disk_bh_spin_resolution_min = 0.02

        # Set up number of BH in disk
        disk_bh_num = setupdiskblackholes.setup_disk_nbh(
            opts.nsc_mass,
            opts.nsc_ratio_bh_num_star_num,
            opts.nsc_ratio_bh_mass_star_mass,
            opts.nsc_radius_outer,
            opts.nsc_density_index_outer,
            opts.smbh_mass,
            opts.disk_radius_outer,
            opts.disk_aspect_ratio_avg,
            opts.nsc_radius_crit,
            opts.nsc_density_index_inner,
        )

        # generate initial BH parameter arrays
        print("Generate initial BH parameter arrays")
        bh_initial_locations = setupdiskblackholes.setup_disk_blackholes_location(
                disk_bh_num, opts.disk_radius_outer)
        bh_initial_masses = setupdiskblackholes.setup_disk_blackholes_masses(
                disk_bh_num,
                opts.nsc_bh_imf_mode, opts.nsc_bh_imf_mass_max, opts.nsc_bh_imf_powerlaw_index)
        bh_initial_spins = setupdiskblackholes.setup_disk_blackholes_spins(
                disk_bh_num,
                opts.nsc_bh_spin_dist_mu, opts.nsc_bh_spin_dist_sigma)
        bh_initial_spin_angles = setupdiskblackholes.setup_disk_blackholes_spin_angles(
                disk_bh_num,
                bh_initial_spins)
        bh_initial_orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(
                disk_bh_num)
        if opts.flag_orb_ecc_damping == 1:
            bh_initial_orb_ecc = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform(disk_bh_num, opts.disk_bh_orb_ecc_max_init)
        else:
            bh_initial_orb_ecc = setupdiskblackholes.setup_disk_blackholes_circularized(disk_bh_num, opts.disk_bh_pro_orb_ecc_crit)

        # KN: should the inclination function be called in AGNBlackhole init since it takes other parameters of the BHs?
        bh_initial_orb_incl = setupdiskblackholes.setup_disk_blackholes_incl(disk_bh_num, bh_initial_locations, bh_initial_orb_ang_mom, disk_aspect_ratio)
        bh_initial_orb_arg_periapse = setupdiskblackholes.setup_disk_blackholes_arg_periapse(disk_bh_num)
        
        # Initialize black holes
        blackholes = AGNBlackHole(mass=bh_initial_masses,
                                  spin=bh_initial_spins,
                                  spin_angle=bh_initial_spin_angles,
                                  orbit_a=bh_initial_locations,
                                  orbit_inclination=bh_initial_orb_incl,
                                  orbit_e=bh_initial_orb_ecc,
                                  orbit_arg_periapse=bh_initial_orb_arg_periapse,
                                  mass_smbh=opts.smbh_mass,
                                  nsystems=disk_bh_num)
        
        # Initialize stars
        stars, disk_star_num = initializediskstars.init_single_stars(opts, id_start_val=blackholes.id_num.max()+1)
        print('disk_bh_num = {}, disk_star_num = {}'.format(disk_bh_num, disk_star_num))

        filing_cabinet = AGNFilingCabinet(category=np.full(blackholes.mass.shape, 0),
                                          agnobj=blackholes
                                          )
        filing_cabinet.add_objects(create_id=False, new_id_num=stars.id_num, new_category=np.full(stars.mass.shape, 2),
                                   new_direction=np.full(stars.mass.shape, 0),
                                   new_mass=stars.mass,
                                   new_spin=stars.spin,
                                   new_spin_angle=stars.spin_angle,
                                   new_orb_a=stars.orbit_a,
                                   new_orb_inclination=stars.orbit_inclination,
                                   new_orb_ang_mom=stars.orb_ang_mom,
                                   new_orb_e=stars.orbit_e,
                                   new_orb_arg_periapse=stars.orbit_arg_periapse,
                                   new_generation=stars.generations)

        # Generate initial inner disk arrays for objects that end up in the inner disk. 
        # This is to track possible EMRIs--we're tossing things in these arrays
        #  that end up with semi-major axis < 50rg
        # Assume all drawn from prograde population for now.
        #   SF: Is this assumption important here? Where does it come up?

        inner_disk_locations = []
        inner_disk_masses = []
        inner_disk_spins = []
        inner_disk_spin_angles = []
        inner_disk_orb_ecc = []
        inner_disk_orb_inc = []
        inner_disk_gens = []

        # Housekeeping: Set up time
        time_init = 0.0
        time_final = opts.timestep_duration_yr*opts.timestep_num

        # Find prograde BH orbiters. Identify BH with orb. ang mom > 0 (orb_ang_mom is only ever +1 or -1)
        prograde_bh_indices = np.where(blackholes.orb_ang_mom > 0)
        prograde_bh_id_num = np.take(blackholes.id_num, prograde_bh_indices)
        blackholes_pro = blackholes.copy()
        blackholes_pro.keep_objects(prograde_bh_indices)

        # Update filing cabinet
        filing_cabinet.change_direction(prograde_bh_id_num, np.ones(blackholes_pro.mass.shape))

        # Find prograde star orbiters.
        prograde_star_indices = np.where(stars.orb_ang_mom > 0)
        prograde_star_id_nums = np.take(stars.id_num, prograde_star_indices)
        stars_pro = stars.copy()
        stars_pro.keep_objects(prograde_star_indices)
        
        # Update filing cabinet
        filing_cabinet.change_direction(prograde_star_id_nums, np.ones(stars_pro.mass.shape))

        # Find retrograde black holes
        retrograde_bh_indices = np.where(blackholes.orb_ang_mom < 0)
        blackholes_retro = blackholes.copy()
        blackholes_retro.keep_objects(retrograde_bh_indices)
        retrograde_bh_id_num = np.take(blackholes.id_num, retrograde_bh_indices)

        # Update filing cabinet
        filing_cabinet.change_direction(retrograde_bh_id_num, np.full(blackholes_retro.mass.shape, -1))

        # Find retrograde stars
        retrograde_star_indices = np.where(stars.orb_ang_mom < 0)
        stars_retro = stars.copy()
        stars_retro.keep_objects(retrograde_star_indices)
        retrograde_star_id_num = np.take(stars.id_num, retrograde_star_indices)

        # Update filing cabinet
        filing_cabinet.change_direction(retrograde_star_id_num, np.full(stars_retro.mass.shape, -1))

        # Writing initial parameters to file
        stars.to_file(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/initial_params_star.dat"))
        blackholes.to_file(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/initial_params_bh.dat"))

        # Housekeeping:
        # Number of binary properties that we want to record (e.g. R1,R2,M1,M2,a1,a2,theta1,theta2,sep,com,t_gw,merger_flag,time of merger, gen_1,gen_2, bin_ang_mom, bin_ecc, bin_incl,bin_orb_ecc, nu_gw, h_bin)
        bin_properties_num = len(binary_field_names.split())+1
        bin_index = 0
        bin_num_total = 0
        number_of_mergers = 0
        frac_bin_retro = opts.fraction_bin_retro
        
        # Set up EMRI output array with properties we want to record (iteration, time, R,M,e,h_char,f_gw)
        num_of_emri_properties = 7
        nemri = 0

        # Set up BBH gw array with properties we want to record (iteration, time, sep, Mb, eb(around c.o.m.),h_char,f_gw)
        # Set up empty list of indices of BBH to track
        bbh_gw_indices = []
        num_of_bbh_gw_properties = 7
        nbbhgw = 0
        num_bbh_gw_tracked = 0

        # Set up empty initial Binary array
        # Initially all zeros, then add binaries plus details as appropriate
        binary_bh_array = np.zeros((bin_properties_num, opts.bin_num_max))

        # Set up normalization for t_gw (SF: I do not like this way of handling, flag for update)
        time_gw_normalization = tgw.normalize_tgw(opts.smbh_mass)
        print("Scale of t_gw (yrs)=", time_gw_normalization)

        # Set up merger array (identical to binary array)
        merger_array = np.zeros((bin_properties_num, opts.bin_num_max))

        # Set up output array (mergerfile)
        nprop_mergers = len(mergerfile.names_rec)
        merged_bh_array = np.zeros((nprop_mergers, opts.bin_num_max))

        # Multiple AGN episodes:
        # If you want to use the output of a previous AGN simulation as an input to another AGN phase
        # Make sure you have a file 'recipes/prior_model_name_population.dat' so that ReadInputs can take it in
        # and in your .ini file set switch prior_agn = 1.0.
        # Initial orb ecc is prior_ecc_factor*uniform[0,0.99]=[0,0.33] for prior_ecc_factor=0.3 (default)
        # SF: No promises this handles retrograde orbiters correctly yet
        if opts.flag_prior_agn == 1.0:

            prior_radii, prior_masses, prior_spins, prior_spin_angles, prior_gens \
                = ReadInputs.ReadInputs_prior_mergers()

            bh_pro_num = blackholes_pro.orbit_a.size

            prior_indices = setupdiskblackholes.setup_prior_blackholes_indices(bh_pro_num, prior_radii)
            prior_indices = prior_indices.astype('int32')
            blackholes_pro.keep_objects(prior_indices)

            print("prior indices", prior_indices)
            print("prior locations", blackholes_pro.orbit_a)
            print("prior gens", blackholes_pro.generations)
            prior_ecc_factor = 0.3
            blackholes_pro.orbit_e = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform_modified(prior_ecc_factor, bh_pro_num)
            print("prior ecc", blackholes_pro.orbit_e)

        # Start Loop of Timesteps
        print("Start Loop!")
        time_passed = time_init
        print("Initial Time(yrs) = ", time_passed)

        bh_mergers_current_num = 0
        timestep_current_num = 0

        while time_passed < time_final:
            # Record
            if not (opts.no_snapshots):

                blackholes_pro.to_file(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_bh_single_{timestep_current_num}.dat"))
                blackholes_retro.to_file(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_bh_single_retro_{timestep_current_num}.dat"))
                stars_pro.to_file(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_stars_single_{timestep_current_num}.dat"))
                stars_retro.to_file(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_stars_single_retro_{timestep_current_num}.dat"))
                
                # Binary output: does not work
                np.savetxt(
                    os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_bh_binary_{timestep_current_num}.dat"),
                    binary_bh_array[:, :bh_mergers_current_num+1].T,
                    header=binary_field_names
                )

                timestep_current_num += 1

            # Order of operations:        
            # No migration until orbital eccentricity damped to e_crit 
            # 1. check orb. eccentricity to see if any prograde_bh_location BH have orb. ecc. <e_crit.
            #    Create array prograde_bh_location_ecrit for those (mask prograde_bh_locations?)
            #       If yes, migrate those BH.
            #       All other BH, damp ecc and spin *down* BH (retrograde accretion), accrete mass.
            # 2. Run close encounters only on those prograde_bh_location_ecrit members.
        
            # Migrate
            # First if feedback present, find ratio of feedback heating torque to migration torque
            if opts.flag_thermal_feedback > 0:
                ratio_heat_mig_torques_agnobj = feedback_hankla21.feedback_hankla(
                    blackholes_pro.orbit_a, disk_surface_density, opts.disk_bh_eddington_ratio, opts.disk_alpha_viscosity)
            else:
                ratio_heat_mig_torques_agnobj = np.ones(len(blackholes_pro.orbit_a))

            # now for stars
            ratio_heat_mig_stars_torques = feedback_hankla21_stars.feedback_hankla_stars(
                stars_pro.orbit_a, disk_surface_density, opts.disk_star_eddington_ratio, opts.disk_alpha_viscosity
            )

            # then migrate as usual
            blackholes_pro.orbit_a = type1.type1_migration(
                opts.smbh_mass,
                blackholes_pro.orbit_a,
                blackholes_pro.mass,
                disk_surface_density,
                disk_aspect_ratio,
                opts.timestep_duration_yr,
                ratio_heat_mig_torques_agnobj,
                opts.disk_radius_trap,
                blackholes_pro.orbit_e,
                opts.disk_bh_pro_orb_ecc_crit
            )

            # Accrete
            blackholes_pro.mass = changebhmass.change_mass(
                blackholes_pro.mass,
                opts.disk_bh_eddington_ratio,
                disk_bh_eddington_mass_growth_rate,
                opts.timestep_duration_yr
            )

            # Spin up
            blackholes_pro.spin = changebh.change_spin_magnitudes(
                blackholes_pro.spin,
                opts.disk_bh_eddington_ratio,
                opts.disk_bh_torque_condition,
                opts.timestep_duration_yr,
                blackholes_pro.orbit_e,
                opts.disk_bh_pro_orb_ecc_crit,
            )

            # Torque spin angle
            blackholes_pro.spin_angle = changebh.change_spin_angles(
                blackholes_pro.spin_angle,
                opts.disk_bh_eddington_ratio,
                opts.disk_bh_torque_condition,
                disk_bh_spin_resolution_min,
                opts.timestep_duration_yr,
                blackholes_pro.orbit_e,
                opts.disk_bh_pro_orb_ecc_crit
            )

            # Damp BH orbital eccentricity
            blackholes_pro.orbit_e = orbital_ecc.orbital_ecc_damping(
                opts.smbh_mass,
                blackholes_pro.orbit_a,
                blackholes_pro.mass,
                disk_surface_density,
                disk_aspect_ratio,
                blackholes_pro.orbit_e,
                opts.timestep_duration_yr,
                opts.disk_bh_pro_orb_ecc_crit,
            )

            # Now do retrograde singles--change semi-major axis
            #   note this is dyn friction only, not true 'migration'
            # change retrograde eccentricity (some damping, some pumping)
            # damp orbital inclination
            blackholes_retro.orbit_e, blackholes_retro.orbit_a, blackholes_retro.orbit_inclination = crude_retro_evol.crude_retro_bh(
                opts.smbh_mass,
                blackholes_retro.mass,
                blackholes_retro.orbit_a,
                blackholes_retro.orbit_e,
                blackholes_retro.orbit_inclination,
                blackholes_retro.orbit_arg_periapse,
                disk_surface_density,
                opts.timestep_duration_yr
            )

            # and now stars

            # Locations
            stars_pro.orbit_a = type1.type1_migration(
                opts.smbh_mass,
                stars_pro.orbit_a,
                stars_pro.mass,
                disk_surface_density,
                disk_aspect_ratio,
                opts.timestep_duration_yr,
                ratio_heat_mig_stars_torques,
                opts.disk_radius_trap,
                stars_pro.orbit_e,
                opts.disk_bh_pro_orb_ecc_crit
            )
            
            # Accrete
            stars_pro.mass = changestarsmass.change_mass(
                stars_pro.mass,
                opts.disk_star_eddington_ratio,
                disk_bh_eddington_mass_growth_rate,  # do we need to alter this for stars?
                opts.timestep_duration_yr
            )
            # Spin up
            stars_pro.spin = changestars.change_spin_magnitudes(
                stars_pro.spin,
                opts.disk_star_eddington_ratio,
                opts.disk_bh_torque_condition,
                opts.timestep_duration_yr,
                stars_pro.orbit_e,
                opts.disk_bh_pro_orb_ecc_crit,
            )

            # Torque spin angle
            stars_pro.spin_angle = changestars.change_spin_angles(
                stars_pro.spin_angle,
                opts.disk_star_eddington_ratio,
                opts.disk_bh_torque_condition,
                disk_bh_spin_resolution_min,
                opts.timestep_duration_yr,
                stars_pro.orbit_e,
                opts.disk_bh_pro_orb_ecc_crit
            )

            # Damp stars orbital eccentricity
            stars_pro.orbit_e = orbital_ecc.orbital_ecc_damping(
                opts.smbh_mass,
                stars_pro.orbit_a,
                stars_pro.mass,
                disk_surface_density,
                disk_aspect_ratio,
                stars_pro.orbit_e,
                opts.timestep_duration_yr,
                opts.disk_bh_pro_orb_ecc_crit,
            )

            # Now do retrograde singles--change semi-major axis
            #   note this is dyn friction only, not true 'migration'
            # change retrograde eccentricity (some damping, some pumping)
            # damp orbital inclination
            
            # This is not working for retrograde stars, just says parameters are unreliable
            # stars_retro.orbit_e, stars_retro.orbit_a, stars_retro.orbit_inclination = crude_retro_evol.crude_retro_bh(
            #     opts.smbh_mass,
            #     stars_retro.mass,
            #     stars_retro.orbit_a,
            #     stars_retro.orbit_e,
            #     stars_retro.orbit_inclination,
            #     stars_retro.orbit_arg_periapse,
            #     disk_surface_density,
            #     opts.timestep_duration_yr
            # )

            # Perturb eccentricity via dynamical encounters
            if opts.flag_dynamic_enc > 0:
                prograde_bh_locn_orb_ecc = dynamics.circular_singles_encounters_prograde(
                    rng,
                    opts.smbh_mass,
                    blackholes_pro.orbit_a,
                    blackholes_pro.mass,
                    disk_surface_density,
                    disk_aspect_ratio,
                    blackholes_pro.orbit_e,
                    opts.timestep_duration_yr,
                    opts.disk_bh_pro_orb_ecc_crit,
                    opts.delta_energy_strong,
                )
                blackholes_pro.orbit_a = prograde_bh_locn_orb_ecc[0][0]
                blackholes_pro.orbit_e = prograde_bh_locn_orb_ecc[1][0]
                
                stars_pro_locn_orb_ecc = dynamics.circular_singles_encounters_prograde(
                    rng,
                    opts.smbh_mass,
                    stars_pro.orbit_a,
                    stars_pro.mass,
                    disk_surface_density,
                    disk_aspect_ratio,
                    stars_pro.orbit_e,
                    opts.timestep_duration_yr,
                    opts.disk_bh_pro_orb_ecc_crit,
                    opts.delta_energy_strong,
                )
                stars_pro.orbit_a = stars_pro_locn_orb_ecc[0][0]
                stars_pro.orbit_e = stars_pro_locn_orb_ecc[1][0]
            
            # Do things to the binaries--first check if there are any:
            if bin_index > 0:
                # First check that binaries are real. Discard any columns where the location or the mass is 0.
                # SF: I believe this step is handling an error checking thing that may have been
                #     set up in the previous timeloop if e.g. a binary either merged or was ionized?
                #     Please explain what this is and how it works right here?
                reality_flag = evolve.reality_check(binary_bh_array, bin_index, bin_properties_num)
                if reality_flag >= 0:
                    # One of the key parameter (mass or location is zero). Not real. Delete binary. Remove column at index = ionization_flag
                    binary_bh_array = np.delete(binary_bh_array, reality_flag, 1)
                    bin_index = bin_index - 1
                else:
                    # If there are still binaries after this, evolve them.
                    # If there are binaries, evolve them
                    # Damp binary orbital eccentricity
                    binary_bh_array = orbital_ecc.orbital_bin_ecc_damping(
                        opts.smbh_mass,
                        binary_bh_array,
                        disk_surface_density,
                        disk_aspect_ratio,
                        opts.timestep_duration_yr,
                        opts.disk_bh_pro_orb_ecc_crit
                    )
                    if (opts.flag_dynamic_enc > 0):
                        # Harden/soften binaries via dynamical encounters
                        # Harden binaries due to encounters with circular singletons (e.g. Leigh et al. 2018)
                        binary_bh_array = dynamics.circular_binaries_encounters_circ_prograde(
                            rng,
                            opts.smbh_mass,
                            blackholes_pro.orbit_a,
                            blackholes_pro.mass,
                            blackholes_pro.orbit_e,
                            opts.timestep_duration_yr,
                            opts.disk_bh_pro_orb_ecc_crit,
                            opts.delta_energy_strong,
                            binary_bh_array,
                            bin_index
                        )

                        # Soften/ ionize binaries due to encounters with eccentric singletons
                        binary_bh_array = dynamics.circular_binaries_encounters_ecc_prograde(
                            rng,
                            opts.smbh_mass,
                            blackholes_pro.orbit_a,
                            blackholes_pro.mass,
                            blackholes_pro.orbit_e,
                            opts.timestep_duration_yr,
                            opts.disk_bh_pro_orb_ecc_crit,
                            opts.delta_energy_strong,
                            binary_bh_array,
                            bin_index
                        )
                    # Harden binaries via gas
                    # Choose between Baruteau et al. 2011 gas hardening, or gas hardening from LANL simulations. To do: include dynamical hardening/softening from encounters
                    binary_bh_array = baruteau11.bin_harden_baruteau(
                        binary_bh_array,
                        bin_properties_num,
                        opts.smbh_mass,
                        opts.timestep_duration_yr,
                        time_gw_normalization,
                        bin_index,
                        time_passed,
                    )
                    # Check closeness of binary. Are black holes at merger condition separation
                    binary_bh_array = evolve.contact_check(binary_bh_array, bin_index, opts.smbh_mass)
                    # Accrete gas onto binary components
                    binary_bh_array = evolve.change_bin_mass(
                        binary_bh_array,
                        opts.disk_bh_eddington_ratio,
                        disk_bh_eddington_mass_growth_rate,
                        opts.timestep_duration_yr,
                        bin_properties_num,
                        bin_index
                    )
                    # Spin up binary components
                    binary_bh_array = evolve.change_bin_spin_magnitudes(
                        binary_bh_array,
                        opts.disk_bh_eddington_ratio,
                        opts.disk_bh_torque_condition,
                        opts.timestep_duration_yr,
                        bin_properties_num,
                        bin_index
                    )
                    # Torque angle of binary spin components
                    binary_bh_array = evolve.change_bin_spin_angles(
                        binary_bh_array,
                        opts.disk_bh_eddington_ratio,
                        opts.disk_bh_torque_condition,
                        disk_bh_spin_resolution_min,
                        opts.timestep_duration_yr,
                        bin_properties_num,
                        bin_index
                    )

                    if (opts.flag_dynamic_enc > 0):
                        # Spheroid encounters
                        binary_bh_array = dynamics.bin_spheroid_encounter(
                            rng,
                            opts.smbh_mass,
                            opts.timestep_duration_yr,
                            binary_bh_array,
                            time_passed,
                            bin_index,
                            opts.nsc_bh_imf_powerlaw_index,
                            opts.nsc_bh_imf_mode,
                            opts.delta_energy_strong,
                            opts.nsc_spheroid_normalization
                        )

                    if (opts.flag_dynamic_enc > 0):
                        # Recapture bins out of disk plane
                        binary_bh_array = dynamics.bin_recapture(
                            bin_index,
                            binary_bh_array,
                            opts.timestep_duration_yr
                        )

                    # Migrate binaries
                    # First if feedback present, find ratio of feedback heating torque to migration torque
                    if opts.flag_thermal_feedback > 0:
                        ratio_heat_mig_torques_bin_com = evolve.com_feedback_hankla(
                            binary_bh_array,
                            disk_surface_density,
                            opts.disk_bh_eddington_ratio,
                            opts.disk_alpha_viscosity
                        )
                    else:
                        ratio_heat_mig_torques_bin_com = np.ones(len(binary_bh_array[9, :]))

                    # Migrate binaries center of mass
                    binary_bh_array = evolve.bin_migration(
                        opts.smbh_mass,
                        binary_bh_array,
                        disk_surface_density,
                        disk_aspect_ratio,
                        opts.timestep_duration_yr,
                        ratio_heat_mig_torques_bin_com,
                        opts.disk_radius_trap,
                        opts.disk_bh_pro_orb_ecc_crit
                    )

                    # Test to see if any binaries separation is O(1r_g)
                    # If so, track them for GW freq, strain.
                    # Minimum BBH separation (in units of r_g)
                    min_bbh_gw_separation = 2.0
                    # If there are binaries AND if any separations are < min_bbh_gw_separation
                    bbh_gw_indices = np.where((binary_bh_array[8, :] < min_bbh_gw_separation) & (binary_bh_array[8, :] > 0))
                    
                    # If bbh_indices exists (ie is not empty)
                    if bbh_gw_indices:
                        # 1st time around.
                        if num_bbh_gw_tracked == 0:
                            old_bbh_gw_freq = 9.e-7*np.ones(np.size(bbh_gw_indices, 1))    
                        if num_bbh_gw_tracked > 0:
                            old_bbh_gw_freq = bbh_gw_freq

                        num_bbh_gw_tracked = np.size(bbh_gw_indices, 1)
                        nbbhgw = nbbhgw + num_bbh_gw_tracked
                        
                        # Now update BBH & generate NEW frequency & evolve  
                        
                        bbh_gw_strain, bbh_gw_freq = evolve.bbh_gw_params(
                            binary_bh_array,
                            bbh_gw_indices,
                            opts.smbh_mass,
                            opts.timestep_duration_yr,
                            old_bbh_gw_freq
                        )
                        
                        if num_bbh_gw_tracked == 1:
                            index = bbh_gw_indices[0]

                            temp_bbh_gw_array[0] = iteration
                            temp_bbh_gw_array[1] = time_passed
                            temp_bbh_gw_array[2] = binary_bh_array[8, index]
                            temp_bbh_gw_array[3] = binary_bh_array[2, index] + binary_bh_array[3, index]
                            temp_bbh_gw_array[4] = binary_bh_array[13, index]
                            temp_bbh_gw_array[5] = bbh_gw_strain
                            temp_bbh_gw_array[6] = bbh_gw_freq

                            bbh_gw_array = np.vstack((bbh_gw_array, temp_bbh_gw_array))

                        if num_bbh_gw_tracked > 1:
                            index = 0
                            for i in range(0, num_bbh_gw_tracked-1):

                                index = bbh_gw_indices[0][i]

                                # Record: iteration, time_passed, bin sep, bin_mass, bin_ecc(around c.o.m.),bin strain, bin freq       
                                temp_bbh_gw_array[0] = iteration
                                temp_bbh_gw_array[1] = time_passed
                                temp_bbh_gw_array[2] = binary_bh_array[8, index]
                                temp_bbh_gw_array[3] = binary_bh_array[2, index] + binary_bh_array[3, index]
                                temp_bbh_gw_array[4] = binary_bh_array[13, index]
                                temp_bbh_gw_array[5] = bbh_gw_strain[i]
                                temp_bbh_gw_array[6] = bbh_gw_freq[i]

                                bbh_gw_array = np.vstack((bbh_gw_array, temp_bbh_gw_array))

                    # Evolve GW frequency and strain
                    binary_bh_array = evolve.evolve_gw(
                        binary_bh_array,
                        bin_index,
                        opts.smbh_mass
                    )

                    # Check and see if merger flagged during hardening (row 11, if negative)
                    merger_flags = binary_bh_array[11, :]
                    any_merger = np.count_nonzero(merger_flags)

                    # Check and see if binary ionization flag raised. 
                    ionization_flag = evolve.ionization_check(binary_bh_array, bin_index, opts.smbh_mass)
                    # Default is ionization flag = -1
                    # If ionization flag >=0 then ionize bin_array[ionization_flag,;]
                    if ionization_flag >= 0:
                        # Append 2 new BH to arrays of single BH locations, masses, spins, spin angles & gens
                        # For now add 2 new orb ecc term of 0.01. TO DO: calculate v_kick and resulting perturbation to orb ecc.
                        new_location_1 = binary_bh_array[0, ionization_flag]
                        new_location_2 = binary_bh_array[1, ionization_flag]
                        new_mass_1 = binary_bh_array[2, ionization_flag]
                        new_mass_2 = binary_bh_array[3, ionization_flag]
                        new_spin_1 = binary_bh_array[4, ionization_flag]
                        new_spin_2 = binary_bh_array[5, ionization_flag]
                        new_spin_angle_1 = binary_bh_array[6, ionization_flag]
                        new_spin_angle_2 = binary_bh_array[7, ionization_flag]
                        new_gen_1 = binary_bh_array[14, ionization_flag]
                        new_gen_2 = binary_bh_array[15, ionization_flag]
                        new_orb_ecc_1 = 0.01
                        new_orb_ecc_2 = 0.01
                        new_orb_inc_1 = 0.0
                        new_orb_inc_2 = 0.0

                        # does not have orb_arg_periapse or orb_ang_mom??
                        # orb_ang_mom is only used to separate the pro and retrograde BH so this makes sense for now
                        blackholes_pro.add_blackholes(new_mass=([new_mass_1, new_mass_2]),
                                                           new_spin=([new_spin_1, new_spin_2]),
                                                           new_spin_angle=([new_spin_angle_1, new_spin_angle_2]),
                                                           new_orb_a=([new_location_1, new_location_2]),
                                                           new_generation=([new_gen_1, new_gen_2]),
                                                           new_orb_e=([new_orb_ecc_1, new_orb_ecc_2]),
                                                           new_orb_inclination=([new_orb_inc_1, new_orb_inc_2]),
                                                           new_orb_ang_mom=[1, 1],
                                                           new_orb_arg_periapse=[1.0, 1.0],
                                                           new_id_num=[blackholes_pro.id_num.max()+1, blackholes_pro.id_num.max()+2]
                                                           )

                        # Delete binary. Remove column at index = ionization_flag
                        binary_bh_array = np.delete(binary_bh_array, ionization_flag, 1)
                        # Reduce number of binaries
                        bin_index = bin_index - 1

                    # Test dynamics of encounters between binaries and eccentric singleton orbiters
                    # dynamics_binary_array = dynamics.circular_binaries_encounters_prograde(rng,opts.smbh_mass, prograde_bh_locations, prograde_bh_masses, disk_surf_model, disk_aspect_ratio_model, bh_orb_ecc, timestep, opts.disk_bh_pro_orb_ecc_crit, opts.delta_energy_strong,norm_tgw,bin_array,bindex,bin_properties_num)         

                    if opts.verbose:
                        print(merger_flags)
                    merger_indices = np.where(merger_flags < 0.0)
                    if isinstance(merger_indices, tuple):
                        merger_indices = merger_indices[0]
                    if opts.verbose:
                        print(merger_indices)
                    if any_merger > 0:
                        for i in range(any_merger):
                            if time_passed <= opts.timestep_duration_yr:
                                print("time_passed,loc1,loc2", time_passed, binary_bh_array[0, merger_indices[i]], binary_bh_array[1, merger_indices[i]])

                        # calculate merger properties
                            merged_mass = tichy08.merged_mass(
                                binary_bh_array[2, merger_indices[i]],
                                binary_bh_array[3, merger_indices[i]],
                                binary_bh_array[4, merger_indices[i]],
                                binary_bh_array[5, merger_indices[i]]
                            )
                            merged_spin = tichy08.merged_spin(
                                binary_bh_array[2, merger_indices[i]],
                                binary_bh_array[3, merger_indices[i]],
                                binary_bh_array[4, merger_indices[i]],
                                binary_bh_array[5, merger_indices[i]],
                                binary_bh_array[16, merger_indices[i]]
                            )
                            merged_chi_eff = chieff.chi_effective(
                                binary_bh_array[2, merger_indices[i]],
                                binary_bh_array[3, merger_indices[i]],
                                binary_bh_array[4, merger_indices[i]],
                                binary_bh_array[5, merger_indices[i]],
                                binary_bh_array[6, merger_indices[i]],
                                binary_bh_array[7, merger_indices[i]],
                                binary_bh_array[16, merger_indices[i]]
                            )
                            merged_chi_p = chieff.chi_p(
                                binary_bh_array[2, merger_indices[i]],
                                binary_bh_array[3, merger_indices[i]],
                                binary_bh_array[4, merger_indices[i]],
                                binary_bh_array[5, merger_indices[i]],
                                binary_bh_array[6, merger_indices[i]],
                                binary_bh_array[7, merger_indices[i]],
                                binary_bh_array[16, merger_indices[i]],
                                binary_bh_array[17, merger_indices[i]]
                            )
                            merged_bh_array[:, bh_mergers_current_num + i] = mergerfile.merged_bh(
                                merged_bh_array,
                                binary_bh_array,
                                merger_indices,
                                i,
                                merged_chi_eff,
                                merged_mass,
                                merged_spin,
                                nprop_mergers,
                                bh_mergers_current_num,
                                merged_chi_p,
                                time_passed
                            )
                        # do another thing
                        merger_array[:, merger_indices] = binary_bh_array[:, merger_indices]
                        # Reset merger marker to zero
                        # Remove merged binary from binary array. Delete column where merger_indices is the label.
                        binary_bh_array = np.delete(binary_bh_array, merger_indices, 1)
                
                        # Reduce number of binaries by number of mergers
                        bin_index = bin_index - len(merger_indices)
                        # Find relevant properties of merged BH to add to single BH arrays
                        num_mergers_this_timestep = len(merger_indices)
                
                        for i in range(0, num_mergers_this_timestep):
                            merged_bh_com = merged_bh_array[0, bh_mergers_current_num + i]
                            merged_mass = merged_bh_array[1, bh_mergers_current_num + i]
                            merged_spin = merged_bh_array[3, bh_mergers_current_num + i]
                            merged_spin_angle = merged_bh_array[4, bh_mergers_current_num + i]
                        # New bh generation is max of generations involved in merger plus 1
                            merged_bh_gen = np.maximum(merged_bh_array[11, bh_mergers_current_num + i], merged_bh_array[12, bh_mergers_current_num + i]) + 1.0
                        # Add to number of mergers
                        bh_mergers_current_num += len(merger_indices)
                        number_of_mergers += len(merger_indices)

                        # Append new merged BH to arrays of single BH locations, masses, spins, spin angles & gens
                        # For now add 1 new orb ecc term of 0.01. TO DO: calculate v_kick and resulting perturbation to orb ecc.
                        # no periapse
                        blackholes_pro.add_blackholes(new_mass=[merged_mass],
                                                      new_orb_a=[merged_bh_com],
                                                      new_spin=[merged_spin],
                                                      new_spin_angle=[merged_spin_angle],
                                                      new_orb_inclination=[0.0],
                                                      new_orb_ang_mom=[1.0],
                                                      new_orb_e=[0.01],
                                                      new_generation=[merged_bh_gen],
                                                      new_orb_arg_periapse=[1.],
                                                      new_id_num=[blackholes_pro.id_num.max()+1])
                        if opts.verbose:
                            print("New BH locations", blackholes_pro.orbit_a)
                        if opts.verbose:
                            print(merger_array)
                    else:
                        # No merger
                        # do nothing! hardening should happen FIRST (and now it does!)
                        if opts.verbose:
                            if bin_index > 0:  # verbose:
                                print(binary_bh_array[:, :int(bin_index)].T)  # this makes printing work as expected
            else:
                if opts.verbose:
                    print("No binaries formed yet")
                    # No Binaries present in bin_array. Nothing to do.
                # Finished evolving binaries

                # If a close encounter within mutual Hill sphere add a new Binary

                # check which binaries should get made
            close_encounters_indices = hillsphere.binary_check2(
                blackholes_pro.orbit_a, blackholes_pro.mass, opts.smbh_mass, blackholes_pro.orbit_e, opts.disk_bh_pro_orb_ecc_crit
            )

            if np.size(close_encounters_indices) > 0:
                # number of new binaries is length of 2nd dimension of close_encounters_indices
                bin_num_new = np.shape(close_encounters_indices)[1]
                # make new binaries
                binary_bh_array = add_new_binary.add_to_binary_array2(
                    rng,
                    binary_bh_array,
                    blackholes_pro.orbit_a,
                    blackholes_pro.mass,
                    blackholes_pro.spin,
                    blackholes_pro.spin_angle,
                    blackholes_pro.generations,
                    close_encounters_indices,
                    bin_index,
                    frac_bin_retro,
                    opts.smbh_mass,
                )
                bin_index += bin_num_new
                # Count towards total of any binary ever made (including those that are ionized)
                bin_num_total += bin_num_new
                # delete corresponding entries for new binary members from singleton arrays
                blackholes_pro.remove_objects(idx_remove=close_encounters_indices)

                # Empty close encounters
                empty = []
                close_encounters_indices = np.array(empty)

            # After this time period, was there a disk capture via orbital grind-down?
            # To do: What eccentricity do we want the captured BH to have? Right now ecc=0.0? Should it be ecc<h at a?             
            # Assume 1st gen BH captured and orb ecc =0.0
            # To do: Bias disk capture to more massive BH!
            capture = time_passed % opts.capture_time_yr
            if capture == 0:
                bh_capture_location = setupdiskblackholes.setup_disk_blackholes_location(
                    1, opts.disk_radius_capture_outer)
                bh_capture_mass = setupdiskblackholes.setup_disk_blackholes_masses(
                    1, opts.nsc_bh_imf_mode, opts.nsc_bh_imf_mass_max, opts.nsc_bh_imf_powerlaw_index)
                bh_capture_spin = setupdiskblackholes.setup_disk_blackholes_spins(
                    1, opts.nsc_bh_spin_dist_mu, opts.nsc_bh_spin_dist_sigma)
                bh_capture_spin_angle = setupdiskblackholes.setup_disk_blackholes_spin_angles(
                    1, bh_capture_spin)
                bh_capture_gen = [1]
                bh_capture_orb_ecc = [0.0]
                bh_capture_orb_incl = [0.0]
                # Append captured BH to existing singleton arrays. Assume prograde and 1st gen BH.
                blackholes_pro.add_blackholes(new_mass=bh_capture_mass,
                                              new_spin=bh_capture_spin,
                                              new_spin_angle=bh_capture_spin_angle,
                                              new_orb_a=bh_capture_location,
                                              new_orb_inclination=bh_capture_orb_incl,
                                              new_orb_ang_mom=np.ones(bh_capture_mass.size),
                                              new_orb_e=bh_capture_orb_ecc,
                                              new_orb_arg_periapse=np.ones(bh_capture_mass.size),
                                              new_generation=bh_capture_gen,
                                              new_id_num=np.arange(blackholes_pro.id_num.max()+1, len(bh_capture_mass) + blackholes_pro.id_num.max()+1,1))
            
            # Test if any BH or BBH are in the danger-zone (<mininum_safe_distance, default =50r_g) from SMBH.
            # Potential EMRI/BBH EMRIs.
            # Find prograde BH in inner disk. Define inner disk as <=50r_g. 
            # Since a 10Msun BH will decay into a 10^8Msun SMBH at 50R_g in ~38Myr and decay time propto a^4.
            # e.g at 25R_g, decay time is only 2.3Myr.
            min_safe_distance = 50.0
            inner_disk_indices = np.where(blackholes_pro.orbit_a < min_safe_distance)
            # adding retros too
            inner_disk_retro_indices = np.where(blackholes_retro.orbit_a < min_safe_distance)
            if np.size(inner_disk_indices) > 0:
                # Add BH to inner_disk_arrays
                inner_disk_locations = np.append(inner_disk_locations, blackholes_pro.orbit_a[inner_disk_indices])
                inner_disk_masses = np.append(inner_disk_masses, blackholes_pro.mass[inner_disk_indices])
                inner_disk_spins = np.append(inner_disk_spins, blackholes_pro.spin[inner_disk_indices])
                inner_disk_spin_angles = np.append(inner_disk_spin_angles, blackholes_pro.spin_angle[inner_disk_indices])
                inner_disk_orb_ecc = np.append(inner_disk_orb_ecc, blackholes_pro.orbit_e[inner_disk_indices])
                inner_disk_orb_inc = np.append(inner_disk_orb_inc, blackholes_pro.orbit_inclination[inner_disk_indices])
                inner_disk_gens = np.append(inner_disk_gens, blackholes_pro.generations[inner_disk_indices])
                # Remove BH from prograde_disk_arrays
                blackholes_pro.remove_objects(idx_remove=inner_disk_indices)
                # Empty disk_indices array
                empty = []
                inner_disk_indices = np.array(empty)

            if np.size(inner_disk_retro_indices) > 0:
                # Add BH to inner_disk_arrays
                inner_disk_locations = np.append(inner_disk_locations, blackholes_retro.orbit_a[inner_disk_retro_indices])
                inner_disk_masses = np.append(inner_disk_masses, blackholes_retro.mass[inner_disk_retro_indices])
                inner_disk_spins = np.append(inner_disk_spins, blackholes_retro.spin[inner_disk_retro_indices])
                inner_disk_spin_angles = np.append(inner_disk_spin_angles, blackholes_retro.spin_angle[inner_disk_retro_indices])
                inner_disk_orb_ecc = np.append(inner_disk_orb_ecc, blackholes_retro.orbit_e[inner_disk_retro_indices])
                inner_disk_orb_inc = np.append(inner_disk_orb_inc, blackholes_retro.orbit_inclination[inner_disk_retro_indices])
                inner_disk_gens = np.append(inner_disk_gens, blackholes_retro.generations[inner_disk_retro_indices])
                
                # Remove BH from retrograde_disk_arrays (don't forget arg periapse!)
                blackholes_retro.remove_objects(idx_remove=inner_disk_retro_indices)
                # Empty disk_indices array
                empty = []
                inner_disk_retro_indices = np.array(empty)
            
            if np.size(inner_disk_locations) > 0:
                inner_disk_locations = dynamics.bh_near_smbh(opts.smbh_mass,
                                                             inner_disk_locations,
                                                             inner_disk_masses,
                                                             inner_disk_orb_ecc,
                                                             opts.timestep_duration_yr)
                num_in_inner_disk = np.size(inner_disk_locations)
                # On 1st run through define old GW freqs (at say 9.e-7 Hz, since evolution change is 1e-6Hz)
                if nemri == 0:
                    old_gw_freq = 9.e-7*np.ones(num_in_inner_disk)
                if nemri > 0:
                    old_gw_freq = emri_gw_freq
                # Now update emris & generate NEW frequency & evolve   
                emri_gw_strain, emri_gw_freq = evolve.evolve_emri_gw(inner_disk_locations,
                                                                     inner_disk_masses, 
                                                                     opts.smbh_mass,
                                                                     opts.timestep_duration_yr,
                                                                     old_gw_freq)

            num_in_inner_disk = np.size(inner_disk_locations)
            nemri = nemri + num_in_inner_disk
            if num_in_inner_disk > 0:
                for i in range(0, num_in_inner_disk):
                    temp_emri_array[0] = iteration
                    temp_emri_array[1] = time_passed
                    temp_emri_array[2] = inner_disk_locations[i]
                    temp_emri_array[3] = inner_disk_masses[i]
                    temp_emri_array[4] = inner_disk_orb_ecc[i]
                    temp_emri_array[5] = emri_gw_strain[i]
                    temp_emri_array[6] = emri_gw_freq[i]

                    emri_array = np.vstack((emri_array, temp_emri_array))

            # if inner_disk_locations[i] <1R_g then merger!
            merger_dist = 1.0
            emri_merger_indices = np.where(inner_disk_locations <= merger_dist)

            # if mergers occurs, remove from inner_disk arrays and stop evolving
            # still getting some nans, but I think that's bc there's retros that should have been
            #  moved to prograde arrays
            if np.size(emri_merger_indices) > 0:
                inner_disk_locations = np.delete(inner_disk_locations, emri_merger_indices)
                inner_disk_masses = np.delete(inner_disk_masses, emri_merger_indices)
                inner_disk_spins = np.delete(inner_disk_spins, emri_merger_indices)
                inner_disk_spin_angles = np.delete(inner_disk_spin_angles, emri_merger_indices)
                inner_disk_orb_ecc = np.delete(inner_disk_orb_ecc, emri_merger_indices)
                inner_disk_orb_inc = np.delete(inner_disk_orb_inc, emri_merger_indices)
                inner_disk_gens = np.delete(inner_disk_gens, emri_merger_indices)
            # Empty emri_merger_indices array
            empty = []
            emri_merger_indices = np.array(empty)
            
            # Here is where we need to move retro to prograde if they've flipped in this timestep
            # If they're IN the disk prograde, OR if they've circularized:
            # stop treating them with crude retro evolution--it will be sad
            # SF: fix the inc threshhold later!!!
            inc_threshhold = 5.0 * np.pi/180.0
            flip_to_prograde_indices = np.where((np.abs(blackholes_retro.orbit_inclination) <= inc_threshhold) | (blackholes_retro.orbit_e == 0.0))
            if np.size(flip_to_prograde_indices) > 0:
                # add to prograde arrays
                blackholes_pro.add_blackholes(new_mass=blackholes_retro.mass[flip_to_prograde_indices],
                                              new_orb_a=blackholes_retro.orbit_a[flip_to_prograde_indices],
                                              new_spin=blackholes_retro.spin[flip_to_prograde_indices],
                                              new_spin_angle=blackholes_retro.spin_angle[flip_to_prograde_indices],
                                              new_orb_inclination=blackholes_retro.orbit_inclination[flip_to_prograde_indices],
                                              new_orb_ang_mom=np.ones(blackholes_retro.mass[flip_to_prograde_indices].size),
                                              new_orb_e=blackholes_retro.orbit_e[flip_to_prograde_indices],
                                              new_orb_arg_periapse=blackholes_retro.orbit_arg_periapse[flip_to_prograde_indices],
                                              new_generation=blackholes_retro.generations[flip_to_prograde_indices],
                                              new_id_num=blackholes_retro.id_num[flip_to_prograde_indices])
                # delete from retro arrays
                blackholes_retro.remove_objects(idx_remove=flip_to_prograde_indices)
            # empty array for flipping to prograde
            empty = []
            flip_to_prograde_indices = np.array(empty)

            # Iterate the time step
            time_passed = time_passed + opts.timestep_duration_yr
            # Print time passed every 10 timesteps for now
            time_iteration_tracker = 10.0*opts.timestep_duration_yr
            if time_passed % time_iteration_tracker == 0:
                print("Time passed=", time_passed)
        # End Loop of Timesteps at Final Time, end all changes & print out results

        print("End Loop!")
        print("Final Time (yrs) = ", time_passed)
        if opts.verbose:
            print("BH locations at Final Time")
            print(blackholes_pro.orbit_a)
        print("Number of binaries = ", bin_index)
        print("Total number of mergers = ", number_of_mergers)
        print("Mergers", merged_bh_array.shape)
        print("Nbh_disk", disk_bh_num)
        # Number of rows in each array, EMRIs and BBH_GW
        # If emri_array is 2-d then this line is ok, but if emri-array is empty then this line defaults to 7 (#elements in 1d)
        if len(emri_array.shape) > 1:
            total_emris = emri_array.shape[0]
        elif len(emri_array.shape) == 1:
            total_emris = 0

        if len(bbh_gw_array.shape) > 1:
            total_bbh_gws = bbh_gw_array.shape[0]
        elif len(bbh_gw_array.shape) == 1:
            total_bbh_gws = 0

        # Write out all the singletons after AGN episode, so can use this as input to another AGN phase.
        # Want to store [Radius, Mass, Spin mag., Spin. angle, gen.]
        # So num_properties_stored = 5 (for now)
        # Note eccentricity will relax, so ignore. Inclination assumed 0deg.

        # Set up array for population that survives AGN episode, so can use as a draw for next episode.
        # Need total of 1) size of prograde_bh_array for number of singles at end of run and 
        # 2) size of bin_array for number of BH in binaries at end of run for
        # number of survivors.
        total_bh_survived = blackholes_pro.orbit_a.shape[0] + 2*bin_index
        num_properties_stored = 5

        # Set up arrays for properties:
        bin_r1 = np.zeros(bin_index)
        bin_r2 = np.zeros(bin_index)
        bin_m1 = np.zeros(bin_index)
        bin_m2 = np.zeros(bin_index)
        bin_a1 = np.zeros(bin_index)
        bin_a2 = np.zeros(bin_index)
        bin_theta1 = np.zeros(bin_index)
        bin_theta2 = np.zeros(bin_index)
        bin_gen1 = np.zeros(bin_index)
        bin_gen2 = np.zeros(bin_index)        

        for i in range(0,bin_index):
            bin_r1[i] = binary_bh_array[0, i]
            bin_r2[i] = binary_bh_array[1, i]
            bin_m1[i] = binary_bh_array[2, i]
            bin_m2[i] = binary_bh_array[3, i]
            bin_a1[i] = binary_bh_array[4, i]
            bin_a2[i] = binary_bh_array[5, i]
            bin_theta1[i] = binary_bh_array[6, i]
            bin_theta2[i] = binary_bh_array[7, i]
            bin_gen1[i] = binary_bh_array[14, i]
            bin_gen2[i] = binary_bh_array[15, i]

        total_emri_array = np.zeros((total_emris, num_of_emri_properties))
        surviving_bh_array = np.zeros((total_bh_survived, num_properties_stored))
        total_bbh_gw_array = np.zeros((total_bbh_gws, num_of_bbh_gw_properties))

        # inclination is set to random value bc not set above
        blackholes_pro.add_blackholes(new_mass=np.concatenate([bin_m1, bin_m2]),
                                      new_spin=np.concatenate([bin_a1, bin_a2]),
                                      new_spin_angle=np.concatenate([bin_theta1, bin_theta2]),
                                      new_orb_a=np.concatenate([bin_r1, bin_r2]),
                                      new_orb_inclination=np.ones(np.concatenate([bin_m1, bin_m2]).size),
                                      new_orb_ang_mom=np.ones(np.concatenate([bin_m1, bin_m2]).size),
                                      new_orb_e=np.ones(np.concatenate([bin_m1, bin_m2]).size),
                                      new_orb_arg_periapse=np.ones(np.concatenate([bin_m1, bin_m2]).size),
                                      new_generation=np.concatenate([bin_gen1, bin_gen2]),
                                      new_id_num=np.arange(blackholes_pro.id_num.max()+1,len(bin_m1) + len(bin_m2) + blackholes_pro.id_num.max()+1, 1))

        print(len(blackholes_pro.id_num))
        print(len(blackholes_pro.mass))
        surviving_bh_array[:, 0] = blackholes_pro.orbit_a
        surviving_bh_array[:, 1] = blackholes_pro.mass
        surviving_bh_array[:, 2] = blackholes_pro.spin
        surviving_bh_array[:, 3] = blackholes_pro.spin_angle
        surviving_bh_array[:, 4] = blackholes_pro.generations

        total_emri_array = emri_array
        total_bbh_gw_array = bbh_gw_array
        if True and number_of_mergers > 0:  # verbose:
            print(merged_bh_array[:, :number_of_mergers].T)

        iteration_save_name = f"run{iteration_zfilled_str}/{opts.fname_output_mergers}"
        np.savetxt(os.path.join(opts.work_directory, iteration_save_name), merged_bh_array[:, :number_of_mergers].T, header=merger_field_names)

        # Add mergers to population array including the iteration number 
        # this line is linebreak between iteration outputs consisting of the repeated iteration number in each column
        iteration_row = np.repeat(iteration, number_of_mergers)
        survivor_row = np.repeat(iteration, num_properties_stored)
        # Append each iteration result to output arrays
        merged_bh_array_pop.append(np.concatenate((iteration_row[np.newaxis], merged_bh_array[:, :number_of_mergers])).T)
        surviving_bh_array_pop.append(np.concatenate((survivor_row[np.newaxis], surviving_bh_array[:total_bh_survived, :])))

        if total_emris > 0:
            emris_array_pop.append(total_emri_array[:total_emris, :])
        # If there are non-zero elements in total_bbh_gw_array
        if total_bbh_gws > 0:
            gw_array_pop.append(total_bbh_gw_array[:total_bbh_gws, :])
    # save all mergers from Monte Carlo
    merger_pop_field_names = "iter " + merger_field_names  # Add "Iter" to field names
    population_header = f"Initial seed: {opts.seed}\n{merger_pop_field_names}"  # Include initial seed
    basename, extension = os.path.splitext(opts.fname_output_mergers)
    population_save_name = f"{basename}_population{extension}"
    survivors_save_name = f"{basename}_survivors{extension}"
    emris_save_name = f"{basename}_emris{extension}"
    gws_save_name = f"{basename}_lvk{extension}"
    np.savetxt(os.path.join(opts.work_directory, population_save_name), np.vstack(merged_bh_array_pop), header=population_header)
    np.savetxt(os.path.join(opts.work_directory, survivors_save_name), np.vstack(surviving_bh_array_pop))
    np.savetxt(os.path.join(opts.work_directory, emris_save_name), np.vstack(emris_array_pop))
    np.savetxt(os.path.join(opts.work_directory, gws_save_name), np.vstack(gw_array_pop))


if __name__ == "__main__":
    main()
