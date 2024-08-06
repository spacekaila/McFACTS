#!/usr/bin/env python3
import os
from os.path import isfile, isdir
from pathlib import Path

from cgi import print_arguments
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import scipy.interpolate

import sys
import argparse

from mcfacts.inputs import ReadInputs

from mcfacts.objects.agnobject import AGNStar

from mcfacts.setup import setupdiskblackholes
from mcfacts.setup import setupdiskstars
from mcfacts.physics.migration.type1 import type1
from mcfacts.physics.accretion.eddington import changebhmass
from mcfacts.physics.accretion.eddington import changestarsmass
from mcfacts.physics.accretion.torque import changebh
from mcfacts.physics.accretion.torque import changestars
from mcfacts.physics.feedback.hankla21 import feedback_hankla21
from mcfacts.physics.dynamics import dynamics
from mcfacts.physics.eccentricity import orbital_ecc
from mcfacts.physics.binary.formation import hillsphere
from mcfacts.physics.binary.formation import add_new_binary
from mcfacts.physics.binary.evolve import evolve
from mcfacts.physics.binary.harden import baruteau11
from mcfacts.physics.binary.merge import tichy08
from mcfacts.physics.binary.merge import chieff
from mcfacts.physics.binary.merge import tgw
#from mcfacts.tests import tests
from mcfacts.outputs import mergerfile

binary_field_names="R1 R2 M1 M2 a1 a2 theta1 theta2 sep com t_gw merger_flag t_mgr  gen_1 gen_2  bin_ang_mom bin_ecc bin_incl bin_orb_ecc nu_gw h_bin"
binary_stars_field_names="R1 R2 M1 M2 R1_star R2_star a1 a2 theta1 theta2 sep com t_gw merger_flag t_mgr  gen_1 gen_2  bin_ang_mom bin_ecc bin_incl bin_orb_ecc nu_gw h_bin"
merger_field_names=' '.join(mergerfile.names_rec)
bh_initial_field_names = "disk_location mass spin spin_angle orb_ang_mom orb_ecc orb_incl"
DEFAULT_INI = Path(__file__).parent.resolve() / ".." / "recipes" / "model_choice.ini"
#DEFAULT_INI = Path(__file__).parent.resolve() / ".." / "recipes" / "paper1_fig_dyn_on.ini"
DEFAULT_PRIOR_POP = Path(__file__).parent.resolve() / ".." / "recipes" / "prior_mergers_population.dat"
#DEFAULT_INI = Path(__file__).parent.resolve() / ".." / "recipes" / "model_choice.ini"
#DEFAULT_PRIOR_POP = Path(__file__).parent.resolve() / ".." / "recipes" / "prior_mergers_population.dat"
assert DEFAULT_INI.is_file()
#assert DEFAULT_PRIOR_POP.is_file()

def arg():
    import argparse
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--n_bins_max", default=1000, type=int)
    parser.add_argument("--n_bins_max_out", default=100, type=int)
    parser.add_argument("--fname-ini",help="Filename of configuration file",
        default=DEFAULT_INI,type=str)
    parser.add_argument("--fname-output-mergers",default="output_mergers.dat",
        help="output merger file (if any)",type=str)
    parser.add_argument("--fname-snapshots-bh",
        default="output_bh_[single|binary]_$(index).dat",
        help="output of BH index file ")
    parser.add_argument("--no-snapshots", action='store_true')
    parser.add_argument("--verbose",action='store_true')
    parser.add_argument("-w", "--work-directory",
        default=Path().parent.resolve(),
        help="Set the working directory for saving output. Default: current working directory",
        type=str
    )
    parser.add_argument("--seed", type=int, default=None,
        help="Set the random seed. Randomly sets one if not passed. Default: None")
    parser.add_argument("--fname-log", default=None, type=str,
        help="Specify a file to save the arguments for mcfacts")
    
    ## Add inifile arguments
    # Read default inifile
    _variable_inputs, _disk_model_radius_array, _surface_density_array, _aspect_ratio_array \
        = ReadInputs.ReadInputs_ini(DEFAULT_INI,False)
    # Loop the arguments
    for name in _variable_inputs:
        _metavar    = name
        _opt        = "--%s"%(name)
        _default    = _variable_inputs[name]
        _dtype      = type(_variable_inputs[name])
        parser.add_argument(
            _opt,
            default=_default,
            type=_dtype,
            metavar=_metavar,
           )

    ## Parse arguments
    opts = parser.parse_args()
    # Check that the inifile exists
    assert isfile(opts.fname_ini)
    # Convert to path objects
    opts.fname_ini = Path(opts.fname_ini)
    assert opts.fname_ini.is_file()
    opts.fname_snapshots_bh = Path(opts.fname_snapshots_bh)
    opts.fname_output_mergers = Path(opts.fname_output_mergers)

    ## Parse inifile
    # Read inifile
    variable_inputs, disk_model_radius_array, surface_density_array, aspect_ratio_array \
        = ReadInputs.ReadInputs_ini(opts.fname_ini, opts.verbose)
    # Okay, this is important. The priority of input arguments is:
    # command line > specified inifile > default inifile
    for name in variable_inputs:
        print(name, hasattr(opts, name), getattr(opts, name), _variable_inputs[name], variable_inputs[name])
        if getattr(opts, name) != _variable_inputs[name]:
            print(name)
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

    # Update opts with variable inputs
    opts.disk_model_radius_array = disk_model_radius_array
    opts.surface_density_array = surface_density_array
    opts.aspect_ratio_array = aspect_ratio_array
    if opts.verbose:
        for item in opts.__dict__:
            print(item, getattr(opts, item))

    # Get the user-defined or default working directory / output location
    opts.work_directory = Path(opts.work_directory).resolve()
    if not isdir(opts.work_directory):
        os.mkdir(opts.work_directory)
    assert opts.work_directory.is_dir()
    try: # check if working directory for output exists
        os.stat(opts.work_directory)
    except FileNotFoundError as e:
        raise e
    print(f"Output will be saved to {opts.work_directory}")

    # Get the parent path to this file and cd to that location for runtime
    opts.runtime_directory = Path(__file__).parent.resolve()
    assert opts.runtime_directory.is_dir()
    os.chdir(opts.runtime_directory)

    # set the seed for random number generation and reproducibility if not user-defined
    if opts.seed == None:
        opts.seed = np.random.randint(low=0, high=int(1e18), dtype=np.int_)
        print(f'Random number generator seed set to: {opts.seed}')


    # Write parameters to log file
    if not opts.fname_log is None:
        with open(opts.work_directory / opts.fname_log, 'w') as F:
            for item in opts.__dict__:
                line = "%s = %s\n"%(item, str(opts.__dict__[item]))
                F.write(line)
    return opts


def main():
    """
    """
    # Setting up automated input parameters
    # see IOdocumentation.txt for documentation of variable names/types/etc.
    opts = arg()

    # create surface density & aspect ratio functions from input arrays
    surf_dens_func_log = scipy.interpolate.UnivariateSpline(
        opts.disk_model_radius_array, np.log(opts.surface_density_array))
    surf_dens_func = lambda x, f=surf_dens_func_log: np.exp(f(x))

    aspect_ratio_func_log = scipy.interpolate.UnivariateSpline(
        opts.disk_model_radius_array, np.log(opts.aspect_ratio_array))
    aspect_ratio_func = lambda x, f=aspect_ratio_func_log: np.exp(f(x))
    
    merged_bh_array_pop = []

    surviving_bh_array_pop = []
    
    emris_array_pop = []

    gw_array_pop = []

    temp_emri_array = np.zeros(7)

    emri_array = np.zeros(7)

    temp_bbh_gw_array = np.zeros(7)

    bbh_gw_array = np.zeros(7)

    for iteration in range(opts.n_iterations):
        print("Iteration", iteration)
        # Set random number generator for this run with incremented seed
        # ALERT: ONLY this random number generator can be used throughout the code to ensure reproducibility.
        rng = np.random.default_rng(opts.seed + iteration)

        # Make subdirectories for each iteration
        # Fills run number with leading zeros to stay sequential
        iteration_zfilled_str = f"{iteration:>0{int(np.log10(opts.n_iterations))+1}}"
        try: # Make subdir, exit if it exists to avoid clobbering.
            os.makedirs(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}"), exist_ok=False)
        except FileExistsError:
            raise FileExistsError(f"Directory \'run{iteration_zfilled_str}\' exists. Exiting so I don't delete your data.")

        # can index other parameter lists here if needed.
        # galaxy_type = galaxy_models[iteration] # e.g. star forming/spiral vs. elliptical
        # NSC mass
        # SMBH mass


        #Set up number of BH in disk
        n_bh = setupdiskblackholes.setup_disk_nbh(
            opts.M_nsc,
            opts.nbh_nstar_ratio,
            opts.mbh_mstar_ratio,
            opts.r_nsc_out,
            opts.nsc_index_outer,
            opts.mass_smbh,
            opts.disk_outer_radius,
            opts.h_disk_average,
            opts.r_nsc_crit,
            opts.nsc_index_inner,
        )

        #This generates 10^6 more stars than BH so for right now I have artificially limited it to 5000 stars.
        n_stars = setupdiskstars.setup_disk_nstars(
            opts.M_nsc,
            opts.nbh_nstar_ratio,
            opts.mbh_mstar_ratio,
            opts.r_nsc_out,
            opts.nsc_index_outer,
            opts.mass_smbh,
            opts.disk_outer_radius,
            opts.h_disk_average,
            opts.r_nsc_crit,
            opts.nsc_index_inner,
        )
        n_stars = np.int64(5000)

        print('n_bh = {}, n_stars = {}'.format(n_bh,n_stars))


        # generate initial BH parameter arrays
        print("Generate initial BH parameter arrays")
        bh_initial_locations = setupdiskblackholes.setup_disk_blackholes_location(
            rng,
            n_bh,
            opts.disk_outer_radius,
        )
        bh_initial_masses = setupdiskblackholes.setup_disk_blackholes_masses(
            rng,
            n_bh,
            opts.mode_mbh_init,
            opts.max_initial_bh_mass,
            opts.mbh_powerlaw_index,
        )
        bh_initial_spins = setupdiskblackholes.setup_disk_blackholes_spins(
            rng,
            n_bh,
            opts.mu_spin_distribution,
            opts.sigma_spin_distribution
        )
        bh_initial_spin_angles = setupdiskblackholes.setup_disk_blackholes_spin_angles(
            rng,
            n_bh,
            bh_initial_spins
        )
        bh_initial_orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(
            rng,
            n_bh
        )
        if opts.orb_ecc_damping == 1:
            bh_initial_orb_ecc = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform(rng,n_bh)
        else:
            bh_initial_orb_ecc = setupdiskblackholes.setup_disk_blackholes_circularized(rng,n_bh,opts.crit_ecc)

        bh_initial_orb_incl = setupdiskblackholes.setup_disk_blackholes_inclination(rng,n_bh)
         
        bh_initial_generations = np.ones((n_bh,),dtype=int)

        #----------now stars
        n_stars = 200 #working on making this physical

        #Need to write an initialization function so we don't have to generate the arrays by hand.
        print("Generate initial star parameter arrays")

        star_mass = setupdiskstars.setup_disk_stars_masses(rng, n_stars, opts.min_initial_star_mass,opts.max_initial_star_mass,opts.star_mass_powerlaw_index)
        star_radius = setupdiskstars.setup_disk_stars_radii(star_mass)
        star_spin = setupdiskstars.setup_disk_stars_spins(rng, n_stars, opts.mu_star_spin_distribution, opts.sigma_star_spin_distribution)
        star_spin_angle = setupdiskstars.setup_disk_stars_spin_angles(rng, n_stars, star_spin)
        star_orbit_a = setupdiskstars.setup_disk_stars_location(rng, n_stars, opts.disk_outer_radius)
        star_orbit_inclination = setupdiskstars.setup_disk_stars_inclination(rng,n_stars)
        star_orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(rng,n_stars)
        if opts.orb_ecc_damping == 1:
            star_orbit_e = setupdiskstars.setup_disk_stars_eccentricity_uniform(rng,n_stars)
        else:
            star_orbit_e = setupdiskstars.setup_disk_stars_circularized(rng,n_stars,opts.crit_ecc)
        star_Y = opts.stars_initial_Y
        star_Z = opts.stars_initial_Z

        stars = AGNStar(mass = star_mass,
                        spin = star_spin,
                        spin_angle = star_spin_angle,
                        orbit_a = star_orbit_a, #this is location
                        orbit_inclination = star_orbit_inclination,
                        orbit_e = star_orbit_e,
                        orb_ang_mom = star_orb_ang_mom,
                        star_radius = star_radius,
                        star_Y = star_Y,
                        star_Z = star_Z,
                        n_stars = n_stars)


        #Generate initial inner disk arrays for objects that end up in the inner disk.
        #Assume all drawn from prograde population for now.
        inner_disk_locations = []
        inner_disk_masses =[]
        inner_disk_spins = []
        inner_disk_spin_angles = []
        inner_disk_orb_ecc = []
        inner_disk_orb_inc = []
        inner_disk_gens = []

        # assign functions to variable names (continuity issue)
        # Disk surface density (in kg/m^2) is a function of radius, where radius is in r_g
        disk_surface_density = surf_dens_func
        # and disk aspect ratio is also a function of radius, where radius is in r_g
        disk_aspect_ratio = aspect_ratio_func
        # Housekeeping: Set up time
        initial_time = 0.0
        final_time = opts.timestep*opts.number_of_timesteps

        # Find prograde BH orbiters. Identify BH with orb. ang mom =+1
        bh_orb_ang_mom_indices = np.array(bh_initial_orb_ang_mom)
        prograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == 1)
        #retrograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == -1)
        prograde_bh_locations = bh_initial_locations[prograde_orb_ang_mom_indices]
        sorted_prograde_bh_locations = np.sort(prograde_bh_locations)
        #Use masses of prograde BH only
        prograde_bh_masses = bh_initial_masses[prograde_orb_ang_mom_indices]
        # Orbital eccentricities
        prograde_bh_orb_ecc = bh_initial_orb_ecc[prograde_orb_ang_mom_indices]
        print("Prograde orbital eccentricities",prograde_bh_orb_ecc)
        # Find which orbital eccentricities are <=h the disk aspect ratio and set up a mask
        #prograde_bh_crit_ecc = np.ma.masked_where(prograde_bh_orb_ecc >= aspect_ratio_func(prograde_bh_locations),prograde_bh_orb_ecc)
        # Orb eccentricities <2h (simple exponential damping): mask entries > 2*aspect_ratio
        #prograde_bh_modest_ecc = np.ma.masked_where(prograde_bh_orb_ecc > 2.0*aspect_ratio_func(prograde_bh_locations),prograde_bh_orb_ecc)
        #Orb eccentricities >2h (modified exponential damping): mask entries < 2*aspect_ratio
        #prograde_bh_large_ecc = np.ma.masked_where(prograde_bh_orb_ecc < 2.0*aspect_ratio_func(prograde_bh_locations),prograde_bh_orb_ecc)
        # Apply ecc damping to this masked array (where true)
        #prograde_bh_orb_ecc_damp = orbital_ecc.orbital_ecc_damping(opts.mass_smbh, prograde_bh_locations, prograde_bh_masses, surf_dens_func, aspect_ratio_func, prograde_bh_orb_ecc, opts.timestep, opts.crit_ecc)

        #print('modest ecc ',prograde_bh_modest_ecc)
        #print('damped ecc',prograde_bh_orb_ecc_damp)

        # Test dynamics
        #post_dynamics_orb_ecc = dynamics.circular_singles_encounters_prograde(rng,opts.mass_smbh, prograde_bh_locations, prograde_bh_masses, surf_dens_func, aspect_ratio_func, prograde_bh_orb_ecc, opts.timestep, opts.crit_ecc, de)

        #First sort all stars by location
        stars.sort(stars.orbit_a)

        #prograde stars stuff
        prograde_stars_orb_ang_mom_indices = np.where(stars.orb_ang_mom == 1)
        prograde_stars = AGNStar(mass = stars.mass[prograde_stars_orb_ang_mom_indices],
                                 spin = stars.spin[prograde_stars_orb_ang_mom_indices],
                                 spin_angle = stars.spin_angle[prograde_stars_orb_ang_mom_indices],
                                 orbit_a = stars.orbit_a[prograde_stars_orb_ang_mom_indices],
                                 orbit_inclination = stars.orbit_inclination[prograde_stars_orb_ang_mom_indices],
                                 orb_ang_mom = stars.orb_ang_mom[prograde_stars_orb_ang_mom_indices],
                                 orbit_e = stars.orbit_e[prograde_stars_orb_ang_mom_indices],
                                 star_radius=stars.star_radius[prograde_stars_orb_ang_mom_indices],
                                 star_Y=stars.star_Y[prograde_stars_orb_ang_mom_indices],
                                 star_Z=stars.star_Z[prograde_stars_orb_ang_mom_indices])






        # Migrate
        # First if feedback present, find ratio of feedback heating torque to migration torque
        #if feedback > 0:
        #        ratio_heat_mig_torques = feedback_hankla21.feedback_hankla(prograde_bh_locations, surf_dens_func, opts.frac_Eddington_ratio, opts.alpha)
        #else:
        #        ratio_heat_mig_torques = np.ones(len(prograde_bh_locations))
        # then migrate as usual
        #prograde_bh_locations_new = type1.type1_migration(opts.mass_smbh , prograde_bh_locations, prograde_bh_masses, disk_surface_density, disk_aspect_ratio, opts.timestep, ratio_heat_mig_torques, opts.trap_radius, prograde_bh_orb_ecc,opts.crit_ecc)


        #Orbital inclinations
        prograde_bh_orb_incl = bh_initial_orb_incl[prograde_orb_ang_mom_indices]
        #prograde_stars_orb_incl = stars_initial_orb_incl[prograde_stars_orb_ang_mom_indices]
        #print("Prograde orbital inclinations")

        # Housekeeping: Fractional rate of mass growth per year at 
        # the Eddington rate(2.3e-8/yr)
        mass_growth_Edd_rate = 2.3e-8
    
        # Housekeeping: minimum spin angle resolution 
        # (ie less than this value gets fixed to zero) 
        # e.g 0.02 rad=1deg
        spin_minimum_resolution = 0.02
        # Torque prograde orbiting BH only
        prograde_bh_spins = bh_initial_spins[prograde_orb_ang_mom_indices]
        prograde_bh_spin_angles = bh_initial_spin_angles[prograde_orb_ang_mom_indices]
        prograde_bh_generations = bh_initial_generations[prograde_orb_ang_mom_indices]

        #Torque prograde orbiting stars only
        """prograde_stars_spins = stars_initial_spins[prograde_stars_orb_ang_mom_indices]
        prograde_stars_spin_angles = stars_initial_spin_angles[prograde_stars_orb_ang_mom_indices]
        prograde_stars_generations = stars_initial_generations[prograde_stars_orb_ang_mom_indices]
        prograde_stars_radii = stars_initial_radii[prograde_stars_orb_ang_mom_indices]
        prograde_stars_X = stars_initial_X[prograde_stars_orb_ang_mom_indices]
        prograde_stars_Y = stars_initial_Y[prograde_stars_orb_ang_mom_indices]
        prograde_stars_Z = stars_initial_Z[prograde_stars_orb_ang_mom_indices] """



        # Writing initial parameters to file
        np.savetxt(
                os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/initial_params_bh.dat"),
                np.c_[bh_initial_locations.T,
                      bh_initial_masses.T,
                      bh_initial_spins.T,
                      bh_initial_spin_angles.T,
                      bh_initial_orb_ang_mom.T,
                      bh_initial_orb_ecc.T,
                      bh_initial_orb_incl.T],
                header = bh_initial_field_names
        )

        stars.to_file(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/initial_params_stars2.dat"))


        # Housekeeping:
        # Number of binary properties that we want to record (e.g. R1,R2,M1,M2,a1,a2,theta1,theta2,sep,com,t_gw,merger_flag,time of merger, gen_1,gen_2, bin_ang_mom, bin_ecc, bin_incl,bin_orb_ecc, nu_gw, h_bin)
        number_of_bin_properties = len(binary_field_names.split())+1
        integer_nbinprop = int(number_of_bin_properties)
        bin_index = 0
        nbin_ever_made_index = 0
        test_bin_number = opts.n_bins_max
        integer_test_bin_number = int(test_bin_number)
        number_of_mergers = 0
        int_n_timesteps = int(opts.number_of_timesteps)

        number_of_stars_bin_properties = len(binary_stars_field_names.split())+1
        integer_stars_nbinprop = int(number_of_stars_bin_properties)
        bin_stars_index = 0
        nbin_stars_ever_made_index = 0
        test_bin_stars_number = opts.n_bins_max
        integer_test_bin_stars_number = int(test_bin_stars_number)
        number_of_stars_mergers = 0
        # Set up EMRI output array with properties we want to record (iteration, time, R,M,e,h_char,f_gw)
        
        num_of_emri_properties = 7
        nemri = 0

        #Set up BBH gw array with properties we want to record (iteration, time, sep, Mb, eb(around c.o.m.),h_char,f_gw)
        #Set up empty list of indices of BBH to track
        bbh_gw_indices = []
        num_of_bbh_gw_properties = 7
        nbbhgw = 0

        # Set up empty initial Binary array
        # Initially all zeros, then add binaries plus details as appropriate
        binary_bh_array = np.zeros((integer_nbinprop,integer_test_bin_number))
        binary_stars_array = np.zeros((integer_stars_nbinprop,integer_test_bin_stars_number))
        # Set up empty initial Binary gw array. Initially all zeros, but records gw freq and strain for all binaries ever made at each timestep, including ones that don't merge or are ionized
        gw_data_array =np.zeros((int_n_timesteps,integer_test_bin_number))
        # Set up normalization for t_gw (SF: I do not like this way of handling, flag for update)
        norm_t_gw = tgw.normalize_tgw(opts.mass_smbh)
        print("Scale of t_gw (yrs)=", norm_t_gw)
    
        # Set up merger array (identical to binary array)
        merger_array = np.zeros((integer_nbinprop,integer_test_bin_number))
        merger_stars_array = np.zeros((integer_stars_nbinprop,integer_test_bin_stars_number))

        # Set up output array (mergerfile)
        nprop_mergers=len(mergerfile.names_rec)
        integer_nprop_merge=int(nprop_mergers)
        merged_bh_array = np.zeros((integer_nprop_merge,integer_test_bin_number))

        nprop_stars_mergers=len(mergerfile.names_rec)
        integer_nprop_stars_merge=int(nprop_stars_mergers)
        merged_stars_array = np.zeros((integer_nprop_stars_merge,integer_test_bin_stars_number))

        # Multiple AGN episodes:
        # If you want to use the output of a previous AGN simulation as an input to another AGN phase
        # Make sure you have a file 'recipes/prior_model_name_population.dat' so that ReadInputs can take it in
        # and in your .ini file set switch prior_agn = 1.0.
        # Initial orb ecc is prior_ecc_factor*uniform[0,0.99]=[0,0.33] for prior_ecc_factor=0.3 (default)
        if opts.prior_agn == 1.0:
            
            prior_radii, prior_masses, prior_spins, prior_spin_angles, prior_gens \
                = ReadInputs.ReadInputs_prior_mergers()
            num_of_progrades = prograde_bh_locations.size
            prior_indices = setupdiskblackholes.setup_prior_blackholes_indices(rng,num_of_progrades,prior_radii)
            prior_indices = prior_indices.astype('int32') 
            prograde_bh_locations = prior_radii[prior_indices]
            prograde_bh_masses = prior_masses[prior_indices]
            prograde_bh_spins = prior_spins[prior_indices]
            prograde_bh_spin_angles = prior_spin_angles[prior_indices]
            prograde_bh_generations = prior_gens[prior_indices] 
            prior_ecc_factor = 0.3
            prograde_bh_orb_ecc = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform_modified(rng,prior_ecc_factor,num_of_progrades)

        # Start Loop of Timesteps
        print("Start Loop!")
        time_passed = initial_time
        print("Initial Time(yrs) = ",time_passed)

        n_its = 0
        n_mergers_so_far = 0
        n_stars_mergers_so_far = 0
        n_timestep_index = 0
        n_merger_limit = 1e4

        while time_passed < final_time:
            # Record 
            if not(opts.no_snapshots):
                n_bh_out_size = len(prograde_bh_locations)
                n_stars_out_size = len(prograde_stars.orbit_a)

                #svals = list(map( lambda x: x.shape,[prograde_bh_locations, prograde_bh_masses, prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_orb_ecc, prograde_bh_generations[:n_bh_out_size]]))
                # Single output:  does work
                np.savetxt(
                    os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_bh_single_{n_timestep_index}.dat"),
                    np.c_[prograde_bh_locations.T, prograde_bh_masses.T, prograde_bh_spins.T, prograde_bh_spin_angles.T, prograde_bh_orb_ecc.T, prograde_bh_generations[:n_bh_out_size].T],
                    header="r_bh m a theta ecc gen"
                )
                # np.savetxt(os.path.join(work_directory, "output_bh_single_{}.dat".format(n_timestep_index)), np.c_[prograde_bh_locations.T, prograde_bh_masses.T, prograde_bh_spins.T, prograde_bh_spin_angles.T, prograde_bh_orb_ecc.T, prograde_bh_generations[:n_bh_out_size].T], header="r_bh m a theta ecc gen")
                # Binary output: does not work
                np.savetxt(
                    os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_bh_binary_{n_timestep_index}.dat"),
                    binary_bh_array[:,:n_mergers_so_far+1].T,
                    header=binary_field_names
                )

                #Save star params
                prograde_stars.to_file(os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_stars_single_{n_timestep_index}.dat"))

                # np.savetxt(os.path.join(work_directory, "output_bh_single_{}.dat".format(n_timestep_index)), np.c_[prograde_bh_locations.T, prograde_bh_masses.T, prograde_bh_spins.T, prograde_bh_spin_angles.T, prograde_bh_orb_ecc.T, prograde_bh_generations[:n_bh_out_size].T], header="r_bh m a theta ecc gen")
                # Binary output: does not work
                np.savetxt(
                    os.path.join(opts.work_directory, f"run{iteration_zfilled_str}/output_stars_binary_{n_timestep_index}.dat"),
                    binary_stars_array[:,:n_stars_mergers_so_far+1].T,
                    header=binary_stars_field_names
                )
                # np.savetxt(os.path.join(work_directory, "output_bh_binary_{}.dat".format(n_timestep_index)), binary_bh_array[:,:n_mergers_so_far+1].T, header=binary_field_names)
                n_timestep_index +=1

            #Order of operations:        
            # No migration until orbital eccentricity damped to e_crit 
            # 1. check orb. eccentricity to see if any prograde_bh_location BH have orb. ecc. <e_crit.
            #    Create array prograde_bh_location_ecrit for those (mask prograde_bh_locations?)
            #       If yes, migrate those BH.
            #       All other BH, damp ecc and spin *down* BH (retrograde accretion), accrete mass.
            # 2. Run close encounters only on those prograde_bh_location_ecrit members.
        
            # Migrate
            # First if feedback present, find ratio of feedback heating torque to migration torque
            if opts.feedback > 0:
                ratio_heat_mig_torques = feedback_hankla21.feedback_hankla(
                    prograde_bh_locations, surf_dens_func, opts.frac_Eddington_ratio, opts.alpha)
            else:
                ratio_heat_mig_torques = np.ones(len(prograde_bh_locations))

            #now for stars
            if opts.feedback > 0:
                ratio_heat_mig_stars_torques = feedback_hankla21.feedback_hankla(
                    prograde_stars.orbit_a, surf_dens_func, opts.frac_star_Eddington_ratio, opts.alpha)
            else:
                ratio_heat_mig_stars_torques = np.ones(len(prograde_stars.orbit_a))
            # then migrate as usual
            
            prograde_bh_locations = type1.type1_migration(
                opts.mass_smbh,
                prograde_bh_locations,
                prograde_bh_masses,
                disk_surface_density,
                disk_aspect_ratio,
                opts.timestep,
                ratio_heat_mig_torques,
                opts.trap_radius,
                prograde_bh_orb_ecc,
                opts.crit_ecc
            )
            
            # Accrete
            prograde_bh_masses = changebhmass.change_mass(
                prograde_bh_masses,
                opts.frac_Eddington_ratio,
                mass_growth_Edd_rate,
                opts.timestep
            )
            # Spin up
            prograde_bh_spins = changebh.change_spin_magnitudes(
                prograde_bh_spins,
                opts.frac_Eddington_ratio,
                opts.spin_torque_condition,
                opts.timestep,
                prograde_bh_orb_ecc,
                opts.crit_ecc,
            )
            
            
            # Torque spin angle
            prograde_bh_spin_angles = changebh.change_spin_angles(
                prograde_bh_spin_angles,
                opts.frac_Eddington_ratio,
                opts.spin_torque_condition,
                spin_minimum_resolution,
                opts.timestep,
                prograde_bh_orb_ecc,
                opts.crit_ecc
            )

            # Damp BH orbital eccentricity
            prograde_bh_orb_ecc = orbital_ecc.orbital_ecc_damping(
                opts.mass_smbh,
                prograde_bh_locations,
                prograde_bh_masses,
                surf_dens_func,
                aspect_ratio_func,
                prograde_bh_orb_ecc,
                opts.timestep,
                opts.crit_ecc,
            )
            # Perturb eccentricity via dynamical encounters
            if opts.dynamic_enc > 0:
                prograde_bh_locn_orb_ecc = dynamics.circular_singles_encounters_prograde(
                    rng,
                    opts.mass_smbh,
                    prograde_bh_locations,
                    prograde_bh_masses,
                    surf_dens_func,
                    aspect_ratio_func,
                    prograde_bh_orb_ecc,
                    opts.timestep,
                    opts.crit_ecc,
                    opts.de,
                )
                prograde_bh_locations = prograde_bh_locn_orb_ecc[0]
                prograde_bh_orb_ecc = prograde_bh_locn_orb_ecc[1]
                prograde_bh_locations = prograde_bh_locations[0]
                prograde_bh_orb_ecc = prograde_bh_orb_ecc[0]
            
            # Do things to the binaries--first check if there are any:
            if bin_index > 0:
                #First check that binaries are real. Discard any columns where the location or the mass is 0.
                reality_flag = evolve.reality_check(binary_bh_array, bin_index,integer_nbinprop)
                if reality_flag >= 0:
                   #One of the key parameter (mass or location is zero). Not real. Delete binary. Remove column at index = ionization_flag
                    binary_bh_array = np.delete(binary_bh_array,reality_flag,1) 
                    bin_index = bin_index - 1
                else:
                #If there are still binaries after this, evolve them.
                #if bin_index > 0:
                    # If there are binaries, evolve them
                    #Damp binary orbital eccentricity
                    binary_bh_array = orbital_ecc.orbital_bin_ecc_damping(
                        opts.mass_smbh,
                        binary_bh_array,
                        disk_surface_density,
                        disk_aspect_ratio,
                        opts.timestep,
                        opts.crit_ecc
                    )
                    if (opts.dynamic_enc > 0):
                    # Harden/soften binaries via dynamical encounters
                    #Harden binaries due to encounters with circular singletons (e.g. Leigh et al. 2018)
                        binary_bh_array = dynamics.circular_binaries_encounters_circ_prograde(
                            rng,
                            opts.mass_smbh,
                            prograde_bh_locations,
                            prograde_bh_masses,
                            prograde_bh_orb_ecc ,
                            opts.timestep,
                            opts.crit_ecc,
                            opts.de,
                            binary_bh_array,
                            bin_index
                        )

                        #Soften/ ionize binaries due to encounters with eccentric singletons
                        binary_bh_array = dynamics.circular_binaries_encounters_ecc_prograde(
                            rng,
                            opts.mass_smbh,
                            prograde_bh_locations,
                            prograde_bh_masses,
                            prograde_bh_orb_ecc ,
                            opts.timestep,
                            opts.crit_ecc,
                            opts.de,
                            binary_bh_array,
                            bin_index
                        ) 
                    # Harden binaries via gas
                    #Choose between Baruteau et al. 2011 gas hardening, or gas hardening from LANL simulations. To do: include dynamical hardening/softening from encounters
                    binary_bh_array = baruteau11.bin_harden_baruteau(
                        binary_bh_array,
                        integer_nbinprop,
                        opts.mass_smbh,
                        opts.timestep,
                        norm_t_gw,
                        bin_index,
                        time_passed,
                    )
                    #print("Harden binary")
                    #Check closeness of binary. Are black holes at merger condition separation
                    binary_bh_array = evolve.contact_check(binary_bh_array, bin_index, opts.mass_smbh)
                    #print("Time passed = ", time_passed)
                    # Accrete gas onto binary components
                    binary_bh_array = evolve.change_bin_mass(
                        binary_bh_array,
                        opts.frac_Eddington_ratio,
                        mass_growth_Edd_rate,
                        opts.timestep,
                        integer_nbinprop,
                        bin_index
                    )
                    # Spin up binary components
                    binary_bh_array = evolve.change_bin_spin_magnitudes(
                        binary_bh_array,
                        opts.frac_Eddington_ratio,
                        opts.spin_torque_condition,
                        opts.timestep,
                        integer_nbinprop,
                        bin_index
                    )
                    # Torque angle of binary spin components
                    binary_bh_array = evolve.change_bin_spin_angles(
                        binary_bh_array,
                        opts.frac_Eddington_ratio,
                        opts.spin_torque_condition,
                        spin_minimum_resolution,
                        opts.timestep,
                        integer_nbinprop,
                        bin_index
                    )

                    if (opts.dynamic_enc > 0):
                        #Spheroid encounters
                        binary_bh_array = dynamics.bin_spheroid_encounter(
                            rng,
                            opts.mass_smbh,
                            opts.timestep,
                            binary_bh_array,
                            time_passed,
                            bin_index,
                            opts.mbh_powerlaw_index,
                            opts.mode_mbh_init,
                            opts.de,
                            opts.sph_norm
                        )

                    if (opts.dynamic_enc > 0):
                        #Recapture bins out of disk plane
                        binary_bh_array = dynamics.bin_recapture(
                            bin_index,
                            binary_bh_array,
                            opts.timestep
                        )    
                    #Migrate binaries
                    # First if feedback present, find ratio of feedback heating torque to migration torque
                    #print("feedback",feedback)
                    if opts.feedback > 0:
                        ratio_heat_mig_torques_bin_com = evolve.com_feedback_hankla(
                            binary_bh_array,
                            surf_dens_func,
                            opts.frac_Eddington_ratio,
                            opts.alpha
                        )
                    else:
                        ratio_heat_mig_torques_bin_com = np.ones(len(binary_bh_array[9,:]))   

                    # Migrate binaries center of mass
                    binary_bh_array = evolve.bin_migration(
                        opts.mass_smbh,
                        binary_bh_array,
                        disk_surface_density,
                        disk_aspect_ratio,
                        opts.timestep,
                        ratio_heat_mig_torques_bin_com,
                        opts.trap_radius,
                        opts.crit_ecc
                    )

                    # Test to see if any binaries separation is O(1r_g)
                    # If so, track them for GW freq, strain.
                    #Minimum BBH separation (in units of r_g)
                    min_bbh_gw_separation = 2.0
                    # If there are binaries AND if any separations are < min_bbh_gw_separation
                    bbh_gw_indices = np.where( (binary_bh_array[8,:] < min_bbh_gw_separation) & (binary_bh_array[8,:]>0))
                    #print("bbh_gw_indices",bbh_gw_indices)
                    # If bbh_indices exists (ie is not empty)
                    if bbh_gw_indices:
                        
                        num_bbh_gw_tracked = np.size(bbh_gw_indices,1)
                        #print("N_tracked",num_bbh_gw_tracked)
                        nbbhgw = nbbhgw + num_bbh_gw_tracked
                        #if num_bbh_gw_tracked:
                        #    print("num_bbh_gw_tracked",num_bbh_gw_tracked)
                        #print("gw indices",bbh_gw_indices,binary_bh_array[8,bbh_gw_indices])
                        bbh_gw_strain,bbh_gw_freq = evolve.bbh_gw_params(
                            binary_bh_array, 
                            bbh_gw_indices,
                            opts.mass_smbh
                        )
                        #print("BBH strain, freq",bbh_gw_strain,bbh_gw_freq)
                        #print("num tracked",num_bbh_gw_tracked)
                        if num_bbh_gw_tracked == 1:        
                            index = bbh_gw_indices[0]
                            #print("index",index)
                            # If index is empty (=[]) then assume we're tracking 1 BBH only, i.e. the 0th element.
                            #if not index:
                            #   index = 0
                               #print("actual index used",index)

                            temp_bbh_gw_array[0] = iteration
                            temp_bbh_gw_array[1] = time_passed
                            temp_bbh_gw_array[2] = binary_bh_array[8,index]
                            temp_bbh_gw_array[3] = binary_bh_array[2,index] + binary_bh_array[3,index]
                            temp_bbh_gw_array[4] = binary_bh_array[13,index]
                            temp_bbh_gw_array[5] = bbh_gw_strain
                            temp_bbh_gw_array[6] = bbh_gw_freq
                            #temp_bbh_gw_array[2] = binary_bh_array[8,index][0]
                            #temp_bbh_gw_array[3] = binary_bh_array[2,index][0] + binary_bh_array[3,index][0]
                            #temp_bbh_gw_array[4] = binary_bh_array[13,index][0]
                            #temp_bbh_gw_array[5] = bbh_gw_strain[0]
                            #temp_bbh_gw_array[6] = bbh_gw_freq[0]
                            #print("temp_bbh_gw_array",temp_bbh_gw_array)
                            bbh_gw_array = np.vstack((bbh_gw_array,temp_bbh_gw_array))
                            
                        if num_bbh_gw_tracked > 1:
                            index = 0
                            for i in range(0,num_bbh_gw_tracked-1):
                                #print("num_gw_tracked",num_bbh_gw_tracked)
                                #print("i,bbh_gw_indices",i,bbh_gw_indices,bbh_gw_indices[0],bbh_gw_strain,bbh_gw_strain[i])
                                
                                index = bbh_gw_indices[0][i]
                            
                                #Record: iteration, time_passed, bin sep, bin_mass, bin_ecc(around c.o.m.),bin strain, bin freq       
                                temp_bbh_gw_array[0] = iteration
                                temp_bbh_gw_array[1] = time_passed
                                temp_bbh_gw_array[2] = binary_bh_array[8,index]
                                temp_bbh_gw_array[3] = binary_bh_array[2,index] + binary_bh_array[3,index]
                                temp_bbh_gw_array[4] = binary_bh_array[13,index]
                                temp_bbh_gw_array[5] = bbh_gw_strain[i]
                                temp_bbh_gw_array[6] = bbh_gw_freq[i]
                                #print("temp_bbh_gw_array",temp_bbh_gw_array)
                                bbh_gw_array = np.vstack((bbh_gw_array,temp_bbh_gw_array))
                                #print("bbh_gw_array",bbh_gw_array)
                            
                    
                    #Evolve GW frequency and strain
                    binary_bh_array = evolve.evolve_gw(
                        binary_bh_array,
                        bin_index,
                        opts.mass_smbh
                    )
                    
                    #Check and see if merger flagged during hardening (row 11, if negative)
                    merger_flags = binary_bh_array[11,:]
                    any_merger = np.count_nonzero(merger_flags)

                    # Check and see if binary ionization flag raised. 
                    ionization_flag = evolve.ionization_check(binary_bh_array, bin_index, opts.mass_smbh)
                    # Default is ionization flag = -1
                    # If ionization flag >=0 then ionize bin_array[ionization_flag,;]
                    if ionization_flag >= 0:
                        #Comment out for now
                        #print("Ionize binary here!")
                        #print("Number of binaries before ionizing",bin_index)
                        #print("Index of binary to be ionized=",ionization_flag )
                        #print("Bin sep.,Bin a_com",binary_bh_array[8,ionization_flag],binary_bh_array[9,ionization_flag])

                        # Append 2 new BH to arrays of single BH locations, masses, spins, spin angles & gens
                        # For now add 2 new orb ecc term of 0.01. TO DO: calculate v_kick and resulting perturbation to orb ecc.
                        new_location_1 = binary_bh_array[0,ionization_flag]
                        new_location_2 = binary_bh_array[1,ionization_flag]
                        new_mass_1 = binary_bh_array[2,ionization_flag]
                        new_mass_2 = binary_bh_array[3,ionization_flag]
                        new_spin_1 = binary_bh_array[4,ionization_flag]
                        new_spin_2 = binary_bh_array[5,ionization_flag]
                        new_spin_angle_1 = binary_bh_array[6,ionization_flag]
                        new_spin_angle_2 = binary_bh_array[7,ionization_flag]
                        new_gen_1 = binary_bh_array[14,ionization_flag]
                        new_gen_2 = binary_bh_array[15,ionization_flag]
                        new_orb_ecc_1 = 0.01
                        new_orb_ecc_2 = 0.01
                        new_orb_inc_1 = 0.0
                        new_orb_inc_2 = 0.0


                        prograde_bh_locations = np.append(prograde_bh_locations,new_location_1)
                        prograde_bh_locations = np.append(prograde_bh_locations,new_location_2)
                        prograde_bh_masses = np.append(prograde_bh_masses,new_mass_1)
                        prograde_bh_masses = np.append(prograde_bh_masses,new_mass_2)
                        prograde_bh_spins = np.append(prograde_bh_spins,new_spin_1)
                        prograde_bh_spins = np.append(prograde_bh_spins,new_spin_2)
                        prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,new_spin_angle_1)
                        prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,new_spin_angle_2)
                        prograde_bh_generations = np.append(prograde_bh_generations,new_gen_1)
                        prograde_bh_generations = np.append(prograde_bh_generations,new_gen_2)
                        prograde_bh_orb_ecc = np.append(prograde_bh_orb_ecc,new_orb_ecc_1)
                        prograde_bh_orb_ecc = np.append(prograde_bh_orb_ecc,new_orb_ecc_2)
                        prograde_bh_orb_incl = np.append(prograde_bh_orb_incl,new_orb_inc_1)
                        prograde_bh_orb_incl = np.append(prograde_bh_orb_incl,new_orb_inc_2)
                        #Sort new prograde bh_locations
                        sorted_prograde_bh_locations=np.sort(prograde_bh_locations)

                        #Delete binary. Remove column at index = ionization_flag
                        binary_bh_array = np.delete(binary_bh_array,ionization_flag,1)
                        #Reduce number of binaries
                        bin_index = bin_index - 1
                        #Comment out for now
                        #print("Number of binaries remaining", bin_index)

                    #Test dynamics of encounters between binaries and eccentric singleton orbiters
                    #dynamics_binary_array = dynamics.circular_binaries_encounters_prograde(rng,opts.mass_smbh, prograde_bh_locations, prograde_bh_masses, disk_surf_model, disk_aspect_ratio_model, bh_orb_ecc, timestep, opts.crit_ecc, opts.de,norm_tgw,bin_array,bindex,integer_nbinprop)         
                
                    if opts.verbose:
                        print(merger_flags)
                    merger_indices = np.where(merger_flags < 0.0)
                    if isinstance(merger_indices,tuple):
                        merger_indices = merger_indices[0]
                    if opts.verbose:
                        print(merger_indices)
                    #print(binary_bh_array[:,merger_indices])
                    if any_merger > 0:
                        for i in range(any_merger):
                            #print("Merger!")
                            # send properties of merging objects to static variable names
                            #mass_1[i] = binary_bh_array[2,merger_indices[i]]
                            #mass_2[i] = binary_bh_array[3,merger_indices[i]]
                            #spin_1[i] = binary_bh_array[4,merger_indices[i]]
                            #spin_2[i] = binary_bh_array[5,merger_indices[i]]
                            #angle_1[i] = binary_bh_array[6,merger_indices[i]]
                            #angle_2[i] = binary_bh_array[7,merger_indices[i]]
                            #bin_ang_mom[i] = binary_bh_array[16,merger_indices]
                            if time_passed <= opts.timestep:
                                print("time_passed,loc1,loc2",time_passed,binary_bh_array[0,merger_indices[i]],binary_bh_array[1,merger_indices[i]])

                        # calculate merger properties
                            merged_mass = tichy08.merged_mass(
                                binary_bh_array[2,merger_indices[i]],
                                binary_bh_array[3,merger_indices[i]],
                                binary_bh_array[4,merger_indices[i]],
                                binary_bh_array[5,merger_indices[i]]
                            )
                            merged_spin = tichy08.merged_spin(
                                binary_bh_array[2,merger_indices[i]],
                                binary_bh_array[3,merger_indices[i]],
                                binary_bh_array[4,merger_indices[i]],
                                binary_bh_array[5,merger_indices[i]],
                                binary_bh_array[16,merger_indices[i]]
                            )
                            merged_chi_eff = chieff.chi_effective(
                                binary_bh_array[2,merger_indices[i]],
                                binary_bh_array[3,merger_indices[i]],
                                binary_bh_array[4,merger_indices[i]],
                                binary_bh_array[5,merger_indices[i]],
                                binary_bh_array[6,merger_indices[i]],
                                binary_bh_array[7,merger_indices[i]],
                                binary_bh_array[16,merger_indices[i]]
                            )
                            merged_chi_p = chieff.chi_p(
                                binary_bh_array[2,merger_indices[i]],
                                binary_bh_array[3,merger_indices[i]],
                                binary_bh_array[4,merger_indices[i]],
                                binary_bh_array[5,merger_indices[i]],
                                binary_bh_array[6,merger_indices[i]],
                                binary_bh_array[7,merger_indices[i]],
                                binary_bh_array[16,merger_indices[i]],
                                binary_bh_array[17,merger_indices[i]]
                            )
                            merged_bh_array[:,n_mergers_so_far + i] = mergerfile.merged_bh(
                                merged_bh_array,
                                binary_bh_array,
                                merger_indices,
                                i,
                                merged_chi_eff,
                                merged_mass,
                                merged_spin,
                                nprop_mergers,
                                n_mergers_so_far,
                                merged_chi_p,
                                time_passed
                            )
                        #    print("Merger properties (M_f,a_f,Chi_eff,Chi_p,theta1,theta2", merged_mass, merged_spin, merged_chi_eff, merged_chi_p,binary_bh_array[6,merger_indices[i]], binary_bh_array[7,merger_indices[i]],)
                        # do another thing
                        merger_array[:,merger_indices] = binary_bh_array[:,merger_indices]
                        #Reset merger marker to zero
                        #n_mergers_so_far=int(number_of_mergers)
                        #Remove merged binary from binary array. Delete column where merger_indices is the label.
                        #print("!Merger properties!",binary_bh_array[:,merger_indices],merger_array[:,merger_indices],merged_bh_array)
                        binary_bh_array=np.delete(binary_bh_array,merger_indices,1)
                
                        #Reduce number of binaries by number of mergers
                        bin_index = bin_index - len(merger_indices)
                        #print("bin index",bin_index)
                        #Find relevant properties of merged BH to add to single BH arrays
                        num_mergers_this_timestep = len(merger_indices)
                
                        #print("num mergers this timestep",num_mergers_this_timestep)
                        #print("n_mergers_so_far",n_mergers_so_far)    
                        for i in range (0, num_mergers_this_timestep):
                            merged_bh_com = merged_bh_array[0,n_mergers_so_far + i]
                            merged_mass = merged_bh_array[1,n_mergers_so_far + i]
                            merged_spin = merged_bh_array[3,n_mergers_so_far + i]
                            merged_spin_angle = merged_bh_array[4,n_mergers_so_far + i]
                        #New bh generation is max of generations involved in merger plus 1
                            merged_bh_gen = np.maximum(merged_bh_array[11,n_mergers_so_far + i],merged_bh_array[12,n_mergers_so_far + i]) + 1.0 
                        #print("Merger at=",merged_bh_com,merged_mass,merged_spin,merged_spin_angle,merged_bh_gen)
                        # Add to number of mergers
                        n_mergers_so_far += len(merger_indices)
                        number_of_mergers += len(merger_indices)

                        # Append new merged BH to arrays of single BH locations, masses, spins, spin angles & gens
                        # For now add 1 new orb ecc term of 0.01. TO DO: calculate v_kick and resulting perturbation to orb ecc.
                        prograde_bh_locations = np.append(prograde_bh_locations,merged_bh_com)
                        prograde_bh_masses = np.append(prograde_bh_masses,merged_mass)
                        prograde_bh_spins = np.append(prograde_bh_spins,merged_spin)
                        prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,merged_spin_angle)
                        prograde_bh_generations = np.append(prograde_bh_generations,merged_bh_gen)
                        prograde_bh_orb_ecc = np.append(prograde_bh_orb_ecc,0.01)
                        prograde_bh_orb_incl = np.append(prograde_bh_orb_incl,0.0)
                        sorted_prograde_bh_locations=np.sort(prograde_bh_locations)
                        if opts.verbose:
                            print("New BH locations", sorted_prograde_bh_locations)
                        #print("Merger Flag!")
                        #print(number_of_mergers)
                        #print("Time ", time_passed)
                        if opts.verbose:
                            print(merger_array)
                    else:                
                        # No merger
                        # do nothing! hardening should happen FIRST (and now it does!)
                        if opts.verbose:
                            if bin_index>0: # verbose:
                                #print(" BH binaries ", bin_index,  binary_bh_array[:,:int(bin_index)].shape)
                                print(binary_bh_array[:,:int(bin_index)].T)  # this makes printing work as expected
            else:            
                    if opts.verbose:
                        print("No binaries formed yet")
                    # No Binaries present in bin_array. Nothing to do.
                #Finished evolving binaries

                #If a close encounter within mutual Hill sphere add a new Binary

                # check which binaries should get made
            close_encounters2 = hillsphere.binary_check2(
                prograde_bh_locations, prograde_bh_masses, opts.mass_smbh, prograde_bh_orb_ecc, opts.crit_ecc
            )
                #print("Output of close encounters", close_encounters2)
                # print(close_encounters)
            if np.size(close_encounters2) > 0:
                    #print("Make binary at time ", time_passed)
                    #print("shape1",np.shape(close_encounters2)[1])
                    #print("shape0",np.shape(close_encounters2)[0])
                    # number of new binaries is length of 2nd dimension of close_encounters2
                    #number_of_new_bins = np.shape(close_encounters2)[1]
                    number_of_new_bins = np.shape(close_encounters2)[1]
                    #print("number of new bins", number_of_new_bins)
                    # make new binaries
                    binary_bh_array = add_new_binary.add_to_binary_array2(
                        rng,
                        binary_bh_array,
                        prograde_bh_locations,
                        prograde_bh_masses,
                        prograde_bh_spins,
                        prograde_bh_spin_angles,
                        prograde_bh_generations,
                        close_encounters2,
                        bin_index,
                        opts.retro,
                        opts.mass_smbh,
                    )
                    bin_index = bin_index + number_of_new_bins
                    #Count towards total of any binary ever made (including those that are ionized)
                    nbin_ever_made_index = nbin_ever_made_index + number_of_new_bins
                    #print("Binary array",binary_bh_array[:,0])
                    # delete corresponding entries for new binary members from singleton arrays
                    prograde_bh_locations = np.delete(prograde_bh_locations, close_encounters2)
                    prograde_bh_masses = np.delete(prograde_bh_masses, close_encounters2)
                    prograde_bh_spins = np.delete(prograde_bh_spins, close_encounters2)
                    prograde_bh_spin_angles = np.delete(prograde_bh_spin_angles, close_encounters2)
                    prograde_bh_generations = np.delete(prograde_bh_generations, close_encounters2)
                    prograde_bh_orb_ecc = np.delete(prograde_bh_orb_ecc, close_encounters2)
                    prograde_bh_orb_incl = np.delete(prograde_bh_orb_incl, close_encounters2)
            
                    #Empty close encounters
                    empty = []
                    close_encounters2 = np.array(empty)

            #After this time period, was there a disk capture via orbital grind-down?
            # To do: What eccentricity do we want the captured BH to have? Right now ecc=0.0? Should it be ecc<h at a?             
            # Assume 1st gen BH captured and orb ecc =0.0
            # To do: Bias disk capture to more massive BH!
            capture = time_passed % opts.capture_time
            if capture == 0:
                bh_capture_location = setupdiskblackholes.setup_disk_blackholes_location(
                    rng, 1, opts.outer_capture_radius)
                bh_capture_mass = setupdiskblackholes.setup_disk_blackholes_masses(
                    rng, 1, opts.mode_mbh_init, opts.max_initial_bh_mass, opts.mbh_powerlaw_index)
                bh_capture_spin = setupdiskblackholes.setup_disk_blackholes_spins(
                    rng, 1, opts.mu_spin_distribution, opts.sigma_spin_distribution)
                bh_capture_spin_angle = setupdiskblackholes.setup_disk_blackholes_spin_angles(
                    rng, 1, bh_capture_spin)
                bh_capture_gen = 1
                bh_capture_orb_ecc = 0.0
                bh_capture_orb_incl = 0.0
                #print("CAPTURED BH",bh_capture_location,bh_capture_mass,bh_capture_spin,bh_capture_spin_angle)
                # Append captured BH to existing singleton arrays. Assume prograde and 1st gen BH.
                prograde_bh_locations = np.append(prograde_bh_locations,bh_capture_location) 
                prograde_bh_masses = np.append(prograde_bh_masses,bh_capture_mass)
                prograde_bh_spins = np.append(prograde_bh_spins,bh_capture_spin)
                prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,bh_capture_spin_angle) 
                prograde_bh_generations = np.append(prograde_bh_generations,bh_capture_gen)
                prograde_bh_orb_ecc = np.append(prograde_bh_orb_ecc,bh_capture_orb_ecc)
                prograde_bh_orb_incl = np.append(prograde_bh_orb_incl,bh_capture_orb_incl)
            
            # Test if any BH or BBH are in the danger-zone (<mininum_safe_distance, default =50r_g) from SMBH. 
            # Potential EMRI/BBH EMRIs.
            # Find prograde BH in inner disk. Define inner disk as <=50r_g. 
            # Since a 10Msun BH will decay into a 10^8Msun SMBH at 50R_g in ~38Myr and decay time propto a^4. 
            # e.g at 25R_g, decay time is only 2.3Myr.
            min_safe_distance = 50.0
            inner_disk_indices = np.where( prograde_bh_locations < min_safe_distance)
            #retrograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == -1)
            #print("inner disk indices",inner_disk_indices)
            #print(prograde_bh_locations[inner_disk_indices],np.size(inner_disk_indices))
            if np.size(inner_disk_indices) > 0:
                # Add BH to inner_disk_arrays
                inner_disk_locations = np.append(inner_disk_locations,prograde_bh_locations[inner_disk_indices])
                inner_disk_masses = np.append(inner_disk_masses,prograde_bh_masses[inner_disk_indices])
                inner_disk_spins = np.append(inner_disk_spins,prograde_bh_spins[inner_disk_indices])
                inner_disk_spin_angles = np.append(inner_disk_spin_angles,prograde_bh_spin_angles[inner_disk_indices])
                inner_disk_orb_ecc = np.append(inner_disk_orb_ecc,prograde_bh_orb_ecc[inner_disk_indices])
                inner_disk_orb_inc = np.append(inner_disk_orb_inc,prograde_bh_orb_incl[inner_disk_indices])
                inner_disk_gens = np.append(inner_disk_gens,prograde_bh_generations[inner_disk_indices])
                #Remove BH from prograde_disk_arrays
                prograde_bh_locations = np.delete(prograde_bh_locations,inner_disk_indices)
                prograde_bh_masses = np.delete(prograde_bh_masses, inner_disk_indices)
                prograde_bh_spins = np.delete(prograde_bh_spins, inner_disk_indices)
                prograde_bh_spin_angles = np.delete(prograde_bh_spin_angles, inner_disk_indices)
                prograde_bh_orb_ecc = np.delete(prograde_bh_orb_ecc, inner_disk_indices)
                prograde_bh_orb_incl = np.delete(prograde_bh_orb_incl,inner_disk_indices)
                prograde_bh_generations = np.delete(prograde_bh_generations,inner_disk_indices)
                # Empty disk_indices array
                empty=[]
                inner_disk_indices = np.array(empty)

            if np.size(inner_disk_locations) > 0:
                inner_disk_locations = dynamics.bh_near_smbh(opts.mass_smbh,inner_disk_locations,inner_disk_masses,inner_disk_orb_ecc,opts.timestep)
                emri_gw_strain,emri_gw_freq = evolve.evolve_emri_gw(inner_disk_locations,inner_disk_masses,opts.mass_smbh)
                #print("EMRI gw strain",emri_gw_strain)
                #print("EMRI gw freq",emri_gw_freq)
            
            num_in_inner_disk = np.size(inner_disk_locations)
            nemri = nemri + num_in_inner_disk
            if num_in_inner_disk > 0:
                for i in range(0,num_in_inner_disk):
                    #print(iteration,time_passed,inner_disk_locations[i],inner_disk_masses[i],inner_disk_orb_ecc[i],emri_gw_strain[i],emri_gw_freq[i])        
                    temp_emri_array[0] = iteration
                    temp_emri_array[1] = time_passed
                    temp_emri_array[2] = inner_disk_locations[i]
                    temp_emri_array[3] = inner_disk_masses[i]
                    temp_emri_array[4] = inner_disk_orb_ecc[i]
                    temp_emri_array[5] = emri_gw_strain[i]
                    temp_emri_array[6] = emri_gw_freq[i]
                    #print("temp_emri_array",temp_emri_array)
                    #print("emri_array",emri_array)
                    emri_array = np.vstack((emri_array,temp_emri_array))
                
            # if inner_disk_locations[i] <1R_g then merger!
            inner_disk_index = -2
            num_in_inner_disk = np.size(inner_disk_locations)
            for j in range(0,num_in_inner_disk):
                if inner_disk_locations[j] <= 1.0:
                #    print("EMRI merger!!")
                    inner_disk_index = j
            
            if inner_disk_index > -2:                
                inner_disk_locations = np.delete(inner_disk_locations,inner_disk_index)
                inner_disk_masses = np.delete(inner_disk_masses,inner_disk_index)
                inner_disk_spins = np.delete(inner_disk_spins,inner_disk_index)
                inner_disk_spin_angles = np.delete(inner_disk_spin_angles,inner_disk_index)
                inner_disk_orb_ecc = np.delete(inner_disk_orb_ecc,inner_disk_index)
                inner_disk_orb_inc = np.delete(inner_disk_orb_inc,inner_disk_index)
                inner_disk_gens = np.delete(inner_disk_gens,inner_disk_index)


            #binary_bh_array = dynamics.bbh_near_smbh(opts.mass_smbh,bin_index,binary_bh_array)
            
            #Iterate the time step
            time_passed = time_passed + opts.timestep
            #Print time passed every 10 timesteps for now
            time_iteration_tracker = 10.0*opts.timestep
            if time_passed % time_iteration_tracker == 0:
                print("Time passed=",time_passed)
            n_its = n_its + 1
        #End Loop of Timesteps at Final Time, end all changes & print out results
    
        print("End Loop!")
        print("Final Time (yrs) = ",time_passed)
        if opts.verbose:
            print("BH locations at Final Time")
            print(prograde_bh_locations)
        print("Number of binaries = ",bin_index)
        print("Total number of mergers = ",number_of_mergers)
        print("Mergers", merged_bh_array.shape)
        print("Nbh_disk",n_bh)
    
        total_emris = emri_array.shape[0]
        total_bbh_gws = bbh_gw_array.shape[0]

            
        # Write out all the singletons after AGN episode, so can use this as input to another AGN phase.
        # Want to store [Radius, Mass, Spin mag., Spin. angle, gen.]
        # So num_properties_stored = 5 (for now)
        # Note eccentricity will relax, so ignore. Inclination assumed 0deg.
        
        # Set up array for population that survives AGN episode, so can use as a draw for next episode.
        # Need total of 1) size of prograde_bh_array for number of singles at end of run and 
        # 2) size of bin_array for number of BH in binaries at end of run for
        # number of survivors.
        # print("No. of singles",prograde_bh_locations.shape[0])
        # print("No of bh in bins",2*bin_index)
        # print("binary array",binary_bh_array)
        total_bh_survived = prograde_bh_locations.shape[0] + 2*bin_index
        # print("Total bh=",total_bh_survived)
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
            bin_r1[i] = binary_bh_array[0,i]
            bin_r2[i] = binary_bh_array[1,i]
            bin_m1[i] = binary_bh_array[2,i]
            bin_m2[i] = binary_bh_array[3,i]
            bin_a1[i] = binary_bh_array[4,i]
            bin_a2[i] = binary_bh_array[5,i]
            bin_theta1[i] = binary_bh_array[6,i]
            bin_theta2[i] = binary_bh_array[7,i]
            bin_gen1[i] = binary_bh_array[14,i]
            bin_gen2[i] = binary_bh_array[15,i]

        total_emri_array = np.zeros((total_emris,num_of_emri_properties))
        surviving_bh_array = np.zeros((total_bh_survived,num_properties_stored))
        total_bbh_gw_array = np.zeros((total_bbh_gws,num_of_bbh_gw_properties))
        #print("BH locs,bin_r1,bin_r2",prograde_bh_locations,bin_r1,bin_r2)
        prograde_bh_locations = np.append(prograde_bh_locations,bin_r1)
        prograde_bh_locations = np.append(prograde_bh_locations,bin_r2)
        #print("prograde BH locs =",prograde_bh_locations)
        prograde_bh_masses = np.append(prograde_bh_masses,bin_m1)
        prograde_bh_masses = np.append(prograde_bh_masses,bin_m2)
        prograde_bh_spins = np.append(prograde_bh_spins,bin_a1)
        prograde_bh_spins = np.append(prograde_bh_spins,bin_a2)
        prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,bin_theta1)
        prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,bin_theta2)
        prograde_bh_generations = np.append(prograde_bh_generations,bin_gen1)
        prograde_bh_generations = np.append(prograde_bh_generations,bin_gen2)


        surviving_bh_array[:,0] = prograde_bh_locations
        surviving_bh_array[:,1] = prograde_bh_masses
        surviving_bh_array[:,2] = prograde_bh_spins
        surviving_bh_array[:,3] = prograde_bh_spin_angles
        surviving_bh_array[:,4] = prograde_bh_generations

        total_emri_array = emri_array
        total_bbh_gw_array = bbh_gw_array
        if True and number_of_mergers > 0: #verbose:
                print(merged_bh_array[:,:number_of_mergers].T)

        iteration_save_name = f"run{iteration_zfilled_str}/{opts.fname_output_mergers}"
        np.savetxt(os.path.join(opts.work_directory, iteration_save_name), merged_bh_array[:,:number_of_mergers].T, header=merger_field_names)

        # Add mergers to population array including the iteration number
        iteration_row = np.repeat(iteration, number_of_mergers)
        survivor_row = np.repeat(iteration,num_properties_stored)
        emri_row = np.repeat(iteration,num_of_emri_properties)
        gw_row = np.repeat(iteration,num_of_bbh_gw_properties)
        merged_bh_array_pop.append(np.concatenate((iteration_row[np.newaxis], merged_bh_array[:,:number_of_mergers])).T)
        #surviving_bh_array_pop.append(np.concatenate((survivor_row[np.newaxis], surviving_bh_array[:,:total_bh_survived])).T)
        surviving_bh_array_pop.append(np.concatenate((survivor_row[np.newaxis], surviving_bh_array[:total_bh_survived,:])))
        emris_array_pop.append(np.concatenate((emri_row[np.newaxis],total_emri_array[:,:total_emris])))
        gw_array_pop.append(np.concatenate((gw_row[np.newaxis],total_bbh_gw_array[:,:total_bbh_gws])))
     # save all mergers from Monte Carlo
    merger_pop_field_names = "iter " + merger_field_names # Add "Iter" to field names
    population_header = f"Initial seed: {opts.seed}\n{merger_pop_field_names}" # Include initial seed
    basename, extension = os.path.splitext(opts.fname_output_mergers)
    population_save_name = f"{basename}_population{extension}"
    survivors_save_name = f"{basename}_survivors{extension}"
    emris_save_name = f"{basename}_emris{extension}"
    gws_save_name = f"{basename}_lvk{extension}"
    np.savetxt(os.path.join(opts.work_directory, population_save_name), np.vstack(merged_bh_array_pop), header=population_header)
    np.savetxt(os.path.join(opts.work_directory, survivors_save_name), np.vstack(surviving_bh_array_pop))
    np.savetxt(os.path.join(opts.work_directory,emris_save_name),np.vstack(emris_array_pop))
    np.savetxt(os.path.join(opts.work_directory,gws_save_name),np.vstack(gw_array_pop))
if __name__ == "__main__":
    main()
