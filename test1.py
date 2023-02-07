from cgi import print_arguments
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

#from inputs.nsc import nscmodel
#from inputs.disk import sirkogoodman03
#from inputs.smbh import smbhmass

from setup import setupdiskblackholes
from physics.migration.type1 import type1
from physics.accretion.eddington import changebhmass
from physics.accretion.torque import changebh
#from physics.feedback import hankla22
#from physics.dynamics import wang22
from physics.binary.formation import hillsphere
from physics.binary.formation import add_new_binary
#from physics.binary.formation import secunda20
#from physics.binary.harden import baruteau11
from physics.binary.merge import tichy08
from physics.binary.merge import chieff


def main():
    """
    """

    #1. Test a merger by calling modules
    print("Test merger")
    mass_1 = 10.0
    mass_2 = 15.0
    spin_1 = 0.1
    spin2 = 0.7
    angle_1 = 1.80
    angle2 = 0.7
    bin_ang_mom = 1.0
    outmass = tichy08.merged_mass(mass_1, mass_2, spin_1, spin2)
    outspin = tichy08.merged_spin(mass_1, mass_2, spin_1, spin2, bin_ang_mom)
    out_chi = chieff.chi_effective(mass_1, mass_2, spin_1, spin2, angle_1, angle2, bin_ang_mom)
    print(outmass,outspin,out_chi)

    #2. Test set-up; Use a choice of input parameters and call setup modules
    #Mass SMBH (units of Msun)
    mass_smbh = 1.e8
    #Number of BH in disk initially
    n_bh = 50.
    integer_nbh = int(n_bh)
    #Disk outer radius (units of r_g)
    disk_outer_radius = 1.e5
    #Mode of initial BH mass distribution (units of M_sun)
    mode_mbh_init = 10.
    #Maximum of initial BH mass distribution (units of M_sun)
    max_initial_bh_mass = 40.0
    #Pareto(powerlaw) initial BH mass index
    mbh_powerlaw_index = 2.
    #Mean of Gaussian initial spin distribution (zero is good)
    mu_spin_distribution = 0.
    #Sigma of Gaussian initial spin distribution (small is good)
    sigma_spin_distribution = 0.1

    print("Generate initial BH parameter arrays")
    bh_initial_locations = setupdiskblackholes.setup_disk_blackholes_location(n_bh, disk_outer_radius)
    bh_initial_masses = setupdiskblackholes.setup_disk_blackholes_masses(n_bh, mode_mbh_init, max_initial_bh_mass, mbh_powerlaw_index)
    bh_initial_spins = setupdiskblackholes.setup_disk_blackholes_spins(n_bh, mu_spin_distribution, sigma_spin_distribution)
    bh_initial_spin_angles = setupdiskblackholes.setup_disk_blackholes_spin_angles(n_bh, bh_initial_spins)
    bh_initial_orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(n_bh)
       

    #3.a Test migration of prograde BH
    #Disk surface density (assume constant for test)
    disk_surface_density = 1.e5
    #Set up time & number of timesteps
    initial_time = 0.0
    #timestep in years. 10kyr = 1.e4 is reasonable fiducial. 
    timestep = 1.e4
    #For timestep = 1.e4, number_of_timesteps=100 gives us 1Myr total time which is fine to start.
    number_of_timesteps = 20.
    final_time = timestep*number_of_timesteps
    print("Migrate BH in disk")
    #Find prograde BH orbiters. Identify BH with orb. ang mom =+1
    bh_orb_ang_mom_indices = np.array(bh_initial_orb_ang_mom)
    prograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == 1)
    #retrograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == -1)
    prograde_bh_locations = bh_initial_locations[prograde_orb_ang_mom_indices]
    sorted_prograde_bh_locations = np.sort(prograde_bh_locations)
    print("Sorted prograde BH locations:")
    print(sorted_prograde_bh_locations)

    #b. Test accretion onto prograde BH
    #fraction of Eddington ratio accretion (1 is perfectly reasonable fiducial!)
    frac_Eddington_ratio = 1.0
    #Fractional rate of mass growth per year at the Eddington rate(2.3e-8/yr)
    mass_growth_Edd_rate = 2.3e-8
    #Use masses of prograde BH only
    prograde_bh_masses = bh_initial_masses[prograde_orb_ang_mom_indices]
    print("Prograde BH initial masses")
    print(prograde_bh_masses)

    #c. Test spin change and spin angle torquing
    #Spin torque condition. 0.1=10% mass accretion to torque fully into alignment.
    # 0.01=1% mass accretion
    spin_torque_condition = 0.1
    #minimum spin angle resolution (ie less than this value gets fixed to zero) e.g 0.02 rad=1deg
    spin_minimum_resolution = 0.02
    #Torque prograde orbiting BH only
    print("Prograde BH initial spins")
    prograde_bh_spins = bh_initial_spins[prograde_orb_ang_mom_indices]
    print(prograde_bh_spins)
    print("Prograde BH initial spin angles")
    prograde_bh_spin_angles = bh_initial_spin_angles[prograde_orb_ang_mom_indices]
    print(prograde_bh_spin_angles)

    #4 Test Binary formation
    #Number of binary properties that we want to record (e.g. M_1,_2,a_1,2,theta_1,2,a_bin,a_com,t_gw,ecc,bin_ang_mom,generation)
    number_of_bin_properties = 13.0
    integer_nbinprop = int(number_of_bin_properties)
    bin_index = 0
    test_bin_number = 12.0
    integer_test_bin_number = int(test_bin_number)
    #Set up empty initial Binary array
    #Initially all zeros, then add binaries plus details as appropriate
    binary_bh_array = np.zeros((integer_nbinprop, integer_test_bin_number))

    #Start Loop of Timesteps
    print("Start Loop!")
    time_passed = initial_time
    print("Initial Time(yrs) = ",time_passed)
    while time_passed < final_time:
        #Migrate
        prograde_bh_locations = type1.dr_migration(prograde_bh_locations, prograde_bh_masses, disk_surface_density, timestep)
        #Accrete
        prograde_bh_masses = changebhmass.change_mass(prograde_bh_masses, frac_Eddington_ratio, mass_growth_Edd_rate, timestep)
        #Spin up    
        prograde_bh_spins = changebh.change_spin_magnitudes(prograde_bh_spins, frac_Eddington_ratio, spin_torque_condition, timestep)
        #Torque spin angle
        prograde_bh_spin_angles = changebh.change_spin_angles(prograde_bh_spin_angles, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep)
        #Calculate size of Hill sphere
        bh_hill_sphere=hillsphere.calculate_hill_sphere(prograde_bh_locations, prograde_bh_masses, mass_smbh)
        #Test for encounters within Hill sphere
        close_encounters=hillsphere.encounter_test(prograde_bh_locations, bh_hill_sphere)
        #If a close encounter within Hill sphere add a new Binary
        if len(close_encounters) > 0:
            sorted_prograde_bh_locations = np.sort(prograde_bh_locations)
            sorted_prograde_bh_location_indices = np.argsort(prograde_bh_locations)
            number_of_new_bins = (len(close_encounters)+1)/2            
            binary_bh_array = add_new_binary.add_to_binary_array(binary_bh_array, prograde_bh_locations, prograde_bh_masses, prograde_bh_spins, prograde_bh_spin_angles, close_encounters, bin_index)
            bin_index = bin_index + number_of_new_bins
            bh_masses_by_sorted_location = prograde_bh_masses[sorted_prograde_bh_location_indices]
            bh_spins_by_sorted_location = prograde_bh_spins[sorted_prograde_bh_location_indices]
            bh_spin_angles_by_sorted_location = prograde_bh_spin_angles[sorted_prograde_bh_location_indices]
            #Delete binary info from individual BH arrays
            sorted_prograde_bh_locations = np.delete(sorted_prograde_bh_locations, close_encounters)
            bh_masses_by_sorted_location = np.delete(bh_masses_by_sorted_location, close_encounters)
            bh_spins_by_sorted_location = np.delete(bh_spins_by_sorted_location, close_encounters)
            bh_spin_angles_by_sorted_location = np.delete(bh_spin_angles_by_sorted_location, close_encounters)
            #Reset arrays
            prograde_bh_locations = sorted_prograde_bh_locations
            prograde_bh_masses = bh_masses_by_sorted_location
            prograde_bh_spins = bh_spins_by_sorted_location
            prograde_bh_spin_angles = bh_spin_angles_by_sorted_location

        #Iterate the time step
        time_passed = time_passed + timestep
    #End Loop of Timesteps at Final Time, end all changes & print out results
    
    print("End Loop!")
    print("Final Time(yrs) = ",time_passed)
    print("BH locations at Final Time")
    print(prograde_bh_locations)
    print("Binaries")
    print(binary_bh_array)

if __name__ == "__main__":
    main()
