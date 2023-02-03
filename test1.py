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
#from physics.feedback import hankla22
#from physics.dynamics import wang22
#from physics.binary.formation import secunda20
#from physics.binary.harden import baruteau11
from physics.binary.merge import tichy08
from physics.binary.merge import chieff

#data=np.loadtxt('inputs.txt',usecols=range(1,21))

def main():
    
    #1. Test a merger by calling modules
    print("Do a thing with a merger")
    mass1=10.0
    mass2=15.0
    spin1=0.1
    spin2=0.7
    angle1=1.80
    angle2=0.7
    bin_ang_mom=1.0
    outmass=tichy08.merged_mass(mass1,mass2,spin1,spin2)
    outspin=tichy08.merged_spin(mass1,mass2,spin1,spin2,bin_ang_mom)
    out_chi=chieff.chi_effective(mass1,mass2,spin1,spin2,angle1,angle2,bin_ang_mom)
    print(outmass,outspin,out_chi)

    #2. Test set-up; Use a choice of input parameters and call setup modules
    n_bh=50.
    disk_outer_radius=1.e5
    #Mode of initial BH mass distribution in M_sun
    mode_mbh_init=10.
    #Maximum of initial BH mass distribution in M_sun
    max_initial_bh_mass=40.0
    #Pareto(powerlaw) initial BH mass index
    mbh_powerlaw_index=2.
    #Mean of Gaussian initial spin distribution (zero is good)
    mu_spin_distribution=0.
    #Sigma of Gaussian initial spin distribution (small is good)
    sigma_spin_distribution=0.1

    print("Generate initial BH parameter arrays")
    print("Initial locations(r_g):")
    bh_initial_locations=setupdiskblackholes.setup_disk_blackholes_location(n_bh,disk_outer_radius)
    print(bh_initial_locations)
    sorted_bh_locations=np.sort(bh_initial_locations)
    print("Initial (sorted) locations")
    print(sorted_bh_locations)
    print("Initial masses(Msun):")
    bh_initial_masses=setupdiskblackholes.setup_disk_blackholes_masses(n_bh,mode_mbh_init,max_initial_bh_mass,mbh_powerlaw_index)
    print(bh_initial_masses)
    print("Initial spins:")
    bh_initial_spins=setupdiskblackholes.setup_disk_blackholes_spins(n_bh,mu_spin_distribution,sigma_spin_distribution)
    print(bh_initial_spins)
    print("Initial spin angles (rads)")
    bh_initial_spin_angles=setupdiskblackholes.setup_disk_blackholes_spin_angles(n_bh,bh_initial_spins)
    print(bh_initial_spin_angles)
    print("Initial orbital angular momentum")
    bh_initial_orb_ang_mom=setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(n_bh)
    print(bh_initial_orb_ang_mom)    

    #3.a Test migration of prograde BH
    #Disk surface density (assume constant for test)
    disk_surface_density=1.e5
    #Set up time & number of timesteps
    initial_time=0.0
    #timestep in years. 10kyr=1.e4 is reasonable fiducial. 
    timestep=1.e4
    #For timestep=1.e4, number_of_timesteps=100 gives us 1Myr total time which is fine to start.
    number_of_timesteps=20.
    final_time=timestep*number_of_timesteps
    print("Migrate BH in disk")
    #Find prograde BH orbiters. Identify BH with orb. ang mom =+1
    bh_orb_ang_mom_indices=np.array(bh_initial_orb_ang_mom)
    prograde_orb_ang_mom_indices=np.where(bh_orb_ang_mom_indices == 1)
    #retrograde_orb_ang_mom_indices=np.where(bh_orb_ang_mom_indices == -1)
    prograde_bh_locations=bh_initial_locations[prograde_orb_ang_mom_indices]
    sorted_prograde_bh_locations=np.sort(prograde_bh_locations)
    print("Sorted prograde BH locations:")
    print(sorted_prograde_bh_locations)

    #b. Test accretion onto prograde BH
    #fraction of Eddington ratio accretion (1 is perfectly reasonable fiducial!)
    frac_Eddington_ratio=1.0
    #Fractional rate of mass growth per year at the Eddington rate(2.3e-8/yr)
    mass_growth_Edd_rate=2.3e-8
    #Use masses of prograde BH only
    prograde_bh_masses=bh_initial_masses[prograde_orb_ang_mom_indices]
    print("Prograde BH initial masses")
    print(prograde_bh_masses)

    #c. Test spin change and torquing
    prograde_bh_spins=bh_initial_spins[prograde_orb_ang_mom_indices]
    prograde_bh_spin_angles=bh_initial_spin_angles[prograde_orb_ang_mom_indices]
    
    print("Start Loop!")
    time_passed=initial_time
    print("Initial Time(yrs)=",time_passed)
    while time_passed<final_time:
        bh_locations=type1.dr_migration(prograde_bh_locations,prograde_bh_masses,disk_surface_density,timestep)
        bh_masses=changebhmass.change_mass(prograde_bh_masses,frac_Eddington_ratio,mass_growth_Edd_rate,timestep)
        #Iterate the time step
        time_passed=time_passed+timestep
    print("End Loop!")
    print("Final Time(yrs)=",time_passed)
    print("(Sorted) BH locations at Final Time")
    sorted_final_bh_locations=np.sort(bh_locations)
    print(sorted_final_bh_locations)
    print("BH masses at Final Time")
    print(bh_masses)

if __name__ == "__main__":
    main()
