from cgi import print_arguments
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

#from inputs.nsc import nscmodel
#from inputs.disk import sirkogoodman03
#from inputs.smbh import smbhmass

from setup import setupdiskblackholes
#from physics.migration import type1
#from physics.accretion import eddington
#from physics.feedback import hankla22
#from physics.dynamics import wang22
#from physics.binary.formation import secunda20
#from physics.binary.harden import baruteau11
from physics.binary.merge import tichy08
from physics.binary.merge import chieff

#data=np.loadtxt('inputs.txt',usecols=range(1,21))

def main():
    
    #Test a merger by calling modules
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

    #Test set-up; Use a choice of input parameters and call setup modules
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
    print("Initial locations(r_g) (sorted):")
    bh_initial_locations=setupdiskblackholes.setup_disk_blackholes_location(n_bh,disk_outer_radius)
    sorted_bh_locations=np.sort(bh_initial_locations)
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
    
if __name__ == "__main__":
    main()
