from cgi import print_arguments
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

#from inputs.nsc import nscmodel
#from inputs.disk import sirkogoodman03
#from inputs.smbh import smbhmass

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

    #test set-up
    n_bh=50.
    
    #Generate some array of random initial BH locations
    print("Generate an initial array of BH locations & sort them")
    disk_outer_radius=1.e5
    rng=np.random.random
    integer_nbh=int(n_bh)
    bh_initial_locations=disk_outer_radius*rng(integer_nbh)
    print(bh_initial_locations)
    sorted_bh_locations=np.sort(bh_initial_locations)
    print(sorted_bh_locations)
    
    #Generate masses for the BH array
    print("Generate an initial array of masses for each BH in locations")
    #Mode of initial BH mass distribution in M_sun
    mode_mbh_init=10.
    #Maximum of initial BH mass distribution in M_sun
    max_initial_bh_mass=40.0
    #Pareto(powerlaw) initial BH mass index
    mbh_powerlaw_index=2.
    bh_initial_masses=(np.random.pareto(mbh_powerlaw_index,integer_nbh)+1)*mode_mbh_init
    #impose maximum mass condition
    bh_initial_masses[bh_initial_masses>max_initial_bh_mass]=max_initial_bh_mass
    print(bh_initial_masses)
    
    #Generate spins for the BH array
    print("Generate an initial array of spins for each BH in location")
    #Mean of Gaussian initial spin distribution (zero is good)
    mu_spin_distribution=0.
    #Sigma of Gaussian initial spin distribution (small is good)
    sigma_spin_distribution=0.1
    bh_initial_spins=np.random.normal(mu_spin_distribution,sigma_spin_distribution,integer_nbh)
    bh_initial_spin_indices=np.array(bh_initial_spins)
    negative_spin_indices=np.where(bh_initial_spin_indices < 0.)
    print(bh_initial_spins)

    #mig_trap_radius=data[17]
    #print(mig_trap_radius)
if __name__ == "__main__":
    main()
