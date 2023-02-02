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
    print("Play with inputs")
    #test set-up
    print("Generate an initial array of BH")
    n_bh=50.
    disk_outer_radius=1.e5
    rng=np.random.random
    integer_nbh=int(n_bh)
    bh_initial_locations=disk_outer_radius*rng(integer_nbh)
    print(bh_initial_locations)

    #mig_trap_radius=data[17]
    #print(mig_trap_radius)
if __name__ == "__main__":
    main()
