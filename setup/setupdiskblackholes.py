import numpy as np

def setup_disk_blackholes_location(n_bh,disk_outer_radius):
    #Return an array of BH locations distributed randomly uniformly in disk
    rng=np.random.random
    integer_nbh=int(n_bh)
    bh_initial_locations=disk_outer_radius*rng(integer_nbh)
    return bh_initial_locations

def setup_disk_blackholes_masses(n_bh,mode_mbh_init,max_initial_bh_mass,mbh_powerlaw_index):
    #Return an array of BH initial masses for a given powerlaw index and max mass
    integer_nbh=int(n_bh)
    bh_initial_masses=(np.random.pareto(mbh_powerlaw_index,integer_nbh)+1)*mode_mbh_init
    #impose maximum mass condition
    bh_initial_masses[bh_initial_masses>max_initial_bh_mass]=max_initial_bh_mass
    return bh_initial_masses

def setup_disk_blackholes_spins(n_bh,mu_spin_distribution,sigma_spin_distribution):
    #Return an array of BH initial spin magnitudes for a given mode and sigma of a distribution
    integer_nbh=int(n_bh)
    bh_initial_spins=np.random.normal(mu_spin_distribution,sigma_spin_distribution,integer_nbh)
    return bh_initial_spins

def setup_disk_blackholes_spin_angles(n_bh):
    return n_bh
