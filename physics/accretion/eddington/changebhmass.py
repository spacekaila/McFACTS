import numpy as np

def change_mass(prograde_bh_masses,frac_Eddington_ratio,mass_growth_Edd_rate,timestep):
#Return new updated mass array due to accretion for prograde orbiting BH after timestep
    bh_new_masses=prograde_bh_masses*np.exp(mass_growth_Edd_rate*frac_Eddington_ratio*timestep)
    return bh_new_masses
