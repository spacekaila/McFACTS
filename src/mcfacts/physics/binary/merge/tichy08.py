
import numpy as np

def merged_mass(masses_1, masses_2, spins_1, spins_2):
# Calculate the final mass of a merged binary
# Only depends on M1,M2,a1,a2
# Using approximations from Tichy \& Maronetti (2008) where
# m_final=(M1+M2)*[1.0-0.2nu-0.208nu^2(a1+a2)]
# where nu is the symmetric mass ratio or nu=q/((1+q)^2)
    # These are arrays
    #mass_1 = -1
    #mass_2 = -1
    #if isinstance(m1, np.ndarray):
    # if len(m1.shape) >0:
    #    mass_1 = m1[0]
    #    mass_2 = m2[0]
    #    total_spin = spin_1[0] + spin_2[0]
    #else:
    #    mass_1 = m1
    #    mass_2 = m2
    #    total_spin = spin_1 + spin_2
    
    primary_masses = np.max(np.c_[masses_1, masses_2])
    secondary_masses = np.min(np.c_[masses_1, masses_2])

    #If issue with primary mass, print!
    if primary_masses < 1.0:
        print("primary,secondary=", primary_masses, secondary_masses)
        mass_ratios = 1.0
    if primary_masses > 1.0:    
        mass_ratios = secondary_masses/primary_masses
    
    total_masses = primary_masses + secondary_masses
    total_spins = spins_1 + spins_2
    nu_factors = (1.0+mass_ratios)**2.0
    nus = mass_ratios/nu_factors
    nus_square = nus*nus
    nus =  (primary_masses* secondary_masses)/ (total_masses)**2 # mass_ratio/nu_factor
    nus_square = nus*nus

    #mass_1 = -1
    #mass_2 = -1
    #if isinstance(m1, np.ndarray):
    # if len(m1.shape) >0:
    #    mass_1 = m1[0]
    #    mass_2 = m2[0]
    #    total_spin = spin_1[0] + spin_2[0]
    #else:
    #    mass_1 = m1
    #    mass_2 = m2
    #    total_mass = mass_1 + mass_2
    #    total_spin=spin_1 + spin_2

    #print(" Mass array ",mass_1 + mass_2)
    #nu =  (mass_1* mass_2)/ (mass_1+mass_2)**2 # mass_ratio/nu_factor
    #nusq = nu*nu

    mass_factors = 1.0-(0.2*nus)-(0.208*nus_square*total_spins)
    merged_masses = (total_masses)*mass_factors
    return merged_masses


def merged_spin(masses_1, masses_2, spins_1, spins_2, bin_ang_mom):
# Calculate the spin magnitude of a merged binary.
# Only depends on M1,M2,a1,a2 and the binary ang mom around its center of mass
# Using approximations from Tichy \& Maronetti(2008) where
# a_final=0.686(5.04nu-4.16nu^2) +0.4[a1/((0.632+1/q)^2)+ a2/((0.632+q)^2)]
# where q=m_2/m_1 and nu=q/((1+q)^2)
    
    primary_masses = np.max(np.c_[masses_1, masses_2])
    secondary_masses = np.min(np.c_[masses_1, masses_2])
    mass_ratios = secondary_masses/primary_masses
    mass_ratios_inv = (1.0/mass_ratios)

    nu_factors = (1.0+mass_ratios)**2.0
    nus = mass_ratios/nu_factors
    nus_square = nus*nus

    spin_factors_1 = (0.632+mass_ratios_inv)**(2.0)
    spin_factors_2 = (0.632+mass_ratios)**2.0

    merged_spins = 0.686*((5.04*nus)-(4.16*nus_square))+(0.4*((spins_1/spin_factors_1)+(spins_2/spin_factors_2)))
    
    return merged_spins

