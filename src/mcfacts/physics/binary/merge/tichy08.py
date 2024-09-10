
import numpy as np

def merged_mass(masses_1, masses_2, spins_1, spins_2):
    """Calculate the final mass of a merged binary.

    Using approximations from Tichy \& Maronetti (2008) where
    m_final=(M1+M2)*[1.0-0.2nu-0.208nu^2(a1+a2)]
    where nu is the symmetric mass ratio or nu=q/((1+q)^2)

    Parameters
    ----------
    masses_1 : float/ndarray
        Masses of objects 1.
    masses_2 : float/ndarray
        Masses of objects 2.
    spins_1 : float/ndarray
        Spin magnitudes of objects 1.
    spins_2 : float/ndarray
        Spin magnitudes of objects 2.

    Returns
    -------
    float/ndarray
        Final mass of merger remnant.
    """

    primary_masses = np.max(np.c_[masses_1, masses_2])
    secondary_masses = np.min(np.c_[masses_1, masses_2])

    #If issue with primary mass, print!
    if primary_masses < 1.0:
        print("primary,secondary=", primary_masses, secondary_masses)
        mass_ratios = 1.0
    if primary_masses > 1.0:
        mass_ratios = secondary_masses / primary_masses

    total_masses = primary_masses + secondary_masses
    total_spins = spins_1 + spins_2
    nu_factors = (1.0 + mass_ratios)**2.0
    nus = mass_ratios / nu_factors
    nus_square = nus * nus

    mass_factors = 1.0-(0.2*nus)-(0.208*nus_square*total_spins)
    merged_masses = (total_masses)*mass_factors
    return merged_masses


def merged_spin(masses_1, masses_2, spins_1, spins_2, bin_ang_mom):
    """Calculate the spin magnitude of a merged binary.

    Only depends on M1,M2,a1,a2 and the binary ang mom around its center of mass.
    Using approximations from Tichy \& Maronetti(2008) where
    a_final=0.686(5.04nu-4.16nu^2) +0.4[a1/((0.632+1/q)^2)+ a2/((0.632+q)^2)]
    where q=m_2/m_1 and nu=q/((1+q)^2)

    Parameters
    ----------
    masses_1 : float/ndarray
        Masses of objects 1.
    masses_2 : float/ndarray
        Masses of objects 2.
    spins_1 : float/ndarray
        Spin magnitudes of objects 1.
    spins_2 : float/ndarray
        Spin magnitudes of objects 2.
    bin_ang_mom : float/ndarray
        Angular momentum magnitudes of the binaries.

    Returns
    -------
    float/ndarray
        Final spin magnitude of merger remnant.
    """

    primary_masses = np.max(np.c_[masses_1, masses_2])
    secondary_masses = np.min(np.c_[masses_1, masses_2])
    mass_ratios = secondary_masses / primary_masses
    mass_ratios_inv = 1.0 / mass_ratios

    nu_factors = (1.0 + mass_ratios)**2.0
    nus = mass_ratios / nu_factors
    nus_square = nus * nus

    spin_factors_1 = (0.632 + mass_ratios_inv)**(2.0)
    spin_factors_2 = (0.632 + mass_ratios)**2.0

    merged_spins = 0.686*((5.04*nus) - (4.16*nus_square)) + \
                   (0.4*((spins_1/spin_factors_1)+(spins_2/spin_factors_2)))

    return merged_spins
