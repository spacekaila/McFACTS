import numpy as np


def chi_effective(mass_1, mass_2, spin_1, spin_2, spin_angle1, spin_angle2, bin_ang_mom):
    # Calculate the chi_effective of a merger
    #chi_eff=;chi=(m1*a1*cos(theta_1) + m2*a2*cos(theta_2))/(mbin)
    total_mass  =mass_1 + mass_2
    angle_1 = spin_angle1
    angle_2 = spin_angle2
    abs_spin1 = np.abs(spin_1)
    abs_spin2 = np.abs(spin_2)

# If L_b=-1 then need to measure angles wrt to 3.1415rad not 0 rad.
    if not(isinstance(bin_ang_mom, np.ndarray)):
           if bin_ang_mom == -1:
               angle_1 = np.pi - spin_angle1
               angle_2 = np.pi - spin_angle2
    else:
        indx_swap = np.where(bin_ang_mom == -1)[0]
        angle_1[indx_swap] = np.pi - spin_angle1[indx_swap]
        angle_2[indx_swap] = np.pi - spin_angle2[indx_swap]

# Calculate each component of chi_effective
    chi_factor1 = (mass_1/total_mass)*abs_spin1*np.cos(angle_1)
    chi_factor2 = (mass_2/total_mass)*abs_spin2*np.cos(angle_2)

    chi_eff = chi_factor1 + chi_factor2

    return chi_eff
