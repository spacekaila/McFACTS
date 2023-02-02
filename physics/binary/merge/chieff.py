import numpy as np

def chi_effective(mass1,mass2,spin1,spin2,spin_angle1,spin_angle2,bin_ang_mom):
    # Calculate the chi_effective of a merger
    #chi_eff=;chi=(m1*a1*cos(theta_1) + m2*a2*cos(theta_2))/(mbin)
    total_mass=mass1+mass2
    angle1=spin_angle1
    angle2=spin_angle2
    abs_spin1=np.abs(spin1)
    abs_spin2=np.abs(spin2)

# If L_b=-1 then need to measure angles wrt to 3.1415rad not 0 rad.
    if bin_ang_mom == -1:
        angle1=3.1415-spin_angle1
        angle2=3.1415-spin_angle2

# Calculate each component of chi_effective
    chi_factor1=(mass1/total_mass)*abs_spin1*np.cos(angle1)
    chi_factor2=(mass2/total_mass)*abs_spin2*np.cos(angle2)

    chi_eff=chi_factor1+chi_factor2

    return chi_eff
