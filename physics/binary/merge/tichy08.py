

def merged_mass(mass_1,mass_2,spin_1,spin_2):
# Calculate the final mass of a merged binary
# Only depends on M1,M2,a1,a2
# Using approximations from Tichy \& Maronetti (2008) where
# m_final=(M1+M2)*[1.0-0.2nu-0.208nu^2(a1+a2)]
# where nu is the symmetric mass ratio or nu=q/((1+q)^2)
    total_bin_mass = mass_1 + mass_2
    total_spin = spin_1 + spin_2
    
    if mass_1 >= mass_2:
        primary = mass_1
        secondary = mass_2
    else:
        primary = mass_2
        secondary = mass_1
    # Mass ratio should always be <1. Include a TEST making sure that's TRUE
    
    mass_ratio = secondary/primary
    nu_factor = (1.0+mass_ratio)**2.0
    nu = mass_ratio/nu_factor
    nusq = nu*nu

    mass_factor = 1.0-(0.2*nu)-(0.208*nusq*total_spin)
    merged_mass = total_bin_mass*mass_factor
    return merged_mass


def merged_spin(mass_1, mass_2, spin_1, spin_2, bin_ang_mom):
# Calculate the spin magnitude of a merged binary.
# Only depends on M1,M2,a1,a2 and the binary ang mom around its center of mass
# Using approximations from Tichy \& Maronetti(2008) where
# a_final=0.686(5.04nu-4.16nu^2) +0.4[a1/((0.632+1/q)^2)+ a2/((0.632+q)^2)]
# where q=m_2/m_1 and nu=q/((1+q)^2)
    if mass_1 >= mass_2:
        primary = mass_1
        secondary = mass_2
    else:
        primary = mass_2
        secondary = mass_1
    mass_ratio = secondary/primary
    inv_mass_ratio = (1.0/mass_ratio)
    nu_factor = (1.0+mass_ratio)**2.0
    nu = mass_ratio/nu_factor
    nusq = nu*nu
    spin1_factor = (0.632+inv_mass_ratio)**(2.0)
    spin2_factor = (0.632+mass_ratio)**2.0
    merged_spin = 0.686*((5.04*nu)-(4.16*nusq))+(0.4*((spin_1/spin1_factor)+(spin_2/spin2_factor)))
    return merged_spin

