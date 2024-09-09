import numpy as np


def chi_effective(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2, bin_ang_momenta):
    """Calculate the effective spin :math:`\chi_{\rm eff}` associated with a merger.

    The measured effective spin of a merger is calculated as

    .. math:: \chi_{\rm eff}=\frac{m_1*\chi_1*\cos(\theta_1) + m_2*\chi_2*\cos(\theta_2)}/{m_{\rm bin}}
    
    Parameters
    ----------
    masses_1 : float/ndarray
        Mass of object 1.
    masses_2 : float/ndarray
        Mass of object 2.
    spins_1 : float/ndarray
        Dimensionless spin magnitude of object 1.
    spins_2 : float/ndarray
        Dimensionless spin magnitude of object 2.
    spin_angles_1 : float/ndarray
        Dimensionless spin angle of object 1.
    spin_angles_2 : float/ndarray
        Dimensionless spin angle of object 2.
    bin_ang_momenta : int/ndarray
        Magnitude of the binary's mutual angular momentum. If 1, the binary
        is prograde (aligned with disk angular momentum). If -1, the binary
        is retrograde (anti-aligned with disk angular momentum).

    Returns
    -------
    float,ndarray
        The effective spin value for this (these) object(s).
    """

    bin_total_masses = masses_1 + masses_2
    angle_1 = spin_angles_1
    angle_2 = spin_angles_2
    spins_abs_1 = np.abs(spins_1)
    spins_abs_2 = np.abs(spins_2)

    # If direction of binary orbital momenta L_b =-1 (retrograde) then need to measure
    #  angles wrt to 3.1415rad not 0 rad.
    # Case for a single binary
    if not isinstance(bin_ang_momenta, np.ndarray):
        if bin_ang_momenta == -1:
            angle_1 = np.pi - spin_angles_1
            angle_2 = np.pi - spin_angles_2
    else: # Case for array of binaries
        indx_swap = np.where(bin_ang_momenta == -1)[0]
        angle_1[indx_swap] = np.pi - spin_angles_1[indx_swap]
        angle_2[indx_swap] = np.pi - spin_angles_2[indx_swap]

    # Calculate each component of chi_effective
    chi_factor1 = (masses_1 / bin_total_masses) * spins_abs_1 * np.cos(angle_1)
    chi_factor2 = (masses_2 / bin_total_masses) * spins_abs_2 * np.cos(angle_2)

    chi_eff = chi_factor1 + chi_factor2

    return chi_eff

def chi_p(mass_1, mass_2, spin_1, spin_2, spin_angle_1, spin_angle_2, bin_ang_mom, bin_inclination_wrt_disk):
    """Calculate the precessing spin component :math:`\chi_p` associated with a merger.
    
    chi_p = max[spin_1_perp, (q(4q+3)/(4+3q))* spin_2_perp]
    
    where 
    
    spin_1_perp = spin_1 * sin(spin_angle_1)
    spin_2_perp = spin_2 * sin(spin_angle_2)
    
    are perpendicular to `spin_1

    and `q=M_2/M_1 where `M_2 < M_1`

    
    Parameters
    ----------
    mass_1 : float/ndarray
        Mass of object 1.
    mass_2 : float/ndarray
        Mass of object 2.
    spin_1 : float/ndarray
        Dimensionless spin magnitude of object 1.
    spin_2 : float/ndarray
        Dimensionless spin magnitude of object 2.
    spin_angle_1 : float/ndarray
        Dimensionless spin angle of object 1.
    spin_angle_2 : float/ndarray
        Dimensionless spin angle of object 2.
    bin_ang_mom : int/ndarray
        Direction of the binary's mutual angular momentum. If 1, the binary
        is prograde (aligned with disk angular momentum). If -1, the binary
        is retrograde (anti-aligned with disk angular momentum).
    bin_inclination_wrt_disk : float/ndarray
        Angle of inclination of the binary with respect to the disk.

    Returns
    -------
    float/ndarray
        _description_
    """

    # If mass1 is the dominant binary partner
    # Define default mass ratio of 1, otherwise choose based on masses
    q = 1.0

    # Define spin angle to include bin_inclination_wrt_disk (all in units of radians)
    spin_angle_1 = spin_angle_1 + bin_inclination_wrt_disk
    spin_angle_2 = spin_angle_2 + bin_inclination_wrt_disk

    # Make sure angles are <pi radians!
    spin_angle_diff_1 = spin_angle_1 - np.pi
    spin_angle_diff_2 = spin_angle_2 - np.pi
    if spin_angle_diff_1 > 0:
        spin_angle_1 = spin_angle_1 - spin_angle_diff_1
    if spin_angle_diff_2 > 0:
        spin_angle_2 = spin_angle_2 - spin_angle_diff_2

    # Define default spins
    spin_1_perp = abs(spin_1) * np.sin(spin_angle_1)
    spin_2_perp = abs(spin_2) * np.sin(spin_angle_2)
    
    if mass_1 > mass_2:
        q = mass_2 / mass_1
        spin_1_perp = abs(spin_1) * np.sin(spin_angle_1)
        spin_2_perp = abs(spin_2) * np.sin(spin_angle_2)
    # If mass2 is the dominant binary partner
    if mass_2 > mass_1:
        q = mass_1 / mass_2
        spin_1_perp = abs(spin_2) * np.sin(spin_angle_2)
        spin_2_perp = abs(spin_1) * np.sin(spin_angle_1)

    q_factor = q * ((4.0 * q) + 3.0)/(4.0 + (3.0 * q))
    
    # Assume spin_1_perp is dominant source of chi_p
    chi_p = spin_1_perp
    # if not then change chi_p definition and output
    if chi_p < q_factor * spin_2_perp:
        chi_p = q_factor * spin_2_perp

    if chi_p < 0:
        print("chi_p,m1,m2,a1,a2,a1p,a2p,theta1,theta2,bin_inc,q_factor=",chi_p,mass_1,mass_2,spin_1,spin_2,abs(spin_1),abs(spin_2),spin_1_perp,spin_2_perp,spin_angle_1,spin_angle_2,bin_inclination_wrt_disk,q_factor)


    return chi_p
