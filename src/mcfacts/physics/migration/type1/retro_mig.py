#this is going to be the \dot a method for retrograde orbiters

import numpy as np
import scipy


def retro_mig(mass_smbh, retrograde_bh_locations, retrograde_bh_masses, retrograde_bh_orb_ecc, retrograde_bh_orb_inc, arg_periapse, timestep):
    """This function calculates how fast the semi-major axis of a retrograde single orbiter
    changes due to dynamical friction (appropriate for BH, NS, maaaybe WD?--check) using
    Wang, Zhu & Lin 2024, MNRAS, 528, 4958. It returns the new locations of the retrograde
    orbiters after 1 timestep.
    
    """

    semi_maj_axis = retrograde_bh_locations
    retro_mass = retrograde_bh_masses
    omega = arg_periapse
    ecc = retrograde_bh_orb_ecc
    inc = retrograde_bh_orb_inc

    period = 2.0 * np.pi * np.sqrt(semi_maj_axis^3/(G*mass_smbh))
    sigma_plus = np.sqrt(1.0 + ecc^2 + 2.0*ecc*np.cos(omega))
    sigma_minus = np.sqrt(1.0 + ecc^2 - 2.0*ecc*np.cos(omega))
    eta_plus = np.sqrt(1.0 + ecc*np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc*np.cos(omega))
    kappa_bar = 0.5 * (np.sqrt(1/eta_plus^7) + np.sqrt(1/eta_minus^7))
    xi_bar = 0.5 * (np.sqrt(sigma_plus^4/eta_plus^13) + np.sqrt(sigma_minus^4/eta_minus^13))
    zeta_bar = xi_bar / kappa_bar
    delta = 0.5 * (sigma_plus/eta_plus^2 + sigma_minus/eta_minus^2)
    tau_a_dyn = (1-ecc^2) * np.sin(inc) * (delta - np.cos(inc))^1.5 \
                * mass_smbh^2 * period / (retro_mass*Sigma*np.pi*semi_lat_rec^2) \
                / (np.sqrt(2)) * kappa_bar * np.abs(np.cos(inc) - zeta_bar)
    
    frac_change = timestep / tau_a_dyn

    retrograde_bh_locations = semi_maj_axis * frac_change

    return retrograde_bh_locations