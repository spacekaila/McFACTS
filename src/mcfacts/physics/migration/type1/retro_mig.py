import numpy as np
import scipy


def retro_mig(
        mass_smbh, 
        retrograde_bh_locations, 
        retrograde_bh_masses, 
        retrograde_bh_orb_ecc, 
        retrograde_bh_orb_inc, 
        retro_arg_periapse, 
        timestep, 
        disk_surf_model):
    """This function calculates how fast the semi-major axis of a retrograde single orbiter
    changes due to dynamical friction (appropriate for BH, NS, maaaybe WD?--check) using
    Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL). It returns the new locations of the retrograde
    orbiters after 1 timestep. Note we have assumed the masses of the orbiters are
    negligible compared to the SMBH (<1% should be fine).
    
    Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    retrograde_bh_locations : float array
        locations of retrograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    retrograde_bh_masses : float array
        mass of retrograde singleton BH at start of timestep in units of solar masses
    retrograde_bh_orb_ecc : float array
        orbital eccentricity of retrograde singleton BH at start of timestep.
    retrograde_bh_orb_inc : float array
        orbital inclination of retrograde singleton BH at start of timestep.
    retro_arg_periapse : float array
        argument of periapse of retrograde singleton BH at start of timestep.
    timestep : float
        size of timestep in years
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g

    Returns
    -------
    retrograde_bh_new_locations : float array
        locations of retrograde singleton BH at end of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)

    """

    # This should probably be set somewhere for the whole code? But...
    KgPerMsun = 1.99e30

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    mass_smbh = mass_smbh * KgPerMsun  # kg
    semi_maj_axis = retrograde_bh_locations * scipy.constants.G * mass_smbh * KgPerMsun/ scipy.constants.c^2  # m
    retro_mass = retrograde_bh_masses * KgPerMsun  # kg
    omega = retro_arg_periapse  # radians
    ecc = retrograde_bh_orb_ecc  # unitless
    inc = retrograde_bh_orb_inc  # radians
    timestep = timestep * scipy.constants.Julian_year # sec

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt(semi_maj_axis^3/(scipy.constants.G * mass_smbh))
    # semi-latus rectum in units of meters
    semi_lat_rec = semi_maj_axis * (1.0-ecc^2)
    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + ecc^2 + 2.0*ecc*np.cos(omega))
    sigma_minus = np.sqrt(1.0 + ecc^2 - 2.0*ecc*np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc*np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc*np.cos(omega))
    # WZL Eqn 65
    kappa_bar = 0.5 * (np.sqrt(1/eta_plus^7) + np.sqrt(1/eta_minus^7))
    # WZL Eqn 66
    xi_bar = 0.5 * (np.sqrt(sigma_plus^4/eta_plus^13) + np.sqrt(sigma_minus^4/eta_minus^13))
    # WZL Eqn 67
    zeta_bar = xi_bar / kappa_bar
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus/eta_plus^2 + sigma_minus/eta_minus^2)
    # WZL Eqn 72
    #   NOTE: preserved retrograde_bh_locations in r_g to feed to disk_surf_model function
    #   tau in units of sec
    tau_a_dyn = (1.0-ecc^2) * np.sin(inc) * (delta - np.cos(inc))^1.5 \
                * mass_smbh^2 * period / (retro_mass*disk_surf_model(retrograde_bh_locations)*np.pi*semi_lat_rec^2) \
                / (np.sqrt(2)) * kappa_bar * np.abs(np.cos(inc) - zeta_bar)
    
    # assume the fractional change in semi-major axis is the fraction
    #   of tau_a_dyn represented by one timestep; in case of semi-maj axis
    #   always moving inwards (because drag) but will need to be careful
    #   when writing ecc function
    frac_change = timestep / tau_a_dyn

    # this is not actually handled correctly for an array... and also what do we do if location = 0?
    if (frac_change < 1.0):
        retrograde_bh_new_locations = retrograde_bh_locations * (1.0 - frac_change)
    else:
        retrograde_bh_new_locations = 0.0

    return retrograde_bh_new_locations