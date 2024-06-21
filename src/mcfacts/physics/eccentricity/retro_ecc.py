import numpy as np
import scipy

def retro_semi_lat(mass_smbh,retrograde_bh_locations,retrograde_bh_masses,retrograde_bh_orb_ecc,retrograde_bh_orb_inc,retro_arg_periapse,disk_surf_model):
    """This function calculates how fast the semi-latus rectum of a retrograde single orbiter
    changes due to dynamical friction (appropriate for BH, NS, maaaybe WD?--check) using
    Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL). It returns the timescale for the retrograde
    orbiters to change their semi-latus rectum (eqn 70). Note we have assumed the masses of 
    the orbiters are negligible compared to the SMBH (<1% should be fine).

    Funny story: if inc = pi exactly, the semi-latus rectum decay is stupid fast
    due to the sin(inc) in tau_p_dyn. However, if you're just a bit
    away from inc = pi (say, pi - 1e-6--but haven't done thorough param search) 
    you get something like sensible answers.
    So we gotta watch out for this

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
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g

    Returns
    -------
    tau_p_dyn : float array
        timescales for the evolution of the semi-latus rectum of each object
    """
    # This should probably be set somewhere for the whole code? But...
    KgPerMsun = 1.99e30

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    mass_smbh = mass_smbh * KgPerMsun  # kg
    semi_maj_axis = retrograde_bh_locations * scipy.constants.G * mass_smbh \
                    / (scipy.constants.c)**2  # m
    retro_mass = retrograde_bh_masses * KgPerMsun  # kg
    omega = retro_arg_periapse  # radians
    ecc = retrograde_bh_orb_ecc  # unitless
    inc = retrograde_bh_orb_inc  # radians

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt(semi_maj_axis**3/(scipy.constants.G * mass_smbh))
    # semi-latus rectum in units of meters
    semi_lat_rec = semi_maj_axis * (1.0-ecc**2)
    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + ecc**2 + 2.0*ecc*np.cos(omega))
    sigma_minus = np.sqrt(1.0 + ecc**2 - 2.0*ecc*np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc*np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc*np.cos(omega))
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0/eta_plus**15) + np.sqrt(1.0/eta_minus**15))
    # WZL Eqn 63
    xi = 0.5 * (np.sqrt(1.0/eta_plus**13) + np.sqrt(1.0/eta_minus**13))
    # WZL Eqn 64
    zeta = xi / kappa
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus/eta_plus**2 + sigma_minus/eta_minus**2)
    # WZL Eqn 70
    #   NOTE: preserved retrograde_bh_locations in r_g to feed to disk_surf_model function
    #   tau in units of sec
    tau_p_dyn = np.sin(inc) * (delta - np.cos(inc))**1.5 \
                * mass_smbh**2 * period / (retro_mass*disk_surf_model(retrograde_bh_locations)*np.pi*semi_lat_rec**2) \
                / (np.sqrt(2)) * kappa * np.abs(np.cos(inc) - zeta)

    return tau_p_dyn

def retro_ecc(mass_smbh,retrograde_bh_locations,retrograde_bh_masses,retrograde_bh_orb_ecc,retrograde_bh_orb_inc,retro_arg_periapse,timestep,disk_surf_model):
    """This function calculates how fast the eccentricity of a retrograde single orbiter
    changes due to dynamical friction (appropriate for BH, NS, maaaybe WD?--check) using
    Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL). It returns the new eccentricities of the retrograde
    orbiters after 1 timestep. Note we have assumed the masses of the orbiters are
    negligible compared to the SMBH (<1% should be fine).

    Funny story: if inc = pi exactly, the semi-major axis & semi-latus rectum decays are 
    stupid fast--in fact at modest initial a, the decay speed (a_0/tau_a_dyn) is > c, 
    so that's wrong.
    I think this is due to the sin(inc) in tau_a_dyn. However, if you're just a bit
    away from inc = pi (say, pi - 1e-8--but haven't done thorough param search) 
    you get something like sensible answers.
    So we gotta watch out for this

    Also, I think this function will still work fine if you feed it prograde bh
    just change the variable name in the function call... (this is not true for migration)
    ... well, actually, I did test it and it fails for inclinations <0.4pi... why?
    error says it fails around the critical inclination angle test... dunno why...

    Also, I'm not sure if it is physically correct to use this, since you're assuming a
    timescale for semi-major axis change that is slower than a migration timescale???

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
    retrograde_bh_new_ecc : float array
        eccentricities of retrograde singleton BH at end of timestep
    """

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    omega = retro_arg_periapse  # radians
    ecc = retrograde_bh_orb_ecc  # unitless
    inc = retrograde_bh_orb_inc  # radians
    timestep = timestep * scipy.constants.Julian_year # sec

    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + ecc**2 + 2.0*ecc*np.cos(omega))
    sigma_minus = np.sqrt(1.0 + ecc**2 - 2.0*ecc*np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc*np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc*np.cos(omega))
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0/eta_plus**15) + np.sqrt(1.0/eta_minus**15))
    # WZL Eqn 63
    xi = 0.5 * (np.sqrt(1.0/eta_plus**13) + np.sqrt(1.0/eta_minus**13))
    # WZL Eqn 64
    zeta = xi / kappa
    # WZL Eqn 65
    kappa_bar = 0.5 * (np.sqrt(1.0/eta_plus**7) + np.sqrt(1.0/eta_minus**7))
    # WZL Eqn 66
    xi_bar = 0.5 * (np.sqrt(sigma_plus**4/eta_plus**13) + np.sqrt(sigma_minus**4/eta_minus**13))
    # WZL Eqn 67
    zeta_bar = xi_bar / kappa_bar

    # call function for tau_p_dyn (WZL Eqn 70)
    tau_p_dyn = retro_semi_lat(mass_smbh,retrograde_bh_locations,retrograde_bh_masses,retrograde_bh_orb_ecc,retrograde_bh_orb_inc,retro_arg_periapse,disk_surf_model)
    # retro_mig function actually comuptes change in a, so need to find tau_a_dyn again, but
    #   fortunately it's a few factors off of tau_p_dyn (this may be a dumb way to handle it)
    tau_a_dyn = tau_p_dyn * (1.0 - ecc**2) * kappa * np.abs(np.cos(inc) - zeta)/(kappa_bar * np.abs(np.cos(inc) - zeta_bar))
    # WZL Eqn 73
    tau_e_dyn = (2.0 * ecc**2 / (1.0 - ecc**2)) * 1.0 / np.abs(1.0/tau_a_dyn - 1.0/tau_p_dyn)

    # assume the fractional change in eccentricity is the fraction
    #   of tau_e_dyn represented by one timestep
    frac_change = timestep / tau_e_dyn

    # Unlike in the retro_mig function I have not set up a check for when
    #   tau_e_dyn < timestep because it appears that when that happens,
    #   tau_a_dyn is also < timestep, so a gets sent to zero before
    #   we even worry about eccentricity, but, uh, yeah, putting a note here
    #   in case there are exceptions I haven't found and should worry about...

    # need to figure out which way the eccentricity is going; use
    #   Eqn 69 in WZL for cosine of critical inclination
    cos_inc_crit = (xi_bar - (1.0 - ecc**2) * xi)/(kappa_bar - (1.0 - ecc**2) * kappa)
    # if the cos(inc) is less than cos(inc_crit), ecc is excited, else it is damped
    frac_change[np.abs(np.cos(inc)) < cos_inc_crit] = -frac_change[np.abs(np.cos(inc)) < cos_inc_crit]

    # accounting for poss increase OR decrease in ecc by flipping sign 
    #   on frac_change above (where appropriate)
    retrograde_bh_new_ecc = retrograde_bh_orb_ecc * (1.0 - frac_change)

    return retrograde_bh_new_ecc

