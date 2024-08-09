import numpy as np
import scipy

def crude_retro_bh(mass_smbh, orbiter_masses, orbiter_semimaj, orbiter_ecc, orbiter_inc, orbiter_periapse, surfdensfunc, timestep):
    """To avoid having to install and couple to SpaceHub, and run N-body code
    this is a distinctly half-assed treatment of retrograde orbiters, based
    LOOSELY on Wang, Zhu & Lin 2024 (WZL). Evolving all orbital params simultaneously.
    Using lots of if statements to pretend we're interpolating.
    Hardcoding some stuff from WZL figs 7, 8 & 12 (see comments).
    Arg of periapse = w in comments below

    Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    orbiter_masses : float array
        mass of retrograde singleton BH at start of timestep in units of solar masses
    orbiter_semimaj : float array
        locations of retrograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
        (actually we assume this is the same as the semi-major axis)
    orbiter_ecc : float array
        orbital eccentricity of retrograde singleton BH at start of timestep.
    orbiter_inc : float array
        orbital inclination in radians of retrograde singleton BH at start of timestep.
    orbiter_periapse : float array
        argument of periapse of retrograde singleton BH at start of timestep.
    surfdensfunc : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
    timestep : float
        length of timestep in years

    Returns
    -------
    new_orbiter_ecc, new_orbiter_semimaj, new_orbiter_inc : float arrays
        updated values of eccentricity, semimajor axis (in r_g) and inclination (in radians)
        after one timestep assuming gas only evolution hacked together badly...
    """
    # first handle cos(w)=+/-1 (assume abs(cos(w))>0.5)
    #   this evolution is multistage:
    #       1. radialize, semimaj axis shrinks (slower), flip (very slowly)
    #       2. flip (very fast), circ (slowly), constant semimaj axis
    #       3. i->0.000 (very fast), circ & shrink semimaj axis slightly slower
    #
    #      For mass_smbh=1e8Msun, orbiter_mass=30Msun, SG disk surf dens (fig 12 WZL)
    #       1. in 1.5e5yrs e=0.7->0.9999 (roughly), a=100rg->60rg, i=175->165deg
    #       2. in 1e4yrs i=165->12deg, e=0.9999->0.9, a=60rg
    #       3. in 1e4yrs i=12->0.0deg, e=0.9->0.5, a=60->20rg

    # setup output arrays
    new_orbiter_ecc = np.zeros(len(orbiter_ecc))
    new_orbiter_inc = np.zeros(len(orbiter_inc))
    new_orbiter_semimaj = np.zeros(len(orbiter_semimaj))

    # hardcoded awfulness coming up:
    mass_smbh_0 = 1e8 # solar masses, for scaling
    orbiter_mass_0 = 30.0 # solar masses
    periapse_1 = 0.0 # radians
    periapse_0 = np.pi/2.0 # radians

    step1_ecc_0 = 0.7
    step1_inc_0 = np.pi * (175.0/180.0) # rad
    step1_semi_maj_0 = 100.0 # r_g

    step2_ecc_0 = 0.9999
    step2_inc_0 = np.pi * (165.0/180.0) # rad
    step2_semi_maj_0 = 60.0 # r_g

    step3_ecc_0 = 0.9
    step3_inc_0 = np.pi * (12.0/180.0) # rad
    step3_semi_maj_0 = step2_semi_maj_0 # r_g

    step3_ecc_f = 0.5
    step3_inc_f = 0.0 # rad
    step3_semi_maj_f = 20.0 # r_g

    stepw0_ecc_0 = 0.7
    stepw0_inc_0 = np.pi * (175.0/180.0) # rad
    stepw0_semi_maj_0 = 100.0 # r_g

    stepw0_ecc_f = 0.5
    stepw0_inc_f = np.pi * (170.0/180.0) # rad
    stepw0_semi_maj_f = 60.0 # r_g

    step1_time = 1.5e5 #years
    step1_delta_ecc = step2_ecc_0 - step1_ecc_0
    step1_delta_semimaj = step1_semi_maj_0 - step2_semi_maj_0 #rg
    step1_delta_inc = step1_inc_0 - step2_inc_0 #rad
 
    step2_time = 1.4e4 # years
    step2_delta_ecc = step2_ecc_0 - step3_ecc_0
    step2_delta_semimaj = step2_semi_maj_0 - step3_semi_maj_0 #rg
    step2_delta_inc = step2_inc_0 - step3_inc_0

    step3_time = 1.4e4 #years
    step3_delta_ecc = step3_ecc_0 - step3_ecc_f
    step3_delta_semimaj = step3_semi_maj_0 - step3_semi_maj_f #rg
    step3_delta_inc = step3_inc_0 - step3_inc_f

    # Then figure out cos(w)=0
    # this evolution does one thing: shrink semimaj axis, circ (slowly), flip (even slower)
    #   scaling from fig 8 WZL comparing cos(w)=0 to cos(w)=+/-1
    #       tau_semimaj~1/100, tau_ecc~1/1000, tau_inc~1/5000
    #       at high inc, large ecc
    #       
    #      Estimating for mass_smbh=1e8Msun, orbiter_mass=30Msun, SG disk surf dens
    #       in 1.5e7yrs a=100rg->60rg, e=0.7->0.5, i=175->170deg
    stepw0_time = 1.5e7 #years
    stepw0_delta_ecc = stepw0_ecc_0 - stepw0_ecc_f
    stepw0_delta_semimaj = stepw0_semi_maj_0 - stepw0_semi_maj_f #rg
    stepw0_delta_inc = stepw0_inc_0 - stepw0_inc_f

    # Doing it in a for loop bc data structures gave me a headache
    for i in range(len(orbiter_periapse)):
        # first we're doing cos(w)~+/-1
        if (np.abs(np.cos(orbiter_periapse[i])) >= 0.5):
            # check that we haven't hit our max ecc for step 1, and remain somewhat retrograde
            if ((orbiter_ecc[i] < step2_ecc_0) and (np.abs(orbiter_inc[i]) >= np.pi/2.0)):
                # adjust our timesteps for actual values of params, vs baseline
                tau_e_current, tau_a_current = tau_ecc_dyn(mass_smbh, orbiter_semimaj[i], orbiter_masses[i], orbiter_periapse[i], orbiter_ecc[i], orbiter_inc[i], surfdensfunc)
                tau_e_ref, tau_a_ref = tau_ecc_dyn(mass_smbh_0, step1_semi_maj_0, orbiter_mass_0, periapse_1, step1_ecc_0, step1_inc_0, surfdensfunc)
                ecc_scale_factor = step1_time * tau_e_current/tau_e_ref
                semimaj_scale_factor = step1_time * tau_a_current/tau_a_ref
                inc_scale_factor = step1_time * tau_inc_dyn(mass_smbh, orbiter_semimaj[i], orbiter_masses[i], orbiter_periapse[i], orbiter_ecc[i], orbiter_inc[i], surfdensfunc) / tau_inc_dyn(mass_smbh_0, step1_semi_maj_0, orbiter_mass_0, periapse_1, step1_ecc_0, step1_inc_0, surfdensfunc)
                new_orbiter_ecc[i] = orbiter_ecc[i] * (1.0 + step1_delta_ecc/orbiter_ecc[i] * (timestep/ecc_scale_factor))
                # catch overshooting ecc=1
                if new_orbiter_ecc[i] > 1.0: new_orbiter_ecc[i] = 1.0
                new_orbiter_semimaj[i] = orbiter_semimaj[i] * (1.0 - step1_delta_semimaj/orbiter_semimaj[i] * (timestep/semimaj_scale_factor))
                # catch overshooting semimaj axis, set to 0.0
                if new_orbiter_semimaj[i] <= 0.0: new_orbiter_semimaj[i] = 0.0
                new_orbiter_inc[i] = orbiter_inc[i] * (1.0 - step1_delta_inc/orbiter_inc[i] * (timestep/inc_scale_factor))
                # catch overshooting inc, set to 0.0
                if new_orbiter_inc[i] <= 0.0: new_orbiter_inc[i] = 0.0
            # check if we have hit max ecc, which sends us to step 2
            elif (orbiter_ecc[i] >= step2_ecc_0):
                # adjust our timesteps for actual values of params, vs baseline
                tau_e_current, tau_a_current = tau_ecc_dyn(mass_smbh, orbiter_semimaj[i], orbiter_masses[i], orbiter_periapse[i], orbiter_ecc[i], orbiter_inc[i], surfdensfunc)
                tau_e_ref, tau_a_ref = tau_ecc_dyn(mass_smbh_0, step2_semi_maj_0, orbiter_mass_0, periapse_1, step2_ecc_0, step2_inc_0, surfdensfunc)
                ecc_scale_factor = step2_time * tau_e_current/tau_e_ref
                semimaj_scale_factor = step2_time * tau_a_current/tau_a_ref
                inc_scale_factor = step2_time * tau_inc_dyn(mass_smbh, orbiter_semimaj[i], orbiter_masses[i], orbiter_periapse[i], orbiter_ecc[i], orbiter_inc[i], surfdensfunc) / tau_inc_dyn(mass_smbh_0, step2_semi_maj_0, orbiter_mass_0, periapse_1, step2_ecc_0, step2_inc_0, surfdensfunc)
                new_orbiter_ecc[i] = orbiter_ecc[i] * (1.0 - step2_delta_ecc/orbiter_ecc[i] * (timestep/ecc_scale_factor))
                # catch overshooting ecc=0
                if new_orbiter_ecc[i] < 0.0: new_orbiter_ecc[i] = 0.0
                new_orbiter_semimaj[i] = orbiter_semimaj[i] * (1.0 - step2_delta_semimaj/orbiter_semimaj[i] * (timestep/semimaj_scale_factor))
                # catch overshooting semimaj axis, set to 0.0
                if new_orbiter_semimaj[i] <= 0.0: new_orbiter_semimaj[i] = 0.0
                new_orbiter_inc[i] = orbiter_inc[i] * (1.0 - step2_delta_inc/orbiter_inc[i] * (timestep/inc_scale_factor))
                # catch overshooting inc, set to 0.0
                if new_orbiter_inc[i] <= 0.0: new_orbiter_inc[i] = 0.0
            # if our inc is even barely prograde... hopefully this works ok...
            # this should work as long as we're only tracking stuff originally retrograde
            elif (np.abs(orbiter_inc[i]) < (np.pi/2.0)):
                # adjust our timesteps for actual values of params, vs baseline
                tau_e_current, tau_a_current = tau_ecc_dyn(mass_smbh, orbiter_semimaj[i], orbiter_masses[i], orbiter_periapse[i], orbiter_ecc[i], orbiter_inc[i], surfdensfunc)
                tau_e_ref, tau_a_ref = tau_ecc_dyn(mass_smbh_0, step3_semi_maj_0, orbiter_mass_0, periapse_1, step3_ecc_0, step3_inc_0, surfdensfunc)
                ecc_scale_factor = step3_time * tau_e_current/tau_e_ref
                semimaj_scale_factor = step3_time * tau_a_current/tau_a_ref
                inc_scale_factor = step3_time * tau_inc_dyn(mass_smbh, orbiter_semimaj[i], orbiter_masses[i], orbiter_periapse[i], orbiter_ecc[i], orbiter_inc[i], surfdensfunc) / tau_inc_dyn(mass_smbh_0, step3_semi_maj_0, orbiter_mass_0, periapse_1, step3_ecc_0, step3_inc_0, surfdensfunc)
                new_orbiter_ecc[i] = orbiter_ecc[i] * (1.0 - step3_delta_ecc/orbiter_ecc[i] * (timestep/ecc_scale_factor))
                # catch overshooting ecc=0
                if new_orbiter_ecc[i] < 0.0: new_orbiter_ecc[i] = 0.0
                new_orbiter_semimaj[i] = orbiter_semimaj[i] * (1.0 - step3_delta_semimaj/orbiter_semimaj[i] * (timestep/semimaj_scale_factor))
                # catch overshooting semimaj axis, set to 0.0
                if new_orbiter_semimaj[i] <= 0.0: new_orbiter_semimaj[i] = 0.0
                new_orbiter_inc[i] = orbiter_inc[i] * (1.0 - step3_delta_inc/orbiter_inc[i] * (timestep/inc_scale_factor))
                # catch overshooting inc, set to 0.0
                if new_orbiter_inc[i] <= 0.0: new_orbiter_inc[i] = 0.0
            # print warning if none of the conditions are satisfied
            else:
                print("Warning: retrograde orbital parameters out of range, behavior unreliable")

        # Want to put something in here that checks if inc is in the disk prograde, and if yes
        #   passes the object to the prograde singleton array. Also want to catch EMRIs...

        # Good news: this works for the original settings... bad news, the other arg periapse doesn't...
        # then do cos(w)=0 (assume abs(cos(w))<0.5)
        elif (np.abs(np.cos(orbiter_periapse[i])) < 0.5):
            # adjust our timesteps for actual values of params, vs baseline
            tau_e_current, tau_a_current = tau_ecc_dyn(mass_smbh, orbiter_semimaj[i], orbiter_masses[i], orbiter_periapse[i], orbiter_ecc[i], orbiter_inc[i], surfdensfunc)
            tau_e_ref, tau_a_ref = tau_ecc_dyn(mass_smbh_0, stepw0_semi_maj_0, orbiter_mass_0, periapse_0, stepw0_ecc_0, stepw0_inc_0, surfdensfunc)
            ecc_scale_factor = stepw0_time * tau_e_current/tau_e_ref
            semimaj_scale_factor = stepw0_time * tau_a_current/tau_a_ref
            inc_scale_factor = stepw0_time * tau_inc_dyn(mass_smbh, orbiter_semimaj[i], orbiter_masses[i], orbiter_periapse[i], orbiter_ecc[i], orbiter_inc[i], surfdensfunc) / tau_inc_dyn(mass_smbh_0, stepw0_semi_maj_0, orbiter_mass_0, periapse_0, stepw0_ecc_0, stepw0_inc_0, surfdensfunc)
            new_orbiter_ecc[i] = orbiter_ecc[i] * (1.0 - stepw0_delta_ecc/orbiter_ecc[i] * (timestep/stepw0_time))
            # catch overshooting ecc=0
            if new_orbiter_ecc[i] < 0.0: new_orbiter_ecc[i] = 0.0
            new_orbiter_semimaj[i] = orbiter_semimaj[i] * (1.0 - stepw0_delta_semimaj/orbiter_semimaj[i] * (timestep/stepw0_time))
            # catch overshooting semimaj axis, set to 0.0
            if new_orbiter_semimaj[i] <= 0.0: new_orbiter_semimaj[i] = 0.0
            new_orbiter_inc[i] = orbiter_inc[i] * (1.0 - stepw0_delta_inc/orbiter_inc[i] * (timestep/stepw0_time))
            # catch overshooting inc, set to 0.0
            if new_orbiter_inc[i] <= 0.0: new_orbiter_inc[i] = 0.0
        else:
            print("Warning: retrograde argument of periapse out of range, behavior unreliable")
    
    return new_orbiter_ecc, new_orbiter_semimaj, new_orbiter_inc

def tau_inc_dyn(mass_smbh, orbiter_semimaj, orbiter_masses, orbiter_periapse, orbiter_ecc, orbiter_inc, surfdensfunc):
    """computes inclination damping timescale from actual variables; used only for scaling


    Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    orbiter_semimaj : float array
        locations of retrograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
        (actually we assume this is the same as the semi-major axis)
    orbiter_masses : float array
        mass of retrograde singleton BH at start of timestep in units of solar masses
    orbiter_periapse : float array
        argument of periapse of retrograde singleton BH at start of timestep.
    orbiter_ecc : float array
        orbital eccentricity of retrograde singleton BH at start of timestep.
    orbiter_inc : float array
        orbital inclination in radians of retrograde singleton BH at start of timestep.
    surfdensfunc : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g

    Returns
    -------
    tau_i_dyn : float array
        inclination damping timescale in seconds
    """
    # This should probably be set somewhere for the whole code? But...
    KgPerMsun = 1.99e30

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    SI_mass_smbh = mass_smbh * KgPerMsun  # kg
    SI_semi_maj_axis = orbiter_semimaj * scipy.constants.G * mass_smbh \
                    / (scipy.constants.c)**2  # m
    SI_orbiter_mass = orbiter_masses * KgPerMsun  # kg
    omega = orbiter_periapse  # radians
    ecc = orbiter_ecc  # unitless
    inc = orbiter_inc  # radians

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt(SI_semi_maj_axis**3/(scipy.constants.G * SI_mass_smbh))
    # semi-latus rectum in units of meters
    semi_lat_rec = SI_semi_maj_axis * (1.0-ecc**2)
    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + ecc**2 + 2.0*ecc*np.cos(omega))
    sigma_minus = np.sqrt(1.0 + ecc**2 - 2.0*ecc*np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc*np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc*np.cos(omega))
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0/eta_plus**15) + np.sqrt(1.0/eta_minus**15))
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus/eta_plus**2 + sigma_minus/eta_minus**2)
    # WZL Eqn 71
    #   NOTE: preserved orbiter_semimaj in r_g to feed to surfdensfunc function
    #   tau in units of sec
    tau_i_dyn = np.sqrt(2.0) * inc * (delta - np.cos(inc))**1.5 \
                * SI_mass_smbh**2 * period / (SI_orbiter_mass*surfdensfunc(orbiter_semimaj)*np.pi*semi_lat_rec**2) \
                / kappa
    
    return tau_i_dyn

def tau_semi_lat(mass_smbh,retrograde_bh_locations,retrograde_bh_masses,retrograde_bh_orb_ecc,retrograde_bh_orb_inc,retro_arg_periapse,disk_surf_model):
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

def tau_ecc_dyn(mass_smbh, orbiter_semimaj, orbiter_masses, orbiter_periapse, orbiter_ecc, orbiter_inc, surfdensfunc):
    """computes eccentricity & semi-maj axis damping timescale from actual variables
    not including migration; used only for scaling

    Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    orbiter_semimaj : float array
        locations of retrograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
        (actually we assume this is the same as the semi-major axis)
    orbiter_masses : float array
        mass of retrograde singleton BH at start of timestep in units of solar masses
    orbiter_periapse : float array
        argument of periapse of retrograde singleton BH at start of timestep.
    orbiter_ecc : float array
        orbital eccentricity of retrograde singleton BH at start of timestep.
    orbiter_inc : float array
        orbital inclination in radians of retrograde singleton BH at start of timestep.
    surfdensfunc : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g

    Returns
    -------
    tau_e_dyn, tau_a_dyn : float arrays
        eccentricty, semi-major axis damping timescales in seconds
    """
    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    omega = orbiter_periapse  # radians
    ecc = orbiter_ecc  # unitless
    inc = orbiter_inc  # radians

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
    tau_p_dyn = tau_semi_lat(mass_smbh,orbiter_semimaj,orbiter_masses,orbiter_ecc,orbiter_inc,orbiter_periapse,surfdensfunc)
    #  also need to find tau_a_dyn, but
    #   fortunately it's a few factors off of tau_p_dyn (this may be a dumb way to handle it)
    tau_a_dyn = tau_p_dyn * (1.0 - ecc**2) * kappa * np.abs(np.cos(inc) - zeta)/(kappa_bar * np.abs(np.cos(inc) - zeta_bar))
    # WZL Eqn 73
    tau_e_dyn = (2.0 * ecc**2 / (1.0 - ecc**2)) * 1.0 / np.abs(1.0/tau_a_dyn - 1.0/tau_p_dyn)

    return tau_e_dyn, tau_a_dyn