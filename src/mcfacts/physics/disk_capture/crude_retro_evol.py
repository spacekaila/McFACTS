import numpy as np
import scipy

def crude_retro_bh_semimaj(mass_smbh, orbiter_masses, orbiter_semimaj, orbiter_ecc, orbiter_inc, orbiter_periapse, surfdensfunc, timestep):
    """
    To avoid having to install and couple to SpaceHub, and run N-body code
    this is a distinctly half-assed treatment of retrograde orbiters, based
    LOOSELY on Wang, Zhu & Lin 2024 (WZL). Evolving all orbital params simultaneously.
    Using lots of if statments to pretend we're interpolating.
    Hardcoding some stuff from WZL figs 7, 8 & 12.
    Arg of periapse = w in comments below

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

    step1_time = 1.5e5 #check units
    step1_delta_ecc = 0.9999 - 0.7
    step1_delta_semimaj = 100.0 - 60.0
    step1_delta_inc = np.pi * (175.0 - 165.0)/180.0
    step2_time = 1.4e4 #check units
    step2_delta_ecc = 0.9999 - 0.9
    step2_delta_semimaj = 0.0
    step2_delta_inc = np.pi * (165.0 - 12.0)/180.0
    step3_time = 1.4e4 #check units

    # Check that we're doing cos(w)~+/-1
    if (abs(cos(orbiter_periapse))>0.5):
        # check that we haven't hit our max ecc for step 1 !!! HAVE TO ADD INC CONDITION!!!
        if (orbiter_ecc < 0.9999):
            new_orbiter_ecc = orbiter_ecc * (1.0 + step1_delta_ecc * (timestep/step1_time))
            new_orbiter_semimaj = orbiter_semimaj * (1.0 + step1_delta_semimaj * (timestep/step1_time))
            new_orbiter_inc = orbiter_inc * (1.0 + step1_delta_inc * (timestep/step1_time))
        # check if we have hit max ecc, which sends us to step 2
        elif (orbiter_ecc >= 0.9999):
            new_orbiter_ecc = orbiter_ecc * (1.0 + step2_delta_ecc * (timestep/step2_time))
            new_orbiter_semimaj = orbiter_semimaj * (1.0 + step2_delta_semimaj * (timestep/step2_time))
            new_orbiter_inc = orbiter_inc * (1.0 + step2_delta_inc * (timestep/step2_time))

    # then do cos(w)=0 (assume abs(cos(w))<0.5)
    # this evolution does one thing: shrink semimaj axis, circ (slowly), flip (even slower)
    #   scaling from fig 8 WZL comparing cos(w)=0 to cos(w)=+/-1
    #       tau_semimaj~1/100, tau_ecc~1/1000, tau_inc~1/5000
    #       at high inc, large ecc
    #       
    #      Estimating for mass_smbh=1e8Msun, orbiter_mass=30Msun, SG disk surf dens
    #       in 1.5e7yrs a=100rg->60rg, e=0.7->0.5, i=175->170deg

    dummy = 1.0

    return dummy