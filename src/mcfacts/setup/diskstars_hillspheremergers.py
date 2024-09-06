import numpy as np

def true_mass_location_relation(N_stars, m_min, M_smbh, P_m, P_r, r_location, R_disk):
    """
    N_stars: number of stars in the initial draw
             int
    m_min: minimum mass considered
            float, M_sun
    M_smbh: mass of SMBH
            float, M_sun
    P_m: exponent for mass cdf, assuming it is in the form P(> m_min) = (m_min/m)^P_m
            float
    P_r: exponent for disk location cdf, assuming the form P(r) = (r_location/R_disk)^P_r
            float
    r_location: semi-major axis of stellar orbit around SMBH
            numpy array, R_sun (for now?)
    R_disk: trap radius of disk
            float, R_sun (for now?)

    Returns:
    mass_threshold: minimum mass for stars to not merge
            array, M_sun
    """
    exp1_top = P_m/(P_m - (1./3.))
    exp1_bottom = 1./(3*(P_m - (1./3.)))
    frac1 = (np.power(m_min,exp1_top))/(np.power(3*M_smbh,exp1_bottom))

    exp2 = 1./(P_m - (1./3.))
    frac2 = np.power(N_stars*P_r,exp2)

    exp3 = P_r/(P_m - (1./3.))
    frac3 = np.power(r_location/R_disk,exp3)

    mass_threshold = frac1*frac2*frac3

    return(mass_threshold)


def get_location_steps(r_location_sorted, mass_threshold, M_smbh, R_disk):
    delta_r_locations = []
    delta_r_locations.append(10)
    loc = delta_r_locations[0]*np.power(mass_threshold[0]/(3.*M_smbh),1./3.) + delta_r_locations[0]
    delta_r_locations.append(loc)
    while loc < R_disk:
        idx = (np.abs(loc - r_location_sorted)).argmin()
        #print(idx)
        loc += r_location_sorted[idx]*np.power(mass_threshold[idx]/(3.*M_smbh),1./3.)
        delta_r_locations.append(loc)
    delta_r_locations = np.array(delta_r_locations)
    return(delta_r_locations)


def hillsphere_mergers (n_stars,masses_initial_sorted, r_locations_initial_sorted, min_initial_star_mass, R_disk, M_smbh, P_m, P_r):
    # P_m and P_r should be added to opts I am just too lazy at the mo

    mass_threshold = true_mass_location_relation(N_stars=n_stars,
                                                 m_min = min_initial_star_mass,
                                                 M_smbh=M_smbh,
                                                 P_m = P_m,
                                                 P_r = P_r,
                                                 r_location=r_locations_initial_sorted,
                                                 R_disk=R_disk)
    
    location_steps = get_location_steps(r_locations_initial_sorted,mass_threshold,M_smbh, R_disk)

    new_masses = []
    new_r = []
    for idx in range(len(location_steps)-1):
        if(len(r_locations_initial_sorted[(r_locations_initial_sorted>location_steps[idx]) & (r_locations_initial_sorted<location_steps[idx+1])]) > 0):
            mass_range = masses_initial_sorted[(r_locations_initial_sorted>location_steps[idx]) & (r_locations_initial_sorted<location_steps[idx+1])]
            r_range = r_locations_initial_sorted[(r_locations_initial_sorted>location_steps[idx]) & (r_locations_initial_sorted<location_steps[idx+1])]
            mt_temp = true_mass_location_relation(N_stars=n_stars,
                                                m_min = min_initial_star_mass,
                                                M_smbh=1.e8,
                                                P_m = 1.35,
                                                P_r = 1.,
                                                r_location=r_range,
                                                R_disk=R_disk)
            mass_range_merge = mass_range[mass_range<=mt_temp]
            r_range_merge = r_range[mass_range<=mt_temp]
            
            mass_range_static = mass_range[mass_range>mt_temp]
            r_range_static = r_range[mass_range>mt_temp]

            if(len(mass_range_static) > 0):
                for m, r in zip(mass_range_static,r_range_static):
                    new_masses.append(m)
                    new_r.append(r)

            if(len(mass_range_merge) > 0):
                new_star_mass = np.sum(mass_range)
                new_star_r = np.average(r_range,weights=mass_range)
                new_masses.append(new_star_mass)
                new_r.append(new_star_r)

    new_masses = np.array(new_masses)
    new_r = np.array(new_r)

    return(new_masses,new_r)



