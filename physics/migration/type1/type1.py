import numpy as np
import scipy

def type1_migration(mass_smbh, prograde_bh_locations, prograde_bh_masses, disk_surf_model, disk_aspect_ratio_model, timestep, feedback_ratio):
    """This function calculates how far an object migrates in an AGN gas disk in a time
    of length timestep, assuming a gas disk surface density and aspect ratio profile, for
    objects of specified masses and starting locations, and returns their new locations
    after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_model : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    timestep : float
        size of timestep in years
    feedback_ratio : function
        ratio of heating/migration torque. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    Returns
    -------
    bh_new_locations : float array
        locations of prograde singleton BH at end of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    """
    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_model, float):
        disk_surface_density = disk_surf_model
    else:
        disk_surface_density = disk_surf_model(prograde_bh_locations)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_model, float):
        disk_aspect_ratio = disk_aspect_ratio_model
    else:
        disk_aspect_ratio = disk_aspect_ratio_model(prograde_bh_locations)

    # compute migration timescale for each orbiter in seconds
    # eqn from Paardekooper 2014, rewritten for R in terms of r_g of SMBH = GM_SMBH/c^2
    # tau = (pi/2) h^2/(q_d*q) * (1/Omega)
    #   where h is aspect ratio, q is m/M_SMBH, q_d = pi R^2 disk_surface_density/M_SMBH
    #   and Omega is the Keplerian orbital frequency around the SMBH
    # here mass_smbh/prograde_bh_masses are both in M_sun, so units cancel
    # c, G and disk_surface_density in SI units
    tau_mig = ((disk_aspect_ratio**2)* scipy.constants.c/(2.0*scipy.constants.G) * (mass_smbh/prograde_bh_masses) / disk_surface_density) / np.sqrt(prograde_bh_locations)
    # ratio of timestep to tau_mig (timestep in years so convert)
    dt = timestep * scipy.constants.year / tau_mig
    # migration distance is original locations times fraction of tau_mig elapsed
    migration_distance = prograde_bh_locations * dt

    # if feedback, modify migration distance
    # Construct empty array same size as prograde_bh_locations 

    bh_new_locations = np.empty_like(prograde_bh_locations)

    #Find indices of objects where feedback ratio <1; these still migrate inwards, but more slowly
    #feedback ratio is a tuple, so need [0] part not [1] part (ie indices not details of array)
    index_inwards_modified = np.where(feedback_ratio < 1)[0]
    if index_inwards_modified.size > 0:
        print("index_inwards_modified",index_inwards_modified)
        print("feedback_ratio",feedback_ratio)
        print("feedback ratio[index_inwards_modified]", feedback_ratio[index_inwards_modified])
        print("prograde_bh_locations[index_inwards_modified]", prograde_bh_locations[index_inwards_modified])
        print("migration_distance[index_inwards_modified]", migration_distance[index_inwards_modified])
    
 
        bh_new_locations[index_inwards_modified] = prograde_bh_locations[index_inwards_modified] - (migration_distance[index_inwards_modified]*(1-feedback_ratio[index_inwards_modified]))
    #Find indices of objects where feedback ratio >1; these migrate outwards.
    index_outwards_modified = np.where(feedback_ratio >1)[0]
    print("index_outwards_modified",index_outwards_modified)
    if index_outwards_modified.size > 0:
        bh_new_locations[index_outwards_modified] = prograde_bh_locations[index_outwards_modified] +(migration_distance[index_outwards_modified]*(feedback_ratio[index_outwards_modified]-1))
    
    index_unchanged = np.where(feedback_ratio == 1)[0]
    if index_unchanged.size > 0:
        bh_new_locations[index_unchanged] = prograde_bh_locations[index_unchanged] - migration_distance[index_unchanged]
    
    print("bh new locations",bh_new_locations)

    # new locations are original ones - distance traveled
    #bh_new_locations = prograde_bh_locations - migration_distance

    return bh_new_locations
