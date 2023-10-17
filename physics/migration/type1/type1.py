import numpy as np


def dr_migration(prograde_bh_locations, prograde_bh_masses, disk_surf_model, disk_aspect_ratio_model, timestep):
    """ This function calculates how far an object migrates in an AGN gas disk in a time
    of length timestep, assuming a gas disk surface density and aspect ratio profile, for
    objects of specified masses and starting locations, and returns their new locations
    after migration over one timestep.

    Note that migration distance is calcuated using a simple scaling from Sirko & Goodman
    (2003) disk model, where 38Myrs=3.8e7yrs is the time for a 5Msun BH to undergo Type I 
    migration to the SMBH from 10^4r_g in that model.

    Parameters
    ----------
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
    # Normalization for scaling everything
    sg_norm = 3.8e7
    #scaled mass= BH mass/lower bound mass (e.g. 5Msun, upper end of lower mass gap)
    scaled_mass = 5.0
    #scaled_aspect=disk_aspect ratio scaled to 0.02 as a fiducial value.
    scaled_aspect = 0.02
    #scaled location= BH location scaled to 10^4r_g
    scaled_location = 1.e4
    #scaled sigma= Disk surface density scaled to 10^5kg/m^2
    scaled_sigma = 1.e5
    #Normalize the locations and BH masses 
    normalized_locations = prograde_bh_locations/scaled_location
    normalized_masses = prograde_bh_masses/scaled_mass
    normalized_locations_sqrt = np.sqrt(normalized_locations)
    #Can normalize the aspect ratio and sigma to these scales when we
    # implement the 1d disk model (interpolate over SG03)
    normalized_sigma = disk_surface_density/scaled_sigma
    normalized_aspect_ratio = disk_aspect_ratio/scaled_aspect
    normalized_aspect_ratio_squared = np.square(normalized_aspect_ratio)
    #So our fiducial timescale should now be 38Myrs as calcd below
    dt_mig = sg_norm*(normalized_aspect_ratio_squared)/((normalized_masses)*(normalized_locations_sqrt)*(normalized_sigma))
    #Effective fractional time of migration is timestep/dt_mig
    fractional_migration_timestep = timestep/dt_mig
    #Migration distance is location of BH * fractional_migration_timestep
    migration_distance = prograde_bh_locations*fractional_migration_timestep
    bh_new_locations = prograde_bh_locations-migration_distance

    return bh_new_locations
