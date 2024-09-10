import numpy as np
import scipy


def type1_migration(smbh_mass, disk_bh_orb_a_pro, disk_bh_mass_pro, disk_surf_density_func, disk_aspect_ratio_func, timestep_duration, disk_feedback_ratio_func, disk_radius_trap, disk_bh_orb_ecc_pro, disk_bh_pro_orb_ecc_crit):
    """This function calculates how far an object migrates in an AGN gas disk in a time
    of length timestep, assuming a gas disk surface density and aspect ratio profile, for
    objects of specified masses and starting locations, and returns their new locations
    after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    disk_bh_orb_a_pro : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    disk_bh_mass_pro : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_density_func : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    timestep_duration : float
        size of timestep in years
    disk_feedback_ratio_func : function
        ratio of heating/migration torque. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_raduis_trap : float
        radius of disk migration trap in units of gravitational radii (r_g=GM_smbh/c^2) 
    disk_bh_orb_ecc_pro : float array
        orbital ecc of prograde singleton BH at start of timestep. Floor in orbital ecc given by e_crit
    disk_bh_pro_orb_ecc_crit : float
        Critical value of orbital eccentricity below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)           
    Returns
    -------
    bh_new_locations : float array
        locations of prograde singleton BH at end of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    """
    # get surface density function, or process if just a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(disk_bh_orb_a_pro)
    # get aspect ratio function, or process if just a float
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(disk_bh_orb_a_pro)

    # Migration can only occur for sufficiently damped orbital ecc. If orb ecc <= e_crit, then migration. 
    # Otherwise, no change in semi-major axis. Wait till orb ecc damped to <=e_crit.
    # Only show BH with orb ecc <=e_crit
    disk_bh_orb_ecc_pro = np.ma.masked_where(disk_bh_orb_ecc_pro > disk_bh_pro_orb_ecc_crit, disk_bh_orb_ecc_pro)
    #Those BH with orb ecc > e_crit
    prograde_bh_not_mig = np.ma.masked_where(disk_bh_orb_ecc_pro <= disk_bh_pro_orb_ecc_crit, disk_bh_orb_ecc_pro)
    #Indices of BH with <=critical ecc
    crit_ecc_prograde_indices = np.ma.nonzero(disk_bh_orb_ecc_pro)
    #Indicies of BH with > critical ecc
    indices_not_mig_BH = np.ma.nonzero(prograde_bh_not_mig)    
    
    #Migration only if there are BH with e<=e_crit
    #if np.size(crit_ecc_prograde_indices) > 0:
    # compute migration timescale for each orbiter in seconds
    # eqn from Paardekooper 2014, rewritten for R in terms of r_g of SMBH = GM_SMBH/c^2
    # tau = (pi/2) h^2/(q_d*q) * (1/Omega)
    #   where h is aspect ratio, q is m/M_SMBH, q_d = pi R^2 disk_surface_density/M_SMBH
    #   and Omega is the Keplerian orbital frequency around the SMBH
    # here smbh_mass/disk_bh_mass_pro are both in M_sun, so units cancel
    # c, G and disk_surface_density in SI units
    tau_mig = ((disk_aspect_ratio**2)* scipy.constants.c/(3.0*scipy.constants.G) * (smbh_mass/disk_bh_mass_pro) / disk_surface_density) / np.sqrt(disk_bh_orb_a_pro)
    # ratio of timestep to tau_mig (timestep in years so convert)
    dt = timestep_duration * scipy.constants.year / tau_mig
    # migration distance is original locations times fraction of tau_mig elapsed
    migration_distance = disk_bh_orb_a_pro * dt
    #Mask migration distance with zeros if orb ecc >= e_crit.
    migration_distance[indices_not_mig_BH] = 0.
     
    # Feedback provides a universal modification of migration distance
    # If feedback off, then feedback_ratio= ones and migration is unchanged
    # Construct empty array same size as prograde_bh_locations 

    bh_new_locations = np.empty_like(disk_bh_orb_a_pro)

    # Find indices of objects where feedback ratio <1; these still migrate inwards, but more slowly
    # feedback ratio is a tuple, so need [0] part not [1] part (ie indices not details of array)
    index_inwards_modified = np.where(disk_feedback_ratio_func < 1)[0]
    index_inwards_size = index_inwards_modified.size
    all_inwards_migrators = disk_bh_orb_a_pro[index_inwards_modified]
    #print("all inwards migrators",all_inwards_migrators)

    #Given a population migrating inwards
    if index_inwards_size > 0: 
        for i in range(0,index_inwards_size):
                # Among all inwards migrators, find location in disk & compare to trap radius
                critical_distance = all_inwards_migrators[i]
                actual_index = index_inwards_modified[i]
                #If outside trap, migrates inwards
                if critical_distance > disk_radius_trap:
                    bh_new_locations[actual_index] = disk_bh_orb_a_pro[actual_index] - (migration_distance[actual_index]*(1-disk_feedback_ratio_func[actual_index]))
                    #If inward migration takes object inside trap, fix at trap.
                    if bh_new_locations[actual_index] <= disk_radius_trap:
                        bh_new_locations[actual_index] = disk_radius_trap
                #If inside trap, migrates out
                if critical_distance < disk_radius_trap:
                    #print("inside trap radius!")
                    bh_new_locations[actual_index] = disk_bh_orb_a_pro[actual_index] + (migration_distance[actual_index]*(1-disk_feedback_ratio_func[actual_index]))
                    #print("bh_inside_trap", bh_new_locations[actual_index])
                    #If outward migration takes object outside trap, fix at trap.
                    if bh_new_locations[actual_index] >= disk_radius_trap:
                        bh_new_locations[actual_index] = disk_radius_trap
                #If at trap, stays there
                if critical_distance == disk_radius_trap:
                    #print("BH AT TRAP!")
                    #print(bh_new_locations[actual_index])
                    bh_new_locations[actual_index] = disk_bh_orb_a_pro[actual_index]

    # Find indices of objects where feedback ratio >1; these migrate outwards. 
    # In Sirko & Goodman (2003) disk model this is well outside migration trap region.
    index_outwards_modified = np.where(disk_feedback_ratio_func >1)[0]

    if index_outwards_modified.size > 0:
        bh_new_locations[index_outwards_modified] = disk_bh_orb_a_pro[index_outwards_modified] +(migration_distance[index_outwards_modified]*(disk_feedback_ratio_func[index_outwards_modified]-1))
    
    #Find indices where feedback ratio is identically 1; shouldn't happen (edge case) if feedback on, but == 1 if feedback off.
    index_unchanged = np.where(disk_feedback_ratio_func == 1)[0]
    if index_unchanged.size > 0:
    # If BH location > trap radius, migrate inwards
        for i in range(0,index_unchanged.size):
            locn_index = index_unchanged[i]
            if disk_bh_orb_a_pro[locn_index] > disk_radius_trap:    
                bh_new_locations[locn_index] = disk_bh_orb_a_pro[locn_index] - migration_distance[locn_index]
            # if new location is <= trap radius, set location to trap radius
                if bh_new_locations[locn_index] <= disk_radius_trap:
                    bh_new_locations[locn_index] = disk_radius_trap

            # If BH location < trap radius, migrate outwards
            if disk_bh_orb_a_pro[locn_index] < disk_radius_trap:
                bh_new_locations[locn_index] = disk_bh_orb_a_pro[locn_index] + migration_distance[locn_index]
                #if new location is >= trap radius, set location to trap radius
                if bh_new_locations[locn_index] >= disk_radius_trap:
                    bh_new_locations[locn_index] = disk_radius_trap
    #print("bh new locations",np.sort(bh_new_locations))
    #print('migration distance2',migration_distance, prograde_bh_orb_ecc)
    # new locations are original ones - distance traveled
    #bh_new_locations = prograde_bh_locations - migration_distance
    
    return bh_new_locations
