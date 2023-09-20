import numpy as np


def change_bin_mass(bin_array, frac_Eddington_ratio, mass_growth_Edd_rate, timestep, integer_nbinprop, bin_index):
    #Return new updated mass array due to accretion for prograde orbiting BH after timestep
    #Extract the binary locations and masses
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    
    for j in range(0, bindex):
            #for i in range(0, integer_nbinprop):
                temp_bh_mass_1 = bin_array[2,j] 
                temp_bh_mass_2 = bin_array[3,j]
                mass_growth_factor = np.exp(mass_growth_Edd_rate*frac_Eddington_ratio*timestep)
                new_bh_mass_1 = temp_bh_mass_1*mass_growth_factor
                new_bh_mass_2 = temp_bh_mass_2*mass_growth_factor
                #Update new bh masses in bin_array
                bin_array[2,j] = new_bh_mass_1
                bin_array[3,j] = new_bh_mass_2

    return bin_array


def change_bin_spin_magnitudes(bin_array, frac_Eddington_ratio, spin_torque_condition, timestep, integer_nbinprop, bin_index):
    #def change_spin_magnitudes(bh_spins,prograde_orb_ang_mom_indices,frac_Eddington_ratio,spin_torque_condition,mass_growth_Edd_rate,timestep):
    #bh_new_spins=bh_spins
    normalized_Eddington_ratio = frac_Eddington_ratio/1.0
    normalized_timestep = timestep/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1
    #Extract the binary locations and spin magnitudes
    #max allowed spin
    max_allowed_spin=0.98
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    
    for j in range(0, bindex):
            #for i in range(0, integer_nbinprop):
                temp_bh_spin_1 = bin_array[4,j] 
                temp_bh_spin_2 = bin_array[5,j]
                spin_change_factor = 4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep
                #print("Spin change factor", spin_change_factor)
                new_bh_spin_1 = temp_bh_spin_1 + spin_change_factor
                #print("Old spin1, new spin1 =",temp_bh_spin_1, new_bh_spin_1)
                new_bh_spin_2 = temp_bh_spin_2 + spin_change_factor
                if new_bh_spin_1 > max_allowed_spin:
                    new_bh_spin_1 = max_allowed_spin
                if new_bh_spin_2 > max_allowed_spin:
                    new_bh_spin_2 = max_allowed_spin
                #Update new bh masses in bin_array
                bin_array[4,j] = new_bh_spin_1
                bin_array[5,j] = new_bh_spin_2

    return bin_array


def change_bin_spin_angles(bin_array, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep, integer_nbinprop, bin_index):
    #Calculate change in spin angle due to accretion during timestep
    normalized_Eddington_ratio = frac_Eddington_ratio/1.0
    normalized_timestep = timestep/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1

    #Extract the binary locations and spin magnitudes
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    
    for j in range(0, bindex):
            #for i in range(0, integer_nbinprop):
                temp_bh_spin_angle_1 = bin_array[6,j] 
                temp_bh_spin_angle_2 = bin_array[7,j]
                #bh_new_spin_angles[prograde_orb_ang_mom_indices]=bh_new_spin_angles[prograde_orb_ang_mom_indices]-(6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
                spin_angle_change_factor = (6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
                new_bh_spin_angle_1 = temp_bh_spin_angle_1 - spin_angle_change_factor
                new_bh_spin_angle_2 = temp_bh_spin_angle_2 - spin_angle_change_factor
                if new_bh_spin_angle_1 < spin_minimum_resolution:
                    new_bh_spin_angle_1 = 0.0
                if new_bh_spin_angle_2 < spin_minimum_resolution:
                    new_bh_spin_angle_2 = 0.0
                bin_array[6,j] = new_bh_spin_angle_1
                bin_array[7,j] = new_bh_spin_angle_2

    return bin_array

def com_migration(bin_array, disk_surf_model, timestep, integer_nbinprop, bin_index):
    #Return updated locations of binary center of mass (com) and location 1,2 
    # based on Type 1 migration prescription
    #sg_norm is a normalization factor for the Sirko & Goodman (2003) disk model
    #38Myrs=3.8e7yrs is the time for a 5Msun BH to undergo Type I migration to
    #the SMBH from 10^4r_g in that model.
    sg_norm = 3.8e7
    #scaled mass= BH mass/lower bound mass (e.g. 5Msun, upper end of lower mass gap)
    scaled_mass = 5.0
    #scaled_aspect=disk_aspect ratio scaled to 0.02 as a fiducial value.
    scaled_aspect = 0.02
    #for test fixed disk aspect ratio
    disk_aspect_ratio = 0.03
    #scaled location= BH location scaled to 10^4r_g
    scaled_location = 1.e4
    #scaled sigma= Disk surface density scaled to 10^5kg/m^2
    scaled_sigma = 1.e5
    
    #Extract the binary locations and masses
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)
    
    #If binary has been removed, it's all zeroes. So find first non-zero binary in array.
    #loc1_bins=bin_array[:,0]
    #live_loc1_bins=np.count_nonzero(loc1_bins) 
    #if live_loc1_bins > 0:
    #    bin_indices = np.where(live_loc1_bins > 0.0)

    for j in range(0,bindex):
            #for i in range(0, integer_nbinprop):
                temp_bh_loc_1 = bin_array[0,j]
                temp_bh_loc_2 = bin_array[1,j]
                temp_bh_mass_1 = bin_array[2,j] 
                temp_bh_mass_2 = bin_array[3,j]
                temp_bin_sep = bin_array[8,j]
                temp_bin_com = bin_array[9,j]
                bin_mass = temp_bh_mass_1 + temp_bh_mass_2
                #Normalize the com location and BH masses 
                if isinstance(disk_surf_model, float):
                    disk_surface_density = disk_surf_model
                else:
                    disk_surface_density = disk_surf_model(temp_bin_com)
                normalized_com = temp_bin_com/scaled_location
                normalized_bin_mass = bin_mass/scaled_mass
                normalized_com_sqrt = np.sqrt(normalized_com)
                #Can normalize the aspect ratio and sigma to these scales when we
                # implement the 1d disk model (interpolate over SG03)
                normalized_sigma = disk_surface_density/scaled_sigma
                normalized_aspect_ratio = disk_aspect_ratio/scaled_aspect
                normalized_aspect_ratio_squared = np.square(normalized_aspect_ratio)
                #So our fiducial timescale should now be 38Myrs as calcd below
                dt_mig = sg_norm*(normalized_aspect_ratio_squared)/((normalized_bin_mass)*(normalized_com_sqrt)*(normalized_sigma))
                #Effective fractional time of migration is timestep/dt_mig
                fractional_migration_timestep = timestep/dt_mig
                #Migration distance is location of bin com * fractional_migration_timestep
                migration_distance = temp_bin_com*fractional_migration_timestep
                new_bin_com = temp_bin_com-migration_distance
                new_bh_loc1 = new_bin_com -(temp_bin_sep*temp_bh_mass_2/bin_mass)
                new_bh_loc2 = new_bh_loc1 + temp_bin_sep
                #Write new values of R1,R2,com to bin_array
                bin_array[0,j] = new_bh_loc1
                bin_array[1,j] = new_bh_loc2
                bin_array[9,j] = new_bin_com

    return bin_array




