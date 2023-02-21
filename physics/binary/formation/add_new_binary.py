import numpy as np


def add_to_binary_array(bin_array, bh_locations, bh_masses, bh_spins, bh_spin_angles, close_encounters, bin_index):
    #Here we add a new binary to this array, take properties from existing individual arrays and create some new ones
    #Column 1 is 1 binary, Column 2 is 2nd binary etc.
    #Extract location,mass,spin,spin angle from arrays & add to this array (=8 params)
    #Create new properties based on these
    # a_bin=R2-R1
    # a_com=Semi-major axis between binary Center of Mass and SMBH.
    # ecc=binary eccentricity (start with zero, but WANT TO DRAW FROM PRESCRIPTION)
    #bin_ang_mom=Is the binary prograde (+1) or retrograde(-1)
    #generation=Hierarchical history of these BHs 11=both 1st g 13=1st g+3rd g etc.
    # 13 params total
    #In Column 1 M1,M2,a1,a2,theta1,theta2,R1,R2,a_bin=(R2-R1),a_com,t_gw,bin_ang_mom,gen
   
    #Start by extracting all relevant data first

    sorted_bh_locations = np.sort(bh_locations)
    sorted_bh_locations_indices = np.argsort(bh_locations)
    bh_masses_by_sorted_location = bh_masses[sorted_bh_locations_indices]
    bh_spins_by_sorted_location = bh_spins[sorted_bh_locations_indices]
    bh_spin_angles_by_sorted_location = bh_spin_angles[sorted_bh_locations_indices]
    #bh_orb_ecc_by_sorted_location=bh_orbital_eccentricities[sorted_bh_locations_indices]

    bindex = int(bin_index)
    number_of_new_bins = (len(close_encounters)+1)/2
    num_new_bins = int(number_of_new_bins)
    if number_of_new_bins > 0:
        print("no of new bins")
        print(num_new_bins)
        #print("indices")
        #print(close_encounters)
        #print(len(close_encounters))
        array_of_indices = close_encounters
        for j in range(bindex, bindex + (num_new_bins)):
            for i in range(0,len(close_encounters)):
                new_indx = array_of_indices[i]
                #print(new_indx)
                bin_array[i,j] = sorted_bh_locations[new_indx]
                bin_array[i+2,j] = bh_masses_by_sorted_location[new_indx]
                bin_array[i+4,j] = bh_spins_by_sorted_location[new_indx]
                bin_array[i+6,j] = bh_spin_angles_by_sorted_location[new_indx]
                #For new binary create initial binary semi-major axis
                temp_loc_1 = bin_array[0,j]
                temp_loc_2 = bin_array[1,j]
                temp_bin_separation = np.abs(temp_loc_1 - temp_loc_2)
                bin_array[8,j] = temp_bin_separation
                #Binary c.o.m.= location_1 + separation*M_2/(M_1+M_2)
                temp_mass_1 = bin_array[2,j]
                temp_mass_2 = bin_array[3,j]
                temp_bin_mass = temp_mass_1 + temp_mass_2
                bin_array[9,j] = temp_loc_1 + (temp_bin_separation*temp_mass_2/temp_bin_mass)
        print("New Binary")
        print(bin_array)
        

    return bin_array
