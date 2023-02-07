import numpy as np


def calculate_hill_sphere(prograde_bh_locations, prograde_bh_masses, mass_smbh):
    #Return the Hill sphere radius (R_Hill) for an array of prograde BH where
    # R_Hill=a(q/3)^1/3 where a=semi-major axis, q=m_bh/M_SMBH
    
    bh_smbh_mass_ratio = prograde_bh_masses/(3.0*mass_smbh)
    mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
    bh_hill_sphere = prograde_bh_locations*mass_ratio_factor
    #Return the BH Hill sphere radii for all orbiters. Prograde should have much larger Hill sphere
    return bh_hill_sphere

def encounter_test(prograde_bh_locations, bh_hill_sphere):
    #Using Hill sphere size and BH locations see if there are encounters within the Hill sphere
    # return indices of BH involved.

    # First sort the prograde bh locations in order from inner disk to outer disk
    sorted_bh_locations = np.sort(prograde_bh_locations)
    #Returns the indices of the original array in order, to get the sorted array
    sorted_bh_location_indices = np.argsort(prograde_bh_locations)
   
    #Find the appropriate (sorted) Hill sphere radii
    sorted_hill_spheres = bh_hill_sphere[sorted_bh_location_indices]

    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    separations = np.diff(sorted_bh_locations)

    # Note that separations are -1 of length of bh_locations
    # Take 1st location off locations array
    sorted_bh_locations_minus_first_element = sorted_bh_locations[1:len(sorted_bh_locations)]
    #Take last location off locations array
    sorted_bh_locations_minus_last_element = sorted_bh_locations[0:len(sorted_bh_locations)-1]

    # Separations are -1 of length of Hill_sphere array
    # Take 1st Hill sphere off Hill sphere array
    sorted_hill_spheres_minus_first = sorted_hill_spheres[1:len(sorted_hill_spheres)]
    # Take last Hill sphere off Hill sphere array
    sorted_hill_spheres_minus_last = sorted_hill_spheres[0:len(sorted_hill_spheres)-1]

    # Compare the Hill sphere distance for each BH with separation to neighbor BH
    #so we compare e.g. r2-r1 vs R_H1, r3-r2 vs R_H2
    comparison_distance_inwards = separations-sorted_hill_spheres_minus_last
    # and e.g. compare r2-r1 vs R_H2, r3-r2 vs R_H3
    comparison_distance_outwards = separations-sorted_hill_spheres_minus_first

    index_in = np.where(comparison_distance_inwards < 0)
    # E.g say r3-r2 <R_H2 then we'll want the info for the BH at r2 & r3. (i,i+1)
    index_out = np.where(comparison_distance_outwards < 0)
    #E.g. say r3-r2 <R_H3 then we'll want the info for the BH at r3 and r2 (i,i-1)
    length_index_in = len(index_in)
    length_index_out = len(index_out)

    new_indx_in = list(range(2*len(index_in)))
    new_indx_out = list(range(2*len(index_out)))

    for ind in range(length_index_in):
        temp_index = index_in[ind]
        new_indx_in[ind] = temp_index
        new_indx_in[ind+1] = temp_index+1
   
    for ind in range(length_index_out):
        temp_index = index_out[ind]
        new_indx_out[ind] = temp_index
        new_indx_out[ind+1] = temp_index+1

    new_indxs = new_indx_in+new_indx_out
    rindx = np.sort(new_indxs)
    result = np.asarray(new_indx_in)

    sorted_in_result = np.sort(result)

    new_result = np.asarray(new_indx_out)
    sorted_out_result = np.sort(new_result)
    
    final_bin_indices = sorted_in_result+sorted_out_result
    sorted_final_bin_indices = np.sort(final_bin_indices)

    #print("check if sorted_in & sorted_out arrays are the same")
    check=np.array_equiv(sorted_in_result, sorted_out_result)
    #print(check)
    # Return the indices of those elements in separation array <0
    # (ie BH are closer than 1 R_Hill)
    # In inwards case, r_i+1 -r_i <R_H_i, so relevant BH indices are i,i+1
   
    # In outwards case, r_i - r_i-1 <R_H_i so relevant BH indices are i,i-1
    final_1d_indx_array = sorted_in_result.flatten()
    sorted_final_1d_indx_array = np.sort(final_1d_indx_array)

    if len(sorted_final_1d_indx_array) > 0:
         print("Binary", sorted_final_1d_indx_array)

    return sorted_final_1d_indx_array
