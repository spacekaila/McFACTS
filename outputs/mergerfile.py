import numpy as np


def merged_bh(merged_array,bin_array, merger_indices,chi,mass,spin,nprops_mergers,num_mergers):
    #Return an array with properties of merger
    #Center of mass, M_total, Chi_eff, a_tot, spin_angle, m1,m2,a1,a2,theta1,theta2,gen1,gen2,time of merger
    number_of_mergers=int(num_mergers)
    merged_array[0,number_of_mergers] = bin_array[9,merger_indices]
    merged_array[1,number_of_mergers] = mass
    merged_array[2,number_of_mergers] = chi
    merged_array[3,number_of_mergers] = spin
    merged_array[4,number_of_mergers] = 0.0
    merged_array[5,number_of_mergers] = bin_array[2,merger_indices]
    merged_array[6,number_of_mergers] = bin_array[3,merger_indices]
    merged_array[7,number_of_mergers] = bin_array[4,merger_indices]
    merged_array[8,number_of_mergers] = bin_array[5,merger_indices]
    merged_array[9,number_of_mergers] = bin_array[6,merger_indices]
    merged_array[10,number_of_mergers] = bin_array[7,merger_indices]
    merged_array[11,number_of_mergers] = bin_array[14,merger_indices]
    merged_array[12,number_of_mergers] = bin_array[15,merger_indices]
    merged_array[13,number_of_mergers] = bin_array[12,merger_indices]
    return merged_array