import numpy as np

names_rec = [
             'iter',
             'CM', 'M', 'chi_eff', 'a_tot',
             'spin_angle', 'm1', 'm2', 'a1',
             'a2', 'theta1', 'theta2', 'gen1',
             'gen2', 't_merge', 'chi_p']
dtype_rec = np.dtype( [(x, float) for x in names_rec])

def merged_bh(merged_array,bin_array,merger_indices,i,chi_here,mass_here,spin_here,nprops_mergers,n_mergers_so_far,chi_p,time_passed):
    """
    Recording merger event.  Designed only to handle one event at a time, with hardcoded labels
    """
    #Comment here
    #Return an array with properties of merger
    #Center of mass, M_total, Chi_eff, a_tot, spin_angle, m1,m2,a1,a2,theta1,theta2,gen1,gen2,time of merger
    
    #merger_indices = np.array(merger_indices_here)
    #print(merger_indices)
    #print("length merger indices",len(merger_indices))
    #n_mergers_so_far=int(num_mergers)
    #n_mergers_this_timestep = len(merger_indices)

    #if len(merger_indices)>1:   # deal with the fact that we have multiple mergers
    #    print(" Multiple mergers - we are about to lose events ")
    #    print(mass_here, chi_here, spin_here)
    #    merger_indices = merger_indices[0]
    #    if isinstance(mass_here, np.ndarray):
    #        mass =  mass_here[0]
    #    else:
    #        mass = mass_here
    #    if isinstance(spin_here, np.ndarray):
    #        spin = spin_here[0]
    #        chi = chi_here[0]
    #    else:
    #        spin = spin_here
    #        chi = chi_here
    #else:
    mass= mass_here
    spin = spin_here
    chi = chi_here
    
    #num_mergers_this_timestep = len(merger_indices)
                
    #print("num mergers this timestep",n_mergers_this_timestep)
    #print("n_mergers_so_far + i",n_mergers_so_far + i)    
    #for i in range (0,n_mergers_this_timestep):
        #            merged_bh_com = merged_bh_array[0,n_mergers_so_far + i]
    merged_array[0,n_mergers_so_far + i] = bin_array[9,merger_indices[i]]
    merged_array[1,n_mergers_so_far + i] = mass
    merged_array[2,n_mergers_so_far + i] = chi
    merged_array[3,n_mergers_so_far + i] = spin
    merged_array[4,n_mergers_so_far + i] = 0.0
    merged_array[5,n_mergers_so_far + i] = bin_array[2,merger_indices[i]]
    merged_array[6,n_mergers_so_far + i] = bin_array[3,merger_indices[i]]
    merged_array[7,n_mergers_so_far + i] = bin_array[4,merger_indices[i]]
    merged_array[8,n_mergers_so_far + i] = bin_array[5,merger_indices[i]]
    merged_array[9,n_mergers_so_far + i] = bin_array[6,merger_indices[i]]
    merged_array[10,n_mergers_so_far + i] = bin_array[7,merger_indices[i]]
    merged_array[11,n_mergers_so_far + i] = bin_array[14,merger_indices[i]]
    merged_array[12,n_mergers_so_far + i] = bin_array[15,merger_indices[i]]
#    merged_array[13,n_mergers_so_far + i] = bin_array[12,merger_indices[i]]
    merged_array[13,n_mergers_so_far + i] = time_passed
    merged_array[14,n_mergers_so_far + i] = chi_p
    return merged_array[:,n_mergers_so_far + i]
