

def change_spin_magnitudes(prograde_bh_spins, frac_Eddington_ratio, spin_torque_condition, timestep):
    """Update the spin magnitude of the embedded black holes based on their accreted mass
        in this timestep.

    Parameters
    ----------
    prograde_bh_spins : float array
        initial spins of black holes in prograde orbits around SMBH
    frac_Eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stellar mass 
        black hole in units of Eddington accretion rate. 1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
    spin_torque_condition : float
        user chosen input set by input file; fraction of initial mass required to be 
        accreted before BH spin is torqued fully into alignment with the AGN disk. 
        We don't know for sure but Bogdanovic et al. says between 0.01=1% and 0.1=10% 
        is what is required
    timestep : float
        length of timestep in units of years

    Returns
    -------
    bh_new_spins : float array
        spin magnitudes of black holes after accreting at prescribed rate for one timestep
    """

    #def change_spin_magnitudes(bh_spins,prograde_orb_ang_mom_indices,frac_Eddington_ratio,spin_torque_condition,mass_growth_Edd_rate,timestep):
    #bh_new_spins=bh_spins
    normalized_Eddington_ratio = frac_Eddington_ratio/1.0
    normalized_timestep = timestep/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1
   
    #bh_new_spins[prograde_orb_ang_mom_indices]=bh_new_spins[prograde_orb_ang_mom_indices]+(4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
    bh_new_spins = prograde_bh_spins + (4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
    # TO DO: Include a condition to keep a maximal (a=+0.98) spin BH at that value once it reaches it
    #Return updated new spins    
    return bh_new_spins


def change_spin_angles(prograde_bh_spin_angles, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep):
    """_summary_

    Parameters
    ----------
    prograde_bh_spin_angles : _type_
        _description_
    frac_Eddington_ratio : _type_
        _description_
    spin_torque_condition : _type_
        _description_
    spin_minimum_resolution : _type_
        _description_
    timestep : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    #Calculate change in spin angle due to accretion during timestep
    normalized_Eddington_ratio = frac_Eddington_ratio/1.0
    normalized_timestep = timestep/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1
   
    #bh_new_spin_angles[prograde_orb_ang_mom_indices]=bh_new_spin_angles[prograde_orb_ang_mom_indices]-(6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
    bh_new_spin_angles = prograde_bh_spin_angles-(6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
   
    #TO DO: Include a condition to keep spin angle at or close to zero once it gets there
    #Return new spin angles
    bh_new_spin_angles[bh_new_spin_angles<spin_minimum_resolution] = 0.0
    return bh_new_spin_angles


