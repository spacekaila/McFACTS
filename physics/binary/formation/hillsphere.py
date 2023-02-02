def calculate_hill_sphere(prograde_bh_locations,prograde_bh_masses,mass_smbh):
    #Return the Hill sphere radius (R_Hill) for an array of prograde BH where
    # R_Hill=a(q/3)^1/3 where a=semi-major axis, q=m_bh/M_SMBH
    
    bh_smbh_mass_ratio=prograde_bh_masses/(3.0*mass_smbh)
    mass_ratio_factor=(bh_smbh_mass_ratio)**(1./3.)
    bh_hill_sphere=prograde_bh_locations*mass_ratio_factor
    #Return the BH Hill sphere radii for all orbiters. Prograde should have much larger Hill sphere
    return bh_hill_sphere