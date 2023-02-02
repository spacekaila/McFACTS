def bin_tgw(mass1,mass2,separation,eccentricity):
    # Calculate the t_gw from Peters(1964), which is
    # t_gw~ (5/128)*(c^5/G^3)*(a^4/M_b^2 mu_b)*(1-e^2)^(7/2)
    # where M_b=M1+M2 (bin mass) and mu_b=M1M2/(M1+M2) the reduced binary mass
    # and can be parameterized as
    # t_gw=6.1e12yr(1-e^2)^(7/2)*(a/10^3r_g)^4 (M_smbh/10^8Msun)^3 (m_bh/10Msun)^-1
    scaled_location=1.e3
    scaled_mass=10.0
    mass_smbh=1.e8
    scaled_smbh_mass=1.e8
    #normalized_locations=bh_locations/scaled_location
    #normalized_masses=bh_masses/scaled_mass
    normalized_smbh_mass=mass_smbh/scaled_smbh_mass
    #eccentricity factor (1-e^2)^7/2 is
    #ecc_factor=(1.0-(bh_orbital_eccentricities)**(2.0))**(3.5)
    #time_for_gw_decay=6.1e12*ecc_factor*(normalized_locations**4.0)*(normalized_smbh_mass**2.0)*(1.0/normalized_masses)
    #return time_for_gw_decay
    return

def normalize_tgw(mass_smbh):
    #Calculate the normalization for timescale of merger (in yrs) due to GW emission.
    #From Peters(1964) t_gw approx (5/128)* c^5/G^3 *a_b^4/(M_{b}^{2}mu_{b})
    # assuming ecc=0.0
    #For a_b in units of r_g=GM_smbh/c^2 we find
    #t_gw=(5/128)*(G/c^3)*(a/r_g)^{4} *(M_s^4)/(M_b^{2}mu_b)
    #Put M_b in units of 10Msun, mu_b = in units of 2.5Msun.
    G=6.7e-11
    c=3.0e8
    M_sun=2.e30
    M_b=10.0
    mu_b=2.5
    year=3.15e7
    norm=(5.0/128.0)*(G/(c**(3.0)))*(mass_smbh**(4.0))*M_sun/((M_b**(2.0))*mu_b)
    t_gw_norm=norm/year
    return t_gw_norm