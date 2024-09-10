import astropy.constants as const
import astropy.units as u

def normalize_tgw(smbh_mass):
    """Gravitational wave timescale normalization.

    Calculate the normalization for timescale of merger (in yrs) due to GW emission.
    From Peters(1964):
    
    t_gw approx (5/128)* c^5/G^3 *a_b^4/(M_{b}^{2}mu_{b}) assuming ecc=0.0.
    
    For a_b in units of r_g=GM_smbh/c^2 we find
    
    t_gw=(5/128)*(G/c^3)*(a/r_g)^{4} *(M_s^4)/(M_b^{2}mu_b)

    Put bin_mass_ref in units of 10Msun is a reference mass.
    reduced_mass in units of 2.5Msun.
    
    Parameters
    ----------
    smbh_mass : float
        Mass of the supermassive black hole.

    Returns
    -------
    float
        normalization to gravitational wave timescale
    """

    # check whether smbh_mass is dimension(al/less)
    # if isinstance(smbh_mass, )

    G = const.G
    c = const.c
    mass_sun = u.M_sun
    year = u.year #3.15e7 
    bin_mass_ref = 10.0 # solar masses
    reduced_mass = 2.5  # reduced mass
    norm = (5.0/128.0)*(G/(c**(3)))*(smbh_mass**(4))*mass_sun/((bin_mass_ref**(2))*reduced_mass)
    t_gw_norm = norm/year

    return t_gw_norm.si.value

    # G = 6.7e-11
    # c = 3.0e8
    # M_sun = 2.e30
    # M_b = 10.0
    # reduced_mass = 2.5
    # year = 3.15e7
    # norm = (5.0/128.0)*(G/(c**(3.0)))*(smbh_mass**(4.0))*M_sun/((M_b**(2.0))*reduced_mass)
    # t_gw_norm = norm/year
    
    # return t_gw_norm