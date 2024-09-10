import numpy as np


def normalize_tgw(mass_smbh):
    #Calculate the normalization for timescale of merger (in yrs) due to GW emission.
    #From Peters(1964) t_gw approx (5/128)* c^5/G^3 *a_b^4/(M_{b}^{2}mu_{b})
    # assuming ecc=0.0
    #For a_b in units of r_g=GM_smbh/c^2 we find
    #t_gw=(5/128)*(G/c^3)*(a/r_g)^{4} *(M_s^4)/(M_b^{2}mu_b)
    #Put M_b in units of 10Msun, mu_b = in units of 2.5Msun.
    G = 6.7e-11
    c = 3.0e8
    M_sun = 2.e30
    M_b = 10.0
    mu_b = 2.5
    year = 3.15e7
    norm = (5.0/128.0)*(G/(c**(3.0)))*(mass_smbh**(4.0))*M_sun/((M_b**(2.0))*mu_b)
    t_gw_norm = norm/year
    
    return t_gw_norm