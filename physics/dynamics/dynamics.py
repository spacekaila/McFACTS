import numpy as np

import scipy

def circular_singles_encounters_prograde(rng, mass_smbh, prograde_bh_locations, prograde_bh_masses, disk_surf_model, disk_aspect_ratio_model, bh_orb_ecc, timestep, crit_ecc, de):
    """"Return array of modified singleton BH orbital eccentricities perturbed by encounters within f*R_Hill, where f is some fraction/multiple of Hill sphere radius R_H
    
    Assume encounters with BH with e<e_crit are the only important ones for now. 
    Since the e<e_crit population is the most likely BBH merger source.
    
    First, find those orbiters with e<e_crit and their
    associated semi-major axes a_circ =[a_circ1, a_circ2, ..] and masses m_circ =[m_circ1,m_circ2, ..]
    Calculate orbital timescale for a_circ1 and a_i and N_orbits/timestep. T_orb =2pi sqrt(a^3/GM_smbh)
    a^3/GM = (10^3r_g)^3/GM_smbh = 10^9 (GM_smbh/c^2)^3/GM_smbh = 10^9 G^2 M_smbh^2/c^6 =10^9 G^2M^2smbh/c^6
    =2pi 10^4.5 GM/c^3 = 2pi 10^4.5 6.7e-11*2e38/(3e8)^3 = 2pi 10^4.5 13.6e27/27e24 =pi 10^7.5 ~ 3yr
    E.g. Orbit~3yr at 10^3r_g around a 10^8Msun SMBH. So 3,000 orbits in timestep=1e4yr.

    Among population of orbiters with e>e_crit, find those orbiters i where a_i*(1-e_i)<a_circ1<a_i*(1-e_i) and
    allow possible interactions.
    
    Calculate mutual Hill sphere R_H of binary (a_circ1,a_i).
    Calculate ratio of 2R_H of binary to size of circular orbit. 
    Thus (2R_H/2pi a_circ1)= odds that a_circ1 is in the region of cross-over per orbit.
    Say R_H ~0.004a_circ1 then ratio is 2*0.004/2*pi~1.e0-3, ie. 1/1000 odds that BH at a_circ1 is in either area of overlap per orbit.
    Then calculate number of orbits of a_i in 1 timestep. If e.g. 500 orbits per timestep of 10kyr, then 
    probability of encounter = (500orbits/timestep)*(R_H/pi a_circ1) ~ 0.5
    Draw a random number from the uniform [0,1] distribution and 
        if rng < probability of encounter, there is an encounter during the timestep
        if rng > probability of encounter, there is no encounter during the timestep

    If encounter:
        Take energy from high ecc. a_i and give to a_circ1
        Assume circ1 keeps a_circ1, but adds ecc (about 20%), extra K.E. 
        Treat interloper as a perturber with average close pass =0.5R_Hill and 
        assume average energy transfer = 20% perturbation (from Sigurdsson & Phinney 1993). 
        This K.E. perturbation goes into additional eccentricity. e-> 0.2 from e=0.01 for same a.
        Take energy from (a_i,e_i). Drop e_i by 20% for now, keeping a_i const (in future do this self-consistently)
        E=-GMm_circ1/a_circ1 = 1/2m_circ1*v_kep^2
        also
        sigma_ecc = sqrt(ecc^2 + incl^2)v_kep so if incl=0 deg (for now)
        En of ecc. interloper = 1/2 m_i sigma_ecc^2.
            Note: Can also use above logic for binary encounters except use binary binding energy instead.
        
        or later could try 
            Deflection angle defl = tan (defl) = dV_perp/V = 2GM/bV^2 kg^-1 m^3 s^-2 kg / m (m s^-1)^2
        so de/e =2GM/bV^2 = 2 G M_bin/0.5R_hill*sigma^2
        and R_hill = a_circ1*(M_bin/3M_smbh)^1/3 and sigma^2 =ecc^2*v_kep^2
        So de/e = 4GM_bin/a_circ1(M_bin/3M_smbh)^1/3 ecc^2 v_kep^2 
        and v_kep = sqrt(GM_smbh/a_i)
        So de/e = 4GM_bin^(2/3)M_smbh^1/3 a_i/a_circ1 ecc^2 GM_smbh = 4(M_bin/M_smbh)^(2/3) (a_i/a_circ1)(1/ecc^2)
        where V_rel = sigma say and b=R_H = a_circ1 (q/3)^1/3
        So defl = 2GM/ a_circ1(q/3)^2/3 ecc^2 10^14 (m/s)^2 (R/10^3r_g)^-1
                = 2 6.7e-11 2.e31/  
    Note! when doing this for binaries. Calculate velocity of encounter compared to a_bin.
    If binary is hard ie GM_bin/a_bin > m3v_rel^2 then:
      harden binary to a_bin = a_bin -da_bin and
      new binary eccentricity e_bin = e_bin + de and 
      and give  da_bin worth of binding energy to extra eccentricity of m3.
    If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
      soften binary to a_bin = a_bin + da_bin and
      new binary eccentricity e_bin = e_bin + de
      and take da_bin worth of binary energy from eccentricity of m3. 


        Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_model : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    bh_orb_ecc : float array
        orbital eccentricity of singleton BH     
    timestep : float
        size of timestep in years
    de : float
        average energy change per strong encounter    
        
        Returns
    -------
    bh_new_loc_orb_ecc : float array
        updated bh locations and orbital eccentricities perturbed by dynamics
        """
    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_model, float):
        disk_surface_density = disk_surf_model
    else:
        disk_surface_density = disk_surf_model(prograde_bh_locations)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_model, float):
        disk_aspect_ratio = disk_aspect_ratio_model
    else:
        disk_aspect_ratio = disk_aspect_ratio_model(prograde_bh_locations)
    #Set up new_bh_orb_ecc
    new_bh_orb_ecc=np.empty_like(bh_orb_ecc)
    
    #Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7 
    mass_ratio = prograde_bh_masses/mass_smbh
    
    #normalized_mass_ratio = mass_ratio/10**(-7)
    #normalized_bh_locations = prograde_bh_locations/1.e4
    #normalized_disk_surf_model = disk_surface_density/1.e5
    #normalized_aspect_ratio = disk_aspect_ratio/0.03

    #Assume all incoming eccentricities are prograde (for now)
    prograde_bh_orb_ecc = bh_orb_ecc

    #Find the e< critical ecc. population
    circ_prograde_population = np.ma.masked_where(prograde_bh_orb_ecc > crit_ecc, prograde_bh_orb_ecc)
    ecc_prograde_population = np.ma.masked_where(prograde_bh_orb_ecc < crit_ecc, prograde_bh_orb_ecc)
    #print('Circ prograde',circ_prograde_population)
    #Find the indices of the e<crit_ecc population     
    circ_prograde_population_indices = np.ma.nonzero(circ_prograde_population)
    ecc_prograde_population_indices = np.ma.nonzero(ecc_prograde_population)
    #print('Circ indices',circ_prograde_population_indices)
    #Find their locations and masses
    circ_prograde_population_locations = prograde_bh_locations[circ_prograde_population_indices]
    circ_prograde_population_masses = prograde_bh_masses[circ_prograde_population_indices]
    #print('Circ locations',circ_prograde_population_locations)
    #print('Circ masses',circ_prograde_population_masses)
    #Ecc locs
    ecc_prograde_population_locations = prograde_bh_locations[ecc_prograde_population_indices]
    ecc_prograde_population_masses = prograde_bh_masses[ecc_prograde_population_indices]

    #T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2) 
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = scipy.constants.pi*((circ_prograde_population_locations)**(1.5))*(2.e30*mass_smbh*scipy.constants.G)/(scipy.constants.c**(3.0)*3.15e7) 
    N_circ_orbs_per_timestep = timestep/orbital_timescales_circ_pops
    #print('Orb. timescales (yr)' , orbital_timescales_circ_pops)
    #print('N_circ_orb/timestep' , N_circ_orbs_per_timestep)
    ecc_orb_min = prograde_bh_locations[ecc_prograde_population_indices]*(1.0-prograde_bh_orb_ecc[ecc_prograde_population_indices])
    ecc_orb_max = prograde_bh_locations[ecc_prograde_population_indices]*(1.0+prograde_bh_orb_ecc[ecc_prograde_population_indices])
    #print('min',ecc_orb_min)
    #print('max',ecc_orb_max)
    #print('len(circ_locns)',len(circ_prograde_population_locations))
    num_poss_ints = 0
    num_encounters = 0
    if len(circ_prograde_population_locations) > 0:
            for i in range(0,len(circ_prograde_population_locations)):    
                for j in range (0,len(ecc_prograde_population_locations)):
                    if circ_prograde_population_locations[i] < ecc_orb_max[j] and circ_prograde_population_locations[i] > ecc_orb_min[j]:
                        #print("Possible Interaction!")
                        #print(prograde_bh_locations[j],prograde_bh_orb_ecc[j],ecc_orb_min[j],circ_prograde_population_locations[i],ecc_orb_max[j])
                        #prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                        # r_h = a_circ1(temp_bin_mass/3mass_smbh)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                        temp_bin_mass = circ_prograde_population_masses[i] + ecc_prograde_population_masses[j]
                        bh_smbh_mass_ratio = temp_bin_mass/(3.0*mass_smbh)
                        mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)                        
                        prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                        prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                        if prob_enc_per_timestep > 1:
                            prob_enc_per_timestep = 1
                        random_uniform_number = np.random.uniform(0,1)
                        if random_uniform_number < prob_enc_per_timestep:
                            #print('Encounter!!',random_uniform_number,prob_enc_per_timestep, i, j)
                            #print(circ_prograde_population, prograde_bh_orb_ecc[j], circ_prograde_population_locations)
                            #print(circ_prograde_population_indices,i,circ_prograde_population_indices[0])
                            indx_array = circ_prograde_population_indices[0]
                            #print(prograde_bh_orb_ecc[indx_array[i]])
                            num_encounters = num_encounters + 1
                            # if close encounter, pump ecc of circ orbiter to e=0.2 from near circular, and incr a_circ1 by 20%
                            # drop ecc of a_i by 20% and drop a_i by 20% (P.E. = -GMm/a)
                            # if already pumped in eccentricity, no longer circular, so don't need to repeat.
                            if prograde_bh_orb_ecc[indx_array[i]] <= crit_ecc:
                                #print(prograde_bh_orb_ecc[indx_array[i]],prograde_bh_orb_ecc[j])
                                prograde_bh_orb_ecc[indx_array[i]] = de
                                prograde_bh_locations[indx_array[i]] = prograde_bh_locations[indx_array[i]]*(1.0 + de)
                                prograde_bh_orb_ecc[j] = prograde_bh_orb_ecc[j]*(1 - de)
                                prograde_bh_locations[j] = prograde_bh_locations[j]*(1 - de)
                                #print(prograde_bh_orb_ecc[indx_array[i]],prograde_bh_orb_ecc[j])
                        
                        num_poss_ints = num_poss_ints + 1
                #print(i,num_poss_ints,num_encounters)
            num_poss_ints = 0
            num_encounters = 0
    
    prograde_bh_locn_orb_ecc = [[prograde_bh_locations],[prograde_bh_orb_ecc]]    
    return prograde_bh_locn_orb_ecc

def circular_binaries_encounters_prograde(mass_smbh, prograde_bh_locations, prograde_bh_masses, disk_surf_model, disk_aspect_ratio_model, bh_orb_ecc, timestep, crit_ecc):
    """"Return array of modified binary BH separations and eccentricities perturbed by encounters within f*R_Hill, where f is some fraction/multiple of Hill sphere radius R_H
    
    Given array of binaries at locations [a_bbh1,a_bbh2] with 
    binary semi-major axes [a_bin1,a_bin2,...] and binary eccentricities [e_bin1,e_bin2,...],
    find all the single BH at locations a_i that within timestep 
        either pass between a_i(1-e_i)< a_bbh1 <a_i(1+e_i)

    Calculate velocity of encounter compared to a_bin.
    If binary is hard ie GM_bin/a_bin > m3v_rel^2 then:
      harden binary to a_bin = a_bin -da_bin and
      new binary eccentricity e_bin = e_bin + de and 
      and give  da_bin worth of binding energy to extra eccentricity of m3.
    If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
      soften binary to a_bin = a_bin + da_bin and
      new binary eccentricity e_bin = e_bin + de
      and take da_bin worth of binary energy from eccentricity of m3. 
    If binary is unbound ie GM_bin/a_bin << m3v_rel^2 then:
      remove binary from binary array
      add binary components m1,m2 back to singleton arrays with new orbital eccentricities e_1,e_2 from energy of encounter.
      Equipartition energy so m1v1^2 =m2 v_2^2 and 
      generate new individual orbital eccentricities e1=v1/v_kep_circ and e_2=v_2/v_kep_circ
      Take energy put into destroying binary from orb. eccentricity of m3.  
        Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_model : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    bh_orb_ecc : float array
        orbital eccentricity of singleton BH     
    timestep : float
        size of timestep in years
        
        Returns
    -------
    bh_new_orb_ecc : float array
        updated orbital eccentricities damped by AGN gas
        """
    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_model, float):
        disk_surface_density = disk_surf_model
    else:
        disk_surface_density = disk_surf_model(prograde_bh_locations)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_model, float):
        disk_aspect_ratio = disk_aspect_ratio_model
    else:
        disk_aspect_ratio = disk_aspect_ratio_model(prograde_bh_locations)
    #Set up new_bh_orb_ecc
    new_bh_orb_ecc=np.empty_like(bh_orb_ecc)
    
    #Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7 
    mass_ratio = prograde_bh_masses/mass_smbh
    
    normalized_mass_ratio = mass_ratio/10**(-7)
    normalized_bh_locations = prograde_bh_locations/1.e4
    normalized_disk_surf_model = disk_surface_density/1.e5
    normalized_aspect_ratio = disk_aspect_ratio/0.03

    #Assume all incoming eccentricities are prograde (for now)
    prograde_bh_orb_ecc = bh_orb_ecc

   
    #Calculate (e/h) ratio for all prograde BH for use in eqn. 2 above
    e_h_ratio = prograde_bh_orb_ecc/disk_aspect_ratio
    
    # Modest orb eccentricities: e < 2h (experience simple exponential damping): mask entries > 2*aspect_ratio; only show BH with e<2h
    prograde_bh_modest_ecc = np.ma.masked_where(prograde_bh_orb_ecc > 2.0*disk_aspect_ratio_model(prograde_bh_locations),prograde_bh_orb_ecc)
    # Large orb eccentricities: e > 2h (experience more complicated damping)
    prograde_bh_large_ecc = np.ma.masked_where(prograde_bh_orb_ecc < 2.0*disk_aspect_ratio_model(prograde_bh_locations),prograde_bh_orb_ecc)
    #Indices of orb eccentricities where e<2h
    modest_ecc_prograde_indices = np.ma.nonzero(prograde_bh_modest_ecc)
    #Indices of orb eccentricities where e>2h
    large_ecc_prograde_indices = np.ma.nonzero(prograde_bh_large_ecc)
    #print('modest ecc indices', modest_ecc_prograde_indices)
    #print('large ecc indices', large_ecc_prograde_indices)
    #Calculate the 1-d array of damping times at all locations since we need t_damp for both modest & large ecc (see eqns above)
    t_damp =1.e5*(1.0/normalized_mass_ratio)*(normalized_aspect_ratio**4)*(1.0/normalized_disk_surf_model)*(1.0/np.sqrt(normalized_bh_locations))
    #timescale ratio for modest ecc damping
    modest_timescale_ratio = timestep/t_damp
    #timescale for large ecc damping from eqn. 2 above
    t_ecc = (t_damp/0.78)*(1 - (0.14*(e_h_ratio)**(2.0)) + (0.06*(e_h_ratio)**(3.0)))
    large_timescale_ratio = timestep/t_ecc
    #print("t_damp",timestep/t_damp)
    #print("t_ecc",timestep/t_ecc)
    
    #print("timescale_ratio",timescale_ratio)
    new_bh_orb_ecc[modest_ecc_prograde_indices] = bh_orb_ecc[modest_ecc_prograde_indices]*np.exp(-modest_timescale_ratio[modest_ecc_prograde_indices])
    new_bh_orb_ecc[large_ecc_prograde_indices] = bh_orb_ecc[large_ecc_prograde_indices]*np.exp(-large_timescale_ratio[large_ecc_prograde_indices])
    new_bh_orb_ecc = np.where(new_bh_orb_ecc<crit_ecc, crit_ecc,new_bh_orb_ecc)
    #print("Old ecc, New ecc",bh_orb_ecc,new_bh_orb_ecc)
    return new_bh_orb_ecc
