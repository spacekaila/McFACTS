import numpy as np
import scipy
from mcfacts.mcfacts_random_state import rng

"""Actual binary creation module

Make binaries using incoming objects, as decided by other functions
elsewhere in the code, in the hillsphere module.
"""

def add_to_binary_array2(
        disk_bins_bhbh,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_spins,
        disk_bh_pro_spin_angles,
        disk_bh_pro_gens,
        disk_bin_bhbh_pro_indices,
        bindex,
        fraction_bin_retro,
        smbh_mass
        ):
    """Create new BH binaries with appropriate parameters.
    
    We take the semi-maj axis, masses, spins, spin angles and generations
    from the relevant singletons, found in hillsphere.binary_check2, and sort
    those parameters into disk_bins_bhbh. We then ADD additional parameters
    relevant only for binaries, including semi-major axis of the binary,
    semi-major axis of the orbit of the center of mass of the binary around
    the SMBH, a flag to permit or suppress retrograde binaries, eventually
    eccentricity and inclination.

    Parameters
    ----------
    disk_bins_bhbh : [21, bindex] mixed array
        binary black hole array, multi-dimensional;
            [0,j]: float
            location of object 1 at formation of binary (distance from SMBH in
            r_g)
            [1,j]: float
            location of object 2 at formation of binary (distance from SMBH in
            r_g)
            [2,j]: float
            mass of obj 1 at time t in units of solar masses
            [3,j]: float
            mass of obj 2 at time t in units of solar masses
            [4,j]: float
            dimensionless spin magnitude of obj 1 at time t
            [5,j]: float
            dimensionless spin magnitude of obj 2 at time t
            [6,j]: float
            spin angle of obj 1 wrt disk gas in radians at time t
            [7,j]: float
            spin angle of obj 2 wrt disk gas in radians at time t
            [8,j]: float
            binary semi-major axis around c.o.m. in units of r_g of SMBH
            [9,j]: float
            binary center of mass location wrt SMBH in r_g
            [10,j]: float
            time to merger through GW alone (not set here)
            [11,j]: int
            merger flag = -2 if merging this timestep, else = 0 (not set here)
            [12,j]: float
            time of merger if binary has already merged (not set here)
            [13,j]: float
            binary eccentricity around binary center of mass
            [14,j]: int
            generation of obj 1 (1=natal black hole, no prior mergers)
            [15,j]: int
            generation of obj 2 (1=natal black hole, no prior mergers)
            [16,j]: int
            binary angular momentum switch +1/-1 for pro/retrograde
            [17,j]: float
            binary orbital inclination
            [18,j]: float
            binary orbital eccentricity of binary center of mass around SMBH
            [19,j]: float
            GW frequency of binary
            nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
            [20,j]: float
            GW strain of binary
            h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
            where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
            For local distances, approx 
            d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc
            assume 1Mpc = 3.1e22m.
            From Ned Wright's calculator
            (https://www.astro.ucla.edu/~wright/CosmoCalc.html)
            (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
    disk_bh_pro_orbs_a : float array
        locations of prograde singleton BH at start of timestep in units of
        gravitational radii (r_g=GM_SMBH/c^2)
    disk_bh_pro_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar
        masses
    disk_bh_pro_spins : float array
        dimensionless spin magnitude of prograde singleton BH at start of
        timestep
    disk_bh_pro_spin_angles : float array
       spin angle of prograde singleton BH wrt the gas disk angular momentum
       in radians at start of time step
    disk_bh_pro_gens : int array
        generation of prograde singleton BH (1=no previous merger history)
    disk_bin_bhbh_pro_indices : [2,N] int array
        array of indices corresponding to locations in disk_bh_pro_orbs_a,
        disk_bh_pro_masses, disk_bh_pro_spins, disk_bh_pro_spin_angles, and
        disk_bh_pro_gens which corresponds to binaries that form in this
        timestep.
        it has a length of the number of binaries to form (N) and a width of 2.
    bindex : int
        counter for length of disk_bins_bhbh before adding new binaries
    fraction_bin_retro : float
        fraction of binaries which form retrograde (wrt to the disk gas)
        around their own center of mass.
        = 0.0 turns all retrograde BBH at formation into prograde BBH. 
        = 0.5 half of the binaries will be retrograde
        = 1.0 all binaries will be retrograde.
    smbh_mass : float
        mass of supermassive black hole in units of solar masses

    Returns
    -------
    disk_bins_bhbh : [21, bindex+N] float array
        as for input, but updated (thus longer), to include newly formed
        binaries
    """
    # get length of current binary array before any additions
    # --we will need to do this, but for now we are using bindex bc too
    #   complex to fix now and appears many places
    # find number of new binaries based on indices from hillsphere.binary_check
    num_new_bins = np.shape(disk_bin_bhbh_pro_indices)[1]

    # If there are new binaries, actually form them!
    if num_new_bins > 0:
        # send close encounter indices to new array
        array_of_indices = disk_bin_bhbh_pro_indices
        bincount = 0
        # for all the new binaries that need to be created
        for j in range(bindex, bindex + num_new_bins):
            # for each member of the binary
            for i in range(0,2):
                # pick the [N,0] or [N,1] index for member 1 and member 2 of
                # binary N
                if num_new_bins == 1:
                    thing1 = array_of_indices[i]
                else:
                    thing1 = array_of_indices[i][bincount]
                
                disk_bins_bhbh[i,j] = disk_bh_pro_orbs_a[thing1]
                disk_bins_bhbh[i+2,j] = disk_bh_pro_masses[thing1]
                disk_bins_bhbh[i+4,j] = disk_bh_pro_spins[thing1]
                disk_bins_bhbh[i+6,j] = disk_bh_pro_spin_angles[thing1]
                # For new binary create initial binary semi-major axis
                temp_loc_1 = disk_bins_bhbh[0,j]
                temp_loc_2 = disk_bins_bhbh[1,j]
                temp_bin_separation = np.abs(temp_loc_1 - temp_loc_2)
                disk_bins_bhbh[8,j] = temp_bin_separation
                # Binary c.o.m.= location_1 + separation*M_2/(M_1+M_2)
                temp_mass_1 = disk_bins_bhbh[2,j]
                temp_mass_2 = disk_bins_bhbh[3,j]
                temp_bin_mass = temp_mass_1 + temp_mass_2
                disk_bins_bhbh[9,j] = temp_loc_1 + \
                    (temp_bin_separation*temp_mass_2/temp_bin_mass)
                # Set up binary eccentricity around its own center of mass.
                # Draw uniform value btwn [0,1]
                disk_bins_bhbh[13,j] = rng.random()
                # Set up binary member generations
                disk_bins_bhbh[i+14,j] = disk_bh_pro_gens[thing1]
                # Set up bin orb. ang. mom.
                # (randomly +1 (pro) or -1(retrograde))
                # random number between [0,1]
                random_uniform_number = rng.random()
                # Generate a random number between [-retro,1-retro] where
                # retro is the fraction of BBH that are retrograde.
                # For default retro =0.1, range is [-0.1,0.9] and 90% of BBH
                # are prograde.
                bh_initial_orb_ang_mom = random_uniform_number - \
                    fraction_bin_retro
                # If retro =0, range = [0,1] and set L_bbh = +1.
                if fraction_bin_retro == 0:
                    bh_initial_orb_ang_mom = 1
                # If retro =0.1 (default), range = [-0.1,0.9].
                # If range <0, L_BBH = -1; if range >0, L_BBH = +1
                if fraction_bin_retro > 0:
                    if bh_initial_orb_ang_mom < 0:
                        bh_initial_orb_ang_mom = -1
                    if bh_initial_orb_ang_mom > 0:
                        bh_initial_orb_ang_mom = 1

                disk_bins_bhbh[16,j] = bh_initial_orb_ang_mom                
                # Set up binary inclination (in units radians). Will want this
                # to be pi radians if retrograde.
                disk_bins_bhbh[17,j] = 0
                # Set up binary orbital eccentricity of com around SMBH.
                # Assume initially v.small (e~0.01)
                disk_bins_bhbh[18,j] = 0.01
                # Set up binary GW frequency
                # nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
                #1rg =1AU=1.5e11m for 1e8Msun
                rg = 1.5e11*(smbh_mass/1.e8)
                temp_bin_separation_meters = temp_bin_separation*rg
                temp_mass_1_kg = 2.e30*temp_mass_1
                temp_mass_2_kg = 2.e30*temp_mass_2
                temp_bin_mass_kg = 2.e30*temp_bin_mass
                m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3/5))/ \
                    (temp_bin_mass_kg**(1/5))
                rg_chirp = (scipy.constants.G * m_chirp)/ \
                    (scipy.constants.c**(2.0))
                if temp_bin_separation_meters < rg_chirp:
                    temp_bin_separation_meters = rg_chirp
                
                # compute GW frequency & strain for each binary
                nu_gw = (1.0/scipy.constants.pi)*np.sqrt(
                    temp_bin_mass_kg *
                    scipy.constants.G /
                    (temp_bin_separation_meters**(3.0))
                    )
                disk_bins_bhbh[19,j] = nu_gw
                Mpc = 3.1e22 # This is terrible use astropy
                d_obs = 421*Mpc
                strain = (4/d_obs)*rg_chirp * \
                    (np.pi*nu_gw*rg_chirp/scipy.constants.c)**(2/3)
                disk_bins_bhbh[20,j] = strain
            bincount = bincount + 1
        
    return disk_bins_bhbh
