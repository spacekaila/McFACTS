from mcfacts.setup import setupdiskstars
from mcfacts.setup import diskstars_hillspheremergers
from mcfacts.objects.agnobject import AGNStar
import numpy as np

def init_single_stars(opts,id_start_val = None):

    # Generate initial number of stars
    n_stars_initial = setupdiskstars.setup_disk_nstars(
            opts.nsc_mass,
            opts.nsc_ratio_bh_num_star_num,
            opts.nsc_ratio_bh_mass_star_mass,
            opts.nsc_radius_outer,
            opts.nsc_density_index_outer,
            opts.smbh_mass,
            opts.disk_radius_outer,
            opts.disk_aspect_ratio_avg,
            opts.nsc_radius_crit,
            opts.nsc_density_index_inner,
        )
    #print(n_stars_initial) 152_248_329
    n_stars_initial = 1_000_000
    
    # Generate initial masses for the initial number of stars, pre-Hill sphere mergers
    masses_initial = setupdiskstars.setup_disk_stars_masses(n_star = n_stars_initial,
                                                            min_initial_star_mass = opts.disk_star_mass_min_init,
                                                            max_initial_star_mass = opts.disk_star_mass_max_init,
                                                            mstar_powerlaw_index = opts.nsc_imf_star_powerlaw_index)
    
    # Generating star locations in an x^2 distribution
    x_vals = np.random.uniform(0.002, 1, n_stars_initial)
    r_locations_initial = np.sqrt(x_vals)
    r_locations_initial_scaled = r_locations_initial*opts.disk_radius_trap

    # Sort the mass and location arrays by the location array
    sort_idx = np.argsort(r_locations_initial_scaled)
    r_locations_initial_sorted = r_locations_initial_scaled[sort_idx]
    masses_initial_sorted = masses_initial[sort_idx]

    masses_stars, r_locations_stars = diskstars_hillspheremergers.hillsphere_mergers(n_stars=n_stars_initial,
                                                                                     masses_initial_sorted=masses_initial_sorted,
                                                                                     r_locations_initial_sorted=r_locations_initial_sorted,
                                                                                     min_initial_star_mass=opts.disk_star_mass_min_init,
                                                                                     R_disk = opts.disk_radius_trap,
                                                                                     M_smbh=opts.smbh_mass,
                                                                                     P_m = 1.35,
                                                                                     P_r = 1.)
    n_stars = len(masses_stars)
    
    star_radius = setupdiskstars.setup_disk_stars_radii(masses_stars)
    star_spin = setupdiskstars.setup_disk_stars_spins(n_stars, opts.nsc_star_spin_dist_mu, opts.nsc_star_spin_dist_sigma)
    star_spin_angle = setupdiskstars.setup_disk_stars_spin_angles(n_stars, star_spin)
    star_orbit_inclination = setupdiskstars.setup_disk_stars_inclination(n_stars)
    #star_orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(rng,n_stars)
    star_orbit_argperiapse = setupdiskstars.setup_disk_stars_arg_periapse(n_stars)
    if opts.flag_orb_ecc_damping == 1:
        star_orbit_e = setupdiskstars.setup_disk_stars_eccentricity_uniform(n_stars)
    else:
        star_orbit_e = setupdiskstars.setup_disk_stars_circularized(n_stars, opts.disk_bh_pro_orb_ecc_crit)
    star_Y = opts.nsc_star_metallicity_y_init
    star_Z = opts.nsc_star_metallicity_z_init

    stars = AGNStar(mass=masses_stars,
                    spin=star_spin,
                    spin_angle=star_spin_angle,
                    orbit_a=r_locations_stars, #this is location
                    orbit_inclination=star_orbit_inclination,
                    orbit_e=star_orbit_e,
                    #orb_ang_mom= star_orb_ang_mom,
                    orbit_arg_periapse=star_orbit_argperiapse,
                    star_radius=star_radius,
                    star_Y=star_Y,
                    star_Z=star_Z,
                    mass_smbh=opts.smbh_mass,
                    id_start_val=id_start_val,
                    n_stars=n_stars)
   
    return(stars,n_stars)
