from mcfacts.setup import setupdiskstars
from mcfacts.setup import diskstars_hillspheremergers
from mcfacts.objects.agnobject import AGNStar
import numpy as np

def init_single_stars(opts,rng):

    # Generate initial number of stars
    n_stars_initial = setupdiskstars.setup_disk_nstars(
            opts.M_nsc,
            opts.nbh_nstar_ratio,
            opts.mbh_mstar_ratio,
            opts.r_nsc_out,
            opts.nsc_index_outer,
            opts.mass_smbh,
            opts.disk_outer_radius,
            opts.h_disk_average,
            opts.r_nsc_crit,
            opts.nsc_index_inner,
        )
    #print(n_stars_initial) 152_248_329
    n_stars_initial = 1_000_000
    
    # Generate initial masses for the initial number of stars, pre-Hill sphere mergers
    masses_initial = setupdiskstars.setup_disk_stars_masses(n_star = n_stars_initial,
                                                            min_initial_star_mass = opts.min_initial_star_mass,
                                                            max_initial_star_mass = opts.max_initial_star_mass,
                                                            mstar_powerlaw_index = opts.star_mass_powerlaw_index)
    
    # Generating star locations in an x^2 distribution
    x_vals = np.random.uniform(0.002, 1, n_stars_initial)
    r_locations_initial = np.sqrt(x_vals)
    r_locations_initial_scaled = r_locations_initial*opts.trap_radius

    # Sort the mass and location arrays by the location array
    sort_idx = np.argsort(r_locations_initial_scaled)
    r_locations_initial_sorted = r_locations_initial_scaled[sort_idx]
    masses_initial_sorted = masses_initial[sort_idx]

    masses_stars, r_locations_stars = diskstars_hillspheremergers.hillsphere_mergers(n_stars=n_stars_initial,
                                                   masses_initial_sorted=masses_initial_sorted,
                                                   r_locations_initial_sorted=r_locations_initial_sorted,
                                                   min_initial_star_mass=opts.min_initial_star_mass,
                                                   R_disk = opts.trap_radius,
                                                   M_smbh=opts.mass_smbh,
                                                   P_m = 1.35,
                                                   P_r = 1.)
    n_stars = len(masses_stars)
    
    star_radius = setupdiskstars.setup_disk_stars_radii(masses_stars)
    star_spin = setupdiskstars.setup_disk_stars_spins(n_stars, opts.mu_star_spin_distribution, opts.sigma_star_spin_distribution)
    star_spin_angle = setupdiskstars.setup_disk_stars_spin_angles(n_stars, star_spin)
    star_orbit_inclination = setupdiskstars.setup_disk_stars_inclination(n_stars)
    #star_orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(rng,n_stars)
    if opts.orb_ecc_damping == 1:
        star_orbit_e = setupdiskstars.setup_disk_stars_eccentricity_uniform(n_stars)
    else:
        star_orbit_e = setupdiskstars.setup_disk_stars_circularized(n_stars, opts.crit_ecc)
    star_Y = opts.stars_initial_Y
    star_Z = opts.stars_initial_Z

    stars = AGNStar(mass = masses_stars,
                        spin = star_spin,
                        spin_angle = star_spin_angle,
                        orbit_a = r_locations_stars, #this is location
                        orbit_inclination = star_orbit_inclination,
                        orbit_e = star_orbit_e,
                        #orb_ang_mom = star_orb_ang_mom,
                        star_radius = star_radius,
                        star_Y = star_Y,
                        star_Z = star_Z,
                        mass_smbh = opts.mass_smbh,
                        n_stars = n_stars)

    
    return(stars,n_stars)
