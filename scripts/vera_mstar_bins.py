#!/usr/bin/env python3
######## Globals ########
N_IT = 100
MSTAR_MIN = 1e9
MSTAR_MAX = 1e12
N_BINS = 20

######## Imports ########
import numpy as np
import os
from os.path import expanduser, join, isfile, isdir
from basil.relations import Neumayer_early_NSC_mass, Neumayer_late_NSC_mass
from basil.relations import SchrammSilvermanSMBH_mass_of_GSM as SMBH_mass_of_GSM

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iterations", default=N_IT)
    parser.add_argument("--mstar-min", default=MSTAR_MIN)
    parser.add_argument("--mstar-max", default=MSTAR_MAX)
    parser.add_argument("--nbins", default=N_BINS)
    parser.add_argument("--wkdir", default='./run_many')
    parser.add_argument("--mcfacts-exe", default="./scripts/mcfacts_sim.py")
    parser.add_argument("--vera-plots-exe", default="./scripts/vera_plots.py")
    parser.add_argument("--fname-nal", default=join(expanduser("~"), "Repos", "nal-data", "GWTC-2.nal.hdf5" ))
    # Handle top level working directory
    opts = parser.parse_args()
    if not isdir(opts.wkdir):
        os.mkdir(opts.wkdir)
    assert isdir(opts.wkdir)
    opts.wkdir=os.path.abspath(opts.wkdir)
    # Check exe
    assert isfile(opts.mcfacts_exe)
    return opts

######## Main ########
def main():
    opts = arg()
    mstar_arr = np.logspace(np.log10(opts.mstar_min),np.log10(opts.mstar_max), opts.nbins)
    SMBH_arr = SMBH_mass_of_GSM(mstar_arr)
    NSC_early_arr = Neumayer_early_NSC_mass(mstar_arr)
    NSC_late_arr = Neumayer_late_NSC_mass(mstar_arr)
    os.mkdir(join(opts.wkdir, 'early'))
    os.mkdir(join(opts.wkdir, 'late'))
    # Mstar loop
    for i in range(opts.nbins):
        mstar = mstar_arr[i]
        mass_smbh = SMBH_arr[i]
        early_mass = NSC_early_arr[i]
        late_mass = NSC_late_arr[i]
        mstar_str = "%.8f"%np.log10(mstar)
        early_dir = join(opts.wkdir, 'early', mstar_str)
        os.mkdir(early_dir)
        late_dir = join(opts.wkdir, 'late', mstar_str)
        os.mkdir(late_dir)
        # Make early iterations
        cmd = "python3 %s --n_iterations %d --fname-log out.log --work-directory %s --mass_smbh %f --M_nsc %f"%(
            opts.mcfacts_exe, opts.n_iterations, early_dir, mass_smbh, early_mass)
        print(cmd)
        os.system(cmd)
        # Make plots for early iterations
        cmd = "python3 %s --fname-mergers %s/output_mergers_population.dat --fname-nal %s --cdf chi_eff chi_p M gen1 gen2 t_merge"%(
            opts.vera_plots_exe, early_dir, opts.fname_nal)
        print(cmd)
        os.system(cmd)
        # Make late iterations
        cmd = "python3 %s --n_iterations %d --fname-log out.log --work-directory %s --mass_smbh %f --M_nsc %f"%(
            opts.mcfacts_exe, opts.n_iterations, late_dir, mass_smbh, late_mass)
        print(cmd)
        os.system(cmd)
        # Make plots for late iterations
        cmd = "python3 %s --fname-mergers %s/output_mergers_population.dat --fname-nal %s --cdf chi_eff chi_p M gen1 gen2 t_merge"%(
            opts.vera_plots_exe, late_dir, opts.fname_nal)
        print(cmd)
        os.system(cmd)
        
    return

######## Execution ########
if __name__ == "__main__":
    main()
