#!/usr/bin/env python
'''Plotting script for mcfacts quantities'''
######## Imports ########
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from os.path import join, basename
#### Local ####
from mcfacts.outputs.mcfacts_handler import McfactsHandler

######## Plots ########
def plot_bbh_vs_smbh_mass(run_directory, fname_out, title=None):
    '''Plot smbh mass'''
    # Load mcfacts data
    MH = McfactsHandler.from_runs(run_directory)
    # Initialize style
    plt.style.use('bmh')
    # Initialize plot
    fig, axes = plt.subplots(nrows=2,sharex='col')
    # Plot things
    axes[0].scatter(np.log10(MH.mass_smbh.value), np.log10(MH.avg_systems))
    axes[1].scatter(np.log10(MH.mass_smbh.value), MH.avg_systems)
    axes[1].set_xlabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{SMBH}}) [\mathrm{M}_{\odot}]$")
    axes[0].set_ylabel(r"$\mathrm{log}_{10}(\mu_{\mathrm{BBH}})$")
    axes[1].set_ylabel(r"$\mu_{\mathrm{BBH}}$")
    axes[0].set_ylim(np.log10(1./min(MH.n_iterations)), max(np.log10(MH.avg_systems)) + 1)
    axes[1].set_ylim(0, axes[1].get_ylim()[1])
    # Check title
    if not (title is None):
        fig.suptitle(title)
    # savefig
    plt.savefig(fname_out)
    # Close plt
    plt.close()
    
def plot_nsc_mass_vs_smbh_mass(run_directory, fname_out, title=None):
    '''Plot smbh mass'''
    # Load mcfacts data
    MH = McfactsHandler.from_runs(run_directory)
    # Initialize style
    plt.style.use('bmh')
    # Initialize plot
    fig, ax = plt.subplots()
    # Plot things
    ax.scatter(np.log10(MH.mass_smbh.value), np.log10(MH.batch_param('M_nsc').value))
    ax.set_xlabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{SMBH}}) [\mathrm{M}_{\odot}]$")
    ax.set_ylabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{NSC}}) [\mathrm{M}_{\odot}]$")
    # Check title
    if not (title is None):
        fig.suptitle(title)
    # savefig
    plt.savefig(fname_out)
    # Close plt
    plt.close()
    
######## Arguments for default ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True, type=str,
        help="Runs for McFacts")
    opts = parser.parse_args()
    return opts

######## main ########
def main():
    opts = arg()
    plot_bbh_vs_smbh_mass(
                          opts.run_directory,
                          join(opts.run_directory, "bbh_vs_smbh_mass.png"),
                          title=basename(opts.run_directory),
                         )
    plot_nsc_mass_vs_smbh_mass(
                               opts.run_directory,
                               join(opts.run_directory, "nsc_mass_vs_smbh_mass.png"),
                               title=basename(opts.run_directory),
                              )
    return

######## Execution ########
if __name__ == "__main__":
    main()
