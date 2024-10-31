#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
import pandas as pd
import glob as g
import os
from scipy.optimize import curve_fit
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data
from mcfacts.vis import plotting
from mcfacts.vis import styles

# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

figsize = "apj_col"

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-directory",
                        default="runs",
                        type=str, help="folder with files for each run")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isdir(opts.runs_directory)
    return opts


def main():
    opts = arg()

    folders = (g.glob(opts.runs_directory + "gal*"))

    data = np.loadtxt(folders[0] + "/initial_params_star.dat",skiprows=2)


    # ========================================
    # Stars initial mass
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))
    bins = np.linspace(np.log10(data[:,2]).min(), np.log10(data[:,2]).max(),20)

    plt.hist(np.log10(data[:,2]), bins=bins)
    plt.xlabel('log initial mass [$M_\odot$]')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.savefig(opts.plots_directory + r"/stars_initial_mass.png",format="png")



######## Execution ########
if __name__ == "__main__":
    main()