#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os.path import abspath, isfile, isdir, split, join
import sys
# Grab those txt files
from importlib import resources as impresources

## Local imports ##
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
from mcfacts.vis import data
from mcfacts.outputs import mergerfile

######### Setup ########
COLUMN_NAMES=mergerfile.names_rec

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname-mergers",
        default="output_mergers_population.dat",
        type=str, help="output_mergers file")
    parser.add_argument("--cdf",
        nargs="+",default=[],help="Fields for cdf plots")
    parser.add_argument("--verbose", action='store_true')
    opts = parser.parse_args()
    assert isfile(opts.fname_mergers)
    opts.wkdir = split(abspath(opts.fname_mergers))[0]
    assert isdir(opts.wkdir)
    return opts

######## Load data ########
def load_mergers_txt(fname, verbose=False):
    # Load the text file
    mergers = np.loadtxt(fname, skiprows=2)
    # Initialize dictionary
    merger_dict = {}
    # Check for iter
    if len(COLUMN_NAMES) < mergers.shape[1]:
        assert len(COLUMN_NAMES) + 1== mergers.shape[1]
        merger_dict['iter'] = mergers[:,0]
    # Loop the column names
    for i, item in enumerate(COLUMN_NAMES):
        # Find the correct index
        if 'iter' in merger_dict:
            index = i + 1
        else:
            index = i
        # update the merger dict
        merger_dict[item] = mergers[:,index]

    ## Select finite values
    mask = np.isfinite(merger_dict['chi_eff'])
    print("Removing %d nans out of %d sample mergers"%(np.sum(~mask), mask.size), file=sys.stderr)
    for item in merger_dict:
        merger_dict[item] = merger_dict[item][mask]

    ## Print things
    # Loop the merger dict
    if verbose:
        for item in merger_dict:
            #print(item, merger_dict[item].dtype, merger_dict[item].shape, merger_dict[item][1])
            print(item, merger_dict[item].dtype, merger_dict[item].shape)
            print(merger_dict[item])
    # Return the merger dict
    return merger_dict

######## Functions ########
def simple_cdf(x):
    _x = np.sort(x)
    cdf = np.arange(_x.size)
    cdf = cdf / np.max(cdf)
    return _x, cdf

######## Plots ########
def plot_cdf(merger_dict, label, fname):
    x, y = simple_cdf(merger_dict[label])
    from matplotlib import pyplot as plt
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.savefig(fname)
    plt.close()


######## Main ########
def main():
    opts = arg()
    merger_dict = load_mergers_txt(opts.fname_mergers, verbose=opts.verbose)

    #### Cdf plots ####
    for _item in opts.cdf:
        assert _item in merger_dict
        fname_item = join(opts.wkdir, "mergers_cdf_%s.png"%(_item))
        plot_cdf(merger_dict, _item, fname_item)
    return
######## Execution ########
if __name__ == "__main__":
    main()
