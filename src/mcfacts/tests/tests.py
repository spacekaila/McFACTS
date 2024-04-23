import numpy as np


def test_merger():
    #This is a test of the merger 
    #1. Test a merger by calling modules
    print("Test Module merger!")
    mass_1 = 10.0
    mass_2 = 15.0
    spin_1 = 0.1
    spin_2 = 0.7
    angle_1 = 1.80
    angle2 = 0.7
    bin_ang_mom = 1.0
    outmass = tichy08.merged_mass(mass_1, mass_2, spin_1, spin_2)
    outspin = tichy08.merged_spin(mass_1, mass_2, spin_1, spin_2, bin_ang_mom)
    out_chi = chieff.chi_effective(mass_1, mass_2, spin_1, spin_2, angle_1, angle2, bin_ang_mom)
    print(outmass,outspin,out_chi)
    #Output should always be constant: 23.560384 0.8402299374639024 0.31214563487176167
    return