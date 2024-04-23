# Declarations
.PHONY: all clean

all: clean tests
tests: mcfacts_sim

######## Definitions ########
#### Package ####
# Version number for the repository
# This is where you change the version number by hand. Not anywhere else.
# Alpha begins at 0.1.0
# Feature-complete Alpha begins at 0.2.0
# Beta begins at 0.3.0
VERSION=0.0.0

### Should work for everyone ###
# Current directory
HERE=$(shell pwd)

#### Scripts ####
MCFACTS_SIM_EXE = ${HERE}/mcfacts_sim.py

######## Instructions ########
#### Install ####

version: clean
	echo "__version__ = ${VERSION}" > __version__.py
	echo "__version__ = ${VERSION}" > src/mcfacts/__version__.py

install: clean version
	pip3 install -e .

#### Test one thing at a time ####

mcfacts_sim: clean
	python3 ${MCFACTS_SIM_EXE}

#### CLEAN ####
clean:
	rm -rf run*
	rm -rf output_mergers_population.dat


