# Declarations
.PHONY: all clean

all: clean tests
tests: test2

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
TEST2_EXE = ${HERE}/test2.py

######## Instructions ########
#### Install ####

version: clean
	echo "__version__ = ${VERSION}" > __version__.py
	echo "__version__ = ${VERSION}" > src/mcfacts/__version__.py

install: clean version
	pip3 install -e .

#### Test one thing at a time ####

test2: clear
	python3 ${TEST2_EXE}

#### CLEAN ####
#clean:
#	rm -rf src/*.egg*

clear:
	rm -rf run*
	rm -rf output_mergers_population.dat


