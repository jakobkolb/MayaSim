#!/bin/bash
# 
# execute this file from command line to compile f90routines.f90
# and create extension module via numpy.f2py.
# `--f90flags` contains flags for the gfortran compiler

f2py --f90flags='-O3 -march=native' -lgomp -c f90routines.f90 -m f90routines
