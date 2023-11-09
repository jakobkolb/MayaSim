#!/bin/bash

f2py    --f90flags='-fopenmp -O3 -ftree-vectorize' -lgomp -c  f90routines.f90 -m f90routines
f2py3   --f90flags='-fopenmp -O3 -ftree-vectorize' -lgomp -c  f90routines.f90 -m f90routines
f2py3.5 --f90flags='-fopenmp -O3 -ftree-vectorize' -lgomp -c  f90routines.f90 -m f90routines


#f2py    --f90flags='-g -fbounds-check' -lgomp -c  f90routines.f90 -m f90routines
#f2py3   --f90flags='-g -fbounds-check' -lgomp -c  f90routines.f90 -m f90routines
#f2py3.5 --f90flags='-g -fbounds-check' -lgomp -c  f90routines.f90 -m f90routines

