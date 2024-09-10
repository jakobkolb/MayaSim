#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=X1_coll
#SBATCH --account=copan
#SBATCH --output=X1_coll_%j.out
#SBATCH --error=X1_coll_%j.err
#SBATCH --chdir=/p/tmp/fritzku/MayaSim
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

module load intel/oneAPI/2024.0.0
module load anaconda/2023.09
export OMP_NUM_THREADS=1

source activate mayasim

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ~/MayaSim/experiments/
mpirun -n $SLURM_NTASKS python X1_aggregate_dynamics.py --mode 2
