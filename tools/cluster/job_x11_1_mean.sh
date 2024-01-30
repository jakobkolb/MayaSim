#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=x11_mean
#SBATCH --account=copan
#SBATCH --output=x11_mean_%j.out
#SBATCH --error=x11_mean_%j.err
#SBATCH --workdir=/p/tmp/fritzku/MayaSim
#SBATCH --nodes=4
#SBATCH --tasks-per-node=15
#SBATCH --time=02:00:00

module load intel/2017.1
module load anaconda/2023.09
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export OMP_NUM_THREADS=1

source activate mayasim

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ~/MayaSim/experiments/
srun -n $SLURM_NTASKS python x11_dynamical_regimes.py --mode 1
