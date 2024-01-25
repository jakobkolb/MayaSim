#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=Maya_x11
#SBATCH --account=copan
#SBATCH --output=ms_x11_%j.out
#SBATCH --error=ms_x11_%j.err
#SBATCH --workdir=/p/tmp/fritzku/MayaSim
#SBATCH --nodes=4
#SBATCH --tasks-per-node=16

module load intel/2017.1
module load anaconda/2023.09
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export OMP_NUM_THREADS=1
export DISABLE_TQDM=True # disable progress bar output to stderr

source activate mayasim

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ~/MayaSim/experiments/
srun -n $SLURM_NTASKS python x11_dynamical_regimes.py --mode=1
