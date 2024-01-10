#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=Maya_1
#SBATCH --output=ms_x1_%j.out
#SBATCH --error=ms_x1_%j.err
#SBATCH --account=copan
#SBATCH --nodes=2
#SBATCH --tasks-per-node=4

module load compiler/intel/16.0.0
module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export OMP_NUM_THREADS=1

source activate py36

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ../Experiments/
srun -n $SLURM_NTASKS python mayasim_X1_default_setup.py 0

