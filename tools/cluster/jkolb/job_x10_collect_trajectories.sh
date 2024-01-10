#!/bin/bash
#SBATCH --qos=priority
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=m10col
#SBATCH --output=m10col_%j.out
#SBATCH --error=m10col_%j.err
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

module load compiler/intel/16.0.0
module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export OMP_NUM_THREADS=1

source activate mayasim

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ../Experiments/
srun -n $SLURM_NTASKS python mayasim_X10_generate_trajectories.py --mode 3
