#!/bin/bash
#SBATCH --qos=short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=Maya_7
#SBATCH --output=ms_x7_%j.out
#SBATCH --error=ms_x7_%j.err
#SBATCH --account=copan
#SBATCH --nodes=8
#SBATCH --tasks-per-node=16

module load compiler/intel/16.0.0
module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export OMP_NUM_THREADS=1

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ../Experiments/
srun -n $SLURM_NTASKS python mayasim_X7_multiple_droughts.py 0
