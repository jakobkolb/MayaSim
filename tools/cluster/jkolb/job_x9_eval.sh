#!/bin/bash
#SBATCH --qos=priority
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=Maya_9_eval
#SBATCH --output=ms_x9_eval_%j.out
#SBATCH --error=ms_x9_eval_%j.err
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16

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
srun -n $SLURM_NTASKS python mayasim_X9_stability_analysis.py 0 2
