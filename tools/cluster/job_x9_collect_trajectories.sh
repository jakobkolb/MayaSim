#!/bin/bash
#SBATCH --qos=priority
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=m9res
#SBATCH --output=m9res_%j.out
#SBATCH --error=m9res_%j.err
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem 7500

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

srun -n $SLURM_NTASKS python res_x9.py -i ../output_data/X9_stability_analysis/results/all_trajectories -o ../output_data/X9_stability_analysis/results/all_trajectories.df5

