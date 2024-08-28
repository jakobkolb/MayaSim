#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=x11_comp
#SBATCH --account=copan
#SBATCH --output=x11_comp_%A_%a.out
#SBATCH --error=x11_comp_%A_%a.err
#SBATCH --chdir=/p/tmp/fritzku/MayaSim
#SBATCH --ntasks=64  # 64 tasks per array job,
#SBATCH --array=0-162  # thus 8 param combs * 32 samples per job, 4 param combs per task.
#SBATCH --time=07:00:00  # time limit per array job

module load intel/oneAPI/2024.0.0
module load anaconda/2023.09
export OMP_NUM_THREADS=1
export DISABLE_TQDM=True # disable progress bar output to stderr

source activate mayasim

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ~/MayaSim/experiments/
mpirun -n $SLURM_NTASKS python x11_dynamical_regimes.py --mode 0 --job_id $SLURM_ARRAY_TASK_ID --max_id $SLURM_ARRAY_TASK_MAX
