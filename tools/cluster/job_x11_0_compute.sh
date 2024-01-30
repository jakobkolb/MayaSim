#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=x11_comp
#SBATCH --account=copan
#SBATCH --output=x11_comp_%A_%a.out
#SBATCH --error=x11_comp_%A_%a.err
#SBATCH --workdir=/p/tmp/fritzku/MayaSim
#SBATCH --nodes=4
#SBATCH --tasks-per-node=16  # 64 tasks per array job,
#SBATCH --array=0-324  # thus 4 param combs * 30 samples per job, 2 param combs per task.
#SBATCH --time=05:30:00

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
srun -n $SLURM_NTASKS python x11_dynamical_regimes.py --mode 0 --job_id $SLURM_ARRAY_TASK_ID --max_id $SLURM_ARRAY_TASK_MAX
