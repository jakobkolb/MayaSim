#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=x11_test
#SBATCH --account=copan
#SBATCH --output=x11_test_%A_%a.out
#SBATCH --error=x11_test_%A_%a.err
#SBATCH --chdir=/p/tmp/fritzku/MayaSim
#SBATCH --ntasks=8
#SBATCH --array=0-4
#SBATCH --time=01:00:00

module load intel/oneAPI/2024.0.0
module load anaconda/2023.09
export OMP_NUM_THREADS=1
export DISABLE_TQDM=True # disable progress bar output to stderr

source activate mayasim

# check if correct python is found
python -m site
which python

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ~/MayaSim/experiments/
mpirun -n $SLURM_NTASKS python x11_dynamical_regimes.py --testing --mode 0 --job_id $SLURM_ARRAY_TASK_ID --max_id $SLURM_ARRAY_TASK_MAX