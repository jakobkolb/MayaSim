#!/usr/bin/env bash
#SBATCH --job-name=X1_subm
#SBATCH --output=X1_submit_%j.out
#SBATCH --error=X1_submit_%j.err
#SBATCH --chdir=/p/tmp/fritzku/MayaSim

jid1=$(sbatch ~/MayaSim/tools/hpc2024/job_X1_0_compute.sh)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } ~/MayaSim/tools/hpc2024/job_X1_1_mean.sh

sleep 2
sbatch --dependency=afterok:${jid1##* } ~/MayaSim/tools/hpc2024/job_X1_2_collect.sh