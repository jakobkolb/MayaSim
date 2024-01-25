#!/usr/bin/env bash

jid1=$(sbatch ~/MayaSim/tools/cluster/job_x11_0_compute.sh)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } ~/MayaSim/tools/cluster/job_x11_1_mean.sh

sleep 2
sbatch --dependency=afterok:${jid1##* } ~/MayaSim/tools/cluster/job_x11_2_collect.sh