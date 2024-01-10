#!/usr/bin/env bash

jid1=$(sbatch job_x10_compute.sh)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } job_x10_pp.sh

sleep 2
sbatch --dependency=afterok:${jid1##* } job_x10_map_frames.sh
