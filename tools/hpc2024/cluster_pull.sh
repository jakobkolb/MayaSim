# download output files from cluster to local:
rsync -t -r --stats --progress  "fritzku@cluster.pik-potsdam.de:/p/tmp/fritzku/MayaSim/output/X1_aggregate_dynamics/*" "./output/X1_aggregate_dynamics/"
