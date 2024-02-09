# download output files from cluster to local:
rsync -t -r --stats --progress  "fritzku@cluster.pik-potsdam.de:/p/tmp/fritzku/MayaSim/output/x11_dynamical_regimes/*" ./output/x11_dynamical_regimes/
