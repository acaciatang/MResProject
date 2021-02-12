#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
echo "loading modules"
module load matlab/R2020b # allows matlab to be run with HPC
module load ffmpeg/3.2.4 # allows videos to be read

#module load anaconda3/personal
#anaconda-setup
module load apps/python/conda
conda create --quiet --yes --name trackEnv python=3.8.5
source activate trackEnv

pushd /usr/local/packages/apps/matlab/2020b/binary/extern/engines/python
python setup.py build -b $TMPDIR install
popd


conda activate trackEnv
python clusterTrack.py



echo "running code"
matlab < $HOME/TrackBEETag/Code/Tracking_cluster.m # run simulation
echo "Done! Moving files"

tar czvf Simulation.tgz *.mat
mv video* $HOME/TrackBEETag/Results # move files to folder
echo "Finished!"

#done