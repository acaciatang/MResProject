#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
echo "loading modules"
module load matlab/R2020b # allows matlab to be run with HPC
module load ffmpeg # allows videos to be read

echo "running code"
matlab < $HOME/TrackBEETag/Code/Tracking_cluster.m # run simulation
echo "Done! Moving files"
mv video* $HOME/TrackBEETag/Results # move files to folder
echo "Finished!"

#done