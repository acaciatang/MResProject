#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
echo "starting to run code"
module load matlab/R2020b # allows R to be run with HPC
module load ffmpeg
echo "opened matlab"
matlab < $HOME/TrackBEETag/Code/Tracking_cluster.m # run simulation
echo "ran simulation"
mv video* $HOME/TrackBEETag/Results # move files to folder
echo "matlab has finished running"