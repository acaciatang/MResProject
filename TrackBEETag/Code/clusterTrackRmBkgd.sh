#!/bin/bash
#PBS -lwalltime=48:00:00
#PBS -lselect=1:ncpus=8:mem=96gb
date
echo "loading modules"
module load matlab/R2020b # allows matlab to be run with HPC
module load anaconda3/personal
source activate trackenv

date
echo "running code"
#python $HOME/TrackBEETag/Code/MakePng.py
python $HOME/TrackBEETag/Code/RemoveBackground.py
matlab < $HOME/TrackBEETag/Code/locate16BitCodes_hpc.m
python $HOME/TrackBEETag/Code/Wrangle.py

date
echo "Done! Moving files"
mv *.mat $HOME/TrackBEETag/Results # move files to folder
mv *.csv $HOME/TrackBEETag/Results
echo "Finished!"
date
#done
