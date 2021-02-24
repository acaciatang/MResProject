#!/bin/bash
#PBS -lwalltime=48:00:00
#PBS -lselect=1:ncpus=8:mem=5gb
echo "loading modules"
module load matlab/R2020b # allows matlab to be run with HPC
module load anaconda3/personal
activate trackenv

echo "running code"
python $HOME/TrackBEETag/Code/MakePng.py
matlab < $HOME/TrackBEETag/Code/locate16BitCodes_hpc.m
python $HOME/TrackBEETag/Code/Wrangle.py

echo "Done! Moving files"
mv *.mat $HOME/TrackBEETag/Results # move files to folder
mv *.csv $HOME/TrackBEETag/Results
echo "Finished!"

#done