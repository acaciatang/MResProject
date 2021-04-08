#!/bin/bash
#PBS -lwalltime=48:00:00
#PBS -lselect=1:ncpus=8:ompthreads=8:mem=24gb
date
echo "loading modules"
module load matlab/R2020b # allows matlab to be run with HPC
module load anaconda3/personal
source activate trackenv

which python

echo "running code"
date
python $HOME/TrackBEETag/Code/pyAV_test.py
matlab < $HOME/TrackBEETag/Code/locate16BitCodes_hpc.m
python $HOME/TrackBEETag/Code/Wrangle.py
date

echo "Done! Moving files"
mv *.mat $HOME/TrackBEETag/AntibioticsResults # move files to folder
mv *.csv $HOME/TrackBEETag/AntibioticsResults
echo "Finished!"
date

#done
