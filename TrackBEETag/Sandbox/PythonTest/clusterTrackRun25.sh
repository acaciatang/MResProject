#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=20gb
echo "loading modules"
module load matlab/R2020b # allows matlab to be run with HPC
module load anaconda3/personal
activate trackenv

echo "running code"
python3 $HOME/TrackBEETag/Code/PythonTest/clusterTrack25.py

echo "Done! Moving files"
mv *.tar $HOME/TrackBEETag/Results # move files to folder
mv *.csv $HOME/TrackBEETag/Results
echo "Finished!"

#done
