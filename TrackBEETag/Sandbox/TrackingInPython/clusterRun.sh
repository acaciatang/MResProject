#!/bin/bash
#PBS -lwalltime=48:00:00
#PBS -lselect=1:ncpus=8:ompthreads=8:mem=24gb
date
echo "loading modules"
module load anaconda3/personal
source activate trackenv

which python

date
echo "Reading Tags"
python $HOME/PythonTrack/Code/PythonTracking.py

date
echo "Wrangling"
python $HOME/PythonTrack/Code/Wrangling2.py

date
echo "Drawing"
python $HOME/PythonTrack/Code/DrawTracks.py

date
echo "Done! Moving files"
mv *.csv $HOME/PythonTrack/Replicate1/Results
mv *.mp4 $HOME/PythonTrack/Replicate1/Results
echo "Finished!"
date

#done


#put files
#sftp tst116@login.cx1.hpc.ic.ac.uk
#[enter imperial password]
#put tst116_HPC_2020_main.R
#put tst116_HPC_2020_cluster.R
#put clusterRun.sh
#exit

#set up
#Ssh -l tst116 login.cx1.hpc.ic.ac.uk
#[enter imperial password]
#mkdir tst116_HPC_2020
#cd tst116_HPC_2020
#cat tst116_HPC_2020_cluster.R
#module load anaconda3/personal
#anaconda-setup
#conda install r

#run in Results
#qsub -J 1-20 clusterTrackPyAV.sh

#stop 

#check
#qstat

#get files
#mv clusterRun.sh.e* error
#mv clusterRun.sh.o* output
#cd tst116_HPC_2020
#tar czvf Simulation.tgz *
#cd ../error
#tar czvf Error.tgz *
#cd ../output
#tar czvf Output.tgz *

#sftp tst116@login.cx1.hpc.ic.ac.uk
#[enter imperial password]
#get tst116_HPC_2020/Simulation.tgz
#get error/Error.tgz
#get output/Output.tgz
#exit