#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
echo "loading modules"
module load matlab/R2020b # allows matlab to be run with HPC
module load ffmpeg/3.2.4 

echo "running code"
matlab < $HOME/TrackBEETag/Code/Tracking_cluster.m # run simulation
echo "Done! Moving files"
mv video* $HOME/TrackBEETag/Results # move files to folder
echo "Finished!"

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
#qsub -J 1-3 ../Code/clusterRunBEE.sh

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