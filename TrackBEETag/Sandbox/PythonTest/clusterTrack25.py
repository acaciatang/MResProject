#import packages
import Track
import os
import subprocess

iter = os.getenv('PBS_ARRAY_INDEX')
files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
filename = files[iter-1]
outname = os.path.splitext(os.path.basename(filename))[0]

Track.track(filename, 25)
subprocess.Popen(["tar", "czvf", outname + ".tgz", "*.mat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)