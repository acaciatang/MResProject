#import packages
import Track
import os

iter = os.getenv('PBS_ARRAY_INDEX')
files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
filename = files[iter-1]

Track.track(filename, 25)
