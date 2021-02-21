#import packages
import Wrangle16
import os

iter = os.getenv('PBS_ARRAY_INDEX')
files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
filename = files[int(iter)-1]

Wrangle16.combine(filename)

