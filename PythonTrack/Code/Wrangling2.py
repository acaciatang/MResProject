#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'RemoveBackground.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
from io import RawIOBase
import sys 
import cv2
import os
import numpy as np
import pandas as pd
import math
import copy
import collections

def byID(outname):
    raw = pd.read_csv('../Results/' + outname + '_raw.csv')

def relabel(outname, ID):
    raw = pd.read_csv('../Results/' + outname + '_raw.csv')

    return 0

def addmissing(ID):

def main(argv):
    if len(sys.argv) == 2:
        #if argv[1] == '.':
        #    files = [f for f in os.listdir('.') if f[-4:-1] == '.MP']
        #    for filename in files:
        #        wrangle(filename)
        #    return 0
        #else:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]
    filename = '../Data/R1D7R2A1_trimmed.MP4'
    outname = os.path.splitext(os.path.basename(filename))[0]
    return wrangle(outname)

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)