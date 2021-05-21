#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'RemoveBackground.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import av
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from matching.games import HospitalResident
import pandas as pd
import copy
import PythonTracking
import subprocess

def main(argv):
    for (dir, subdir, files) in subprocess.os.walk("../Data/Test"):
        for file in files:
            print(file)
            if file != '.DS_Store':
                subprocess.Popen(["mkdir", '../Results/Test/' + os.path.splitext(file)[0]]) #do this first
                #print('Tracking ' + file)
                #img = cv2.imread(dir + '/' + file)
                #outname = '../Results/Test/' + os.path.splitext(file)[0]
                #np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]])
                print('Finished ' + file)

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)