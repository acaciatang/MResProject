#!/usr/bin/env python3

"""Tests algorithm for tag reading based on training data."""

__appname__ = 'Training.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import cv2
import subprocess
import os
import PythonTracking

#code

def main(argv):
    for (dir, subdir, files) in subprocess.os.walk("../Data/Training"):
        for file in files:
            if file != '.DS_Store':
                print('Tracking ' + file)
                img = cv2.imread(dir + '/' + file)
                outname = '../Results/Training/' + os.path.splitext(file)[0]
                PythonTracking.findtags(img, outname)
                print('Finished ' + file)



if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)