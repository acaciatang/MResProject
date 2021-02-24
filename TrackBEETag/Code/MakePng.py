#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'MakePng.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import cv2
import os
import scipy.io
import re
import pandas
import math

def switch(frame, background):
    
    return 0
    
def window(frame, background, kernel):
    
    return 0

def makepng(filename):
    cap = cv2.VideoCapture(filename)
    
    outname = os.path.splitext(os.path.basename(filename))[0]
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(outname + "_" + str(i) + ".png", frame)
        
        i+=1  
    
    cap.release()
    cv2.destroyAllWindows()
    
    return 0

def main(argv):
    """ Main entry point of the program """
    iter = os.getenv('PBS_ARRAY_INDEX')
    files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
    filename = files[int(iter)-1]

    makepng(filename)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)