#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'TestBackgroundCancellation.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import cv2
import os
import numpy as np
import pandas as pd

#code
def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]

    bkgd = getbkgd(filename)
    print("Made background!")
    cap = cv2.VideoCapture(filename)
    outname = os.path.splitext(os.path.basename(filename))[0]
    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(outname + '_edited.avi',fourcc, 20.0, (640,480))
    
    i=0
    edited = list()
    while(cap.isOpened()):
        ret , frame = cap.read()
        if ret == False:
            break
        rmbkgd = rmbkgd_pixel(bkgd, frame, outname, i)
        edited.append(rmbkgd)
        #if i<100:
        cv2.imwrite(outname + "_" + str(i) + "_edited.png", rmbkgd)
        out.write(rmbkgd)
        print("Edited frame" + str(i) + "!")
        i+=1
        #else:
        #    break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)