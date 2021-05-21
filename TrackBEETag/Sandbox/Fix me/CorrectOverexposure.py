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
import matplotlib.colors as mcolors
import math

#code
def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]
    
    outname = os.path.splitext(os.path.basename(filename))[0]
    
    cap = cv2.VideoCapture(filename)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outname + "_tracks.mp4", fourcc, 20.0, (3840,2160))

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
                break
        cv2.imwrite('tags.png',frame)
        break
        G = frame[:, :, 1]
        G = cv2.cvtColor(G,cv2.COLOR_GRAY2RGB)

        # write the flipped frame
        out.write(G)
        print("Wrote frame " + str(i))
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i+=1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)