#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'TestBackgroundCancellation.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import cv2
import os
import math
import numpy as np

def makepng(filename):
    cap = cv2.VideoCapture(filename)
    
    outname = os.path.splitext(os.path.basename(filename))[0]
    global i
    i=0
    while(cap.isOpened()):
        ret , frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(outname + "_" + str(i) + ".png", frame)
        i+=1  
    
    cap.release()
    cv2.destroyAllWindows()
    return 0

def getbkgd(filename):
    cap = cv2.VideoCapture(filename)
    global i
    i = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    global outname
    outname = os.path.splitext(os.path.basename(filename))[0]
    frameNum = list(range(0, int(i), math.floor(i/20)))

    referenceFrames = [cap.read() for i in frameNum]
    combined = np.stack(referenceFrames)
    bkgd = np.median(combined, axis=0)
    cv2.imwrite(outname + "_bkgd.png", bkgd)
    return bkgd

def rmbkgd_pixel(bkgd, frame, outname, frameNum):
    #frame = cv2.imread(outname + "_" + str(frameNum) + ".png", -1)
    thres = 10
    minimum = bkgd - thres
    maximum = bkgd + thres

    def pixel(a, b, c):
        if b < a < c:
            return 255
        else:
            return a

    vpixel = np.vectorize(pixel)
    rmbkgd = vpixel(frame, minimum, maximum)
    #rmbkgd = rmbkgd[:, :, 2]
    cv2.imwrite(outname + "_" + str(frameNum) + "_edited.png", rmbkgd)
    return rmbkgd

def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]

    bkgd = getbkgd(filename)

    cap = cv2.VideoCapture(filename)
    outname = os.path.splitext(os.path.basename(filename))[0]
    i=0
    while(cap.isOpened()):
        ret , frame = cap.read()
        if ret == False:
            break
        rmbkgd_pixel(bkgd, frame, outname, i)
        #cv2.imwrite(outname + "_" + str(i) + ".png", frame)
        i+=1  
    
    cap.release()
    cv2.destroyAllWindows()

    return i

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)