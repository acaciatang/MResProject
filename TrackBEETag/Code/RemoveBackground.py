#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'RemoveBackground.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import cv2
import os
import math
import numpy as np
import statistics

def getbkgd(filename):
    referenceFrames = list()
    cap = cv2.VideoCapture(filename)
    outname = os.path.splitext(os.path.basename(filename))[0]
    
    if not cap.isOpened():
        print("Hold on the file didn't open let's try opening it")
        cap.open()

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print("The end.")
            break
        if i % 100 == 0:
            referenceFrames.append(frame)
        i+=1  
    
    cap.release()
    cv2.destroyAllWindows()
    
    combined = np.stack(referenceFrames)
    
    def mode(a):
        modes = statistics.multimode(a)
        return min(modes)

    a = combined.shape[1]
    b = combined.shape[2]
    bkgd = np.empty((a, b), dtype='uint8')
    for i in range(a):
        for j in range(b):
            bkgd[i, j] = mode(combined[:, i, j, 2])

    cv2.imwrite(outname + "_bkgd.png", bkgd)
    return bkgd

def rmbkgd_pixel(bkgd, frame, outname, frameNum):
    #frame = cv2.imread(outname + "_" + str(frameNum) + ".png", -1)
    #bkgd = cv2.imread(outname + "_bkgd.png", -1)
    frame = frame [:, :, 2]
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
    #cv2.imwrite(outname + "_" + str(frameNum) + ".png", rmbkgd)
    return rmbkgd

def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]

    print (filename)
    bkgd = getbkgd(filename)
    print("Made background!")
    
    cap = cv2.VideoCapture(filename)
    outname = os.path.splitext(os.path.basename(filename))[0]
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #out = cv2.VideoWriter(outname + '_edited.avi',fourcc, 20.0, (640,480))
    
    if not cap.isOpened():
        print("Hold on the file didn't open let's try opening it")
        cap.open()


    i=0
    #edited = list()
    while(cap.isOpened()):
        ret , frame = cap.read()
        if ret == False:
            print("The end.")
            break
        rmbkgd = rmbkgd_pixel(bkgd, frame, outname, i)
        #edited.append(rmbkgd)
        #if i<100:
        cv2.imwrite(outname + "_" + str(i) + ".png", rmbkgd)
        #out.write(rmbkgd)
        print("Edited frame" + str(i) + "!")
        i+=1
        #else:
        #    break
    
    cap.release()
    #out.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)