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

def getbkgd(outname):
    frameNum = list(range(int(i), 0, math.floor(-i/20)))
    referenceNames = [outname + "_" + str(i) + ".png" for i in frameNum]
    referenceFrames = [cv2.imread(i, -1) for i in referenceNames]
    combined = np.stack(referenceFrames)
    bkgd = np.median(combined, axis=0)
    #cv2.imwrite(outname + "_bkgd.png", bkgd)
    return bkgd

def rmbkgd_pixel(bkgd, outname, frameNum):
    frame = cv2.imread(outname + "_" + str(frameNum) + ".png", -1)
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