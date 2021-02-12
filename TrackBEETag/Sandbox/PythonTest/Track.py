#!/usr/bin/env python3

"""Tracks BEETags in video."""

__appname__ = 'Track.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import matlab.engine
import subprocess
import cv2
import os
import scipy.io
import re
import pandas
import math

def wrangle(frameNum): # takes in .mat with tracking data and outputs long-form data with the following fields: frame, ID, centroidX, centroidY, dir
    matOutput = "output" + str(frameNum) + ".mat"
    rawdata = scipy.io.loadmat(matOutput)['output']
    frame = pandas.DataFrame()
    for j in range(len(rawdata)):
        ID = rawdata[j][0][3][0][0]
        centroidX = rawdata[j][0][1][0][0] #topleft is 0, 0
        centroidY = rawdata[j][0][1][0][1]
        frontX = rawdata[j][0][4][0][0]
        frontY = rawdata[j][0][5][0][0]
        dir = math.degrees(math.atan2(frontX-centroidX, centroidY-frontY)) #pointing up = 0
        OneCM = math.sqrt((centroidX - frontX)**2 + (centroidY - frontY)**2)/0.15
        if dir < 0:
            dir = 360+dir
    
        row = [(frameNum, ID, centroidX, centroidY, dir, OneCM)]
        frame = frame.append(row, ignore_index=True)
    return(frame) 

def trackframe(filename, frameNum, method): #frameNum starts at 0
    p = subprocess.Popen(["ffmpeg", "-i", filename, "-vf", "select=eq(n\,"+ str(frameNum) + ")", "-vframes", "1", "out.png"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #stdout, stderr = p.communicate()
    #if len(stderr) > 0 :
    #    print("Oh dear, something went wrong! \nSee below:")
    #    print(stderr)
    
    #arguments = ['threshMode', '1', 'bradleyFilterSize', ['15', '15'], 'bradleyThreshold', '3']
    eng = matlab.engine.start_matlab()
    #eng.addpath('/rds/general/user/tst116/home/TrackBEETag/Code')
    eng.addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Sandbox/PythonTest/MatlabKeep')
    #im = eng.imread('out.png')
    
    if method == 16:
        matOutput = eng.locate16BitCodes_hard(frameNum)
    else:
        matOutput = eng.locateCodes_hard(frameNum)

    p2 = subprocess.Popen(["rm", "out.png"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    eng.quit()

    return wrangle(frameNum)

def track(filename, method):
    #filename = "/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Videos/BEEs.mp4"
    cap = cv2.VideoCapture(filename)
    nframes = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT) ))
    #nchunks = nframes//1000
    #chunks = [1 + 1000*i for i in range(nchunks)]
    
    outname = os.path.splitext(os.path.basename(filename))[0] + '.csv'

    output = pandas.DataFrame()
    for i in range(nframes):
        frameData = trackframe(filename, i, method)
        pandas.concat([output, frameData], ignore_index=True)
    
    output.to_csv(path_or_buf = outname, na_rep = "NA", index = False)

    return 0

def main(argv):
    """ Main entry point of the program """
    track(argv[1], argv[2])
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)