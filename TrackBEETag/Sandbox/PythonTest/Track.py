#!/usr/bin/env python3

"""Tracks BEETags in video."""

__appname__ = 'Track.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
#import matlab.engine
import subprocess
import cv2
import os
import scipy.io
import re
import pandas
import math

def wrangle(frameNum): # takes in .mat with tracking data and outputs long-form data with the following fields: frame, ID, centroidX, centroidY, dir
    matOutput = "output.mat"
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

def trackframe(frameNum, method): #frameNum starts at 0
    #eng = matlab.engine.start_matlab()
    #eng.addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Sandbox/PythonTest/MatlabKeep')
    
    if method == 16:
        #matOutput = eng.locate16BitCodes_hard(frameNum)
        b1 = subprocess.Popen(["matlab", "<", "$HOME/TrackBEETag/Code/PythonTest/MatlabKeep/locate16BitCodes_hard.m"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wrangled = wrangle(frameNum)
        b2 = subprocess.Popen(["mv", "output.mat", "output" + str(frameNum) + ".mat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        #matOutput = eng.locateCodes_hard(frameNum)
        g1 = subprocess.Popen(["ls"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        b1 = subprocess.Popen(["matlab", "<", "$HOME/TrackBEETag/Code/PythonTest/MatlabKeep/locateCodes_hard.m"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        g2 = subprocess.Popen(["ls"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wrangled = wrangle(frameNum)
        b2 = subprocess.Popen(["mv", "output.mat", "output" + str(frameNum) + ".mat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
    #eng.quit()
    return wrangled

def track(filename, method):
    #filename = "/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Videos/video1.mp4"
    cap = cv2.VideoCapture(filename)
    
    outname = os.path.splitext(os.path.basename(filename))[0] + '.csv'
    output = pandas.DataFrame()
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite("out.png",frame)
        
        frameData = trackframe(i, method)  
        output = pandas.concat([output, frameData], ignore_index=True)
        i+=1  
    
    cap.release()
    cv2.destroyAllWindows()
    
    output = output.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm'})
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