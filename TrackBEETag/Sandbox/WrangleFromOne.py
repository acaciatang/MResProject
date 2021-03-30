 #!/usr/bin/env python3

"""Processes .mat files."""

__appname__ = 'Wrangle.py'
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
from pathlib import Path

def wrangle(filename, frameNum): # takes in .mat with tracking data and outputs long-form data with the following fields: frame, ID, centroidX, centroidY, dir
    matOutput = filename + "_" + str(frameNum) + ".mat"
    checkforfile = Path(matOutput)
    if checkforfile.is_file():
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


def wrangle(matfile): # takes in .mat with tracking data and outputs long-form data with the following fields: frame, ID, centroidX, centroidY, dir
    rawdata = scipy.io.loadmat(matfile)['trackingData'][0]
    iter = int(re.findall(r"\d+", matfile)[-1]) -1
    output = pandas.DataFrame()
    for i in range(len(rawdata)):
        for j in range(len(rawdata[i][0])):
            frame = i + 1 + iter*1000
            ID = rawdata[i][0][j][0][3][0][0]
            centroidX = rawdata[i][0][j][0][1][0][0] #topleft is 0, 0
            centroidY = rawdata[i][0][j][0][1][0][1]
            frontX = rawdata[i][0][j][0][4][0][0]
            frontY = rawdata[i][0][j][0][5][0][0]
            dir = math.degrees(math.atan2(frontX-centroidX, centroidY-frontY)) #pointing up = 0
            OneCM = math.sqrt((centroidX - frontX)**2 + (centroidY - frontY)**2)/0.15
            if dir < 0:
                dir = 360+dir
            output = output.append([(frame, ID, centroidX, centroidY, dir, OneCM)], ignore_index=True)
    output = output.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm'})
    return(output) 

def mergeWrangle(filelist):
    merged = pandas.DataFrame()
    for i in filelist:
        merged = merged.append(wrangle(i))
    return(merged)

def combine(filename):
    cap = cv2.VideoCapture(filename)
    
    outname = os.path.splitext(os.path.basename(filename))[0]
    output = pandas.DataFrame()
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frameData = wrangle(outname, i)  
        output = pandas.concat([output, frameData], ignore_index=True)
        i+=1  
    
    cap.release()
    cv2.destroyAllWindows()
    
    output = output.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm'})
    output.to_csv(path_or_buf = outname + ".csv", na_rep = "NA", index = False)
    #output.to_csv(path_or_buf = "P1000023" + ".csv", na_rep = "NA", index = False)

    return 0

def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]
    
    combine(filename)
    return 0

def main(argv):
    """ Main entry point of the program """
    VideoID = str(argv[1])
    #VideoID = 'BEE'
    n=0
    for (dir, subdir, files) in subprocess.os.walk('../../TrackBEETag/Results'):
        for file in files:
            NAME = [i for i in file]
            if ''.join(NAME[0:len(VideoID)+1]) == VideoID + '_':
                n = n + 1
        VideoChunks = [dir + '/' + VideoID + '_' + str(i+1) + '.mat' for i in range(n)]

    merged = mergeWrangle(VideoChunks)
    merged.to_csv(path_or_buf = "../Data/" + VideoID + "_Merged.csv", na_rep = "NA", index = False)
    
    removed = removeraretag(merged, int(argv[2]))
    #removed = removeraretag(merged, 5)
    removed.to_csv(path_or_buf = "../Data/" + VideoID + "_Removed.csv", na_rep = "NA", index = False)
    
    IDs = removed["ID"].unique()
    wrangled = mergeSpeed(removed, IDs)
    wrangled = wrangled.sort_index(ascending=True)
    wrangled.to_csv(path_or_buf = "../Data/" + VideoID + "_Wrangled.csv", na_rep = "NA", index = False)
    print("Done! Wrangled data from " + VideoID + " is in Data.")
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)