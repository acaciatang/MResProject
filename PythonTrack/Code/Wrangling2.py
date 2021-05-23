#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'RemoveBackground.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
from PythonTracking import closesttosides
from _typeshed import OpenBinaryModeUpdating
from io import RawIOBase
import sys 
import cv2
import os
import numpy as np
import pandas as pd
import math
import copy
import collections

def caldis(row1, row2):
    phy = math.sqrt((row1[3]-row2[3])**2 + (row1[4]-row2[4])**2)
    time = abs(row1[0] - row2[0])
    return [phy, time]

def relabel(oneID, gaps, head, thres1, thres2):
    noID = pd.read_csv('../Results/' + outname + '_noID.csv')
    for h in head:
        find closest in noID
        check if within threshold
        if yes, add row to oneID
    for g in gaps:
        find closest based on either end otherwise same
    return oneID

def addmissing(oneID):
    for i in range(oneID.shape[0]-1): #for each row except the last
        if 1 < oneID["frame"][i+1] - oneID["frame"][i] < thres and math.sqrt((oneID["centroidX"][i+1] - oneID["centroidX"][i])**2 + (oneID["centroidY"][i+1] - oneID["centroidY"][i])**2) < 10000: #threshold by time and distance
            addframe = pd.DataFrame(list(range(int(oneID["frame"][i] + 1), int(oneID["frame"][i+1]))))
            addX = pd.DataFrame([oneID["centroidX"][i] + (oneID["centroidX"][i+1]-oneID["centroidX"][i]) *(j+1)/len(addframe) for j in range(len(addframe))])
            addY = pd.DataFrame([oneID["centroidY"][i] + (oneID["centroidY"][i+1]-oneID["centroidY"][i]) *(j+1)/len(addframe) for j in range(len(addframe))])
            addme = pd.concat([addframe, addX, addY], axis = 1)
            missing = missing.append(addme)
    
    if missing.shape != (0, 0):
        missing.columns = ["frame", "centroidX", "centroidY"]    
        oneID = oneID.append(missing)
        oneID = oneID.sort_values("frame", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)

    return oneID

def wrangle(outname):
    #split data by ID, find gaps
    raw = pd.read_csv('../Results/' + outname + '_raw.csv')
    IDs = set(raw.ID)
    wrangled = pd.DataFrame()
    for id in IDs:
        oneID = raw[raw["ID"] == id]
        gaps = [i for i in range(max(raw.frame)) if ((i not in list(oneID.frame)) and (i > min(oneID.frame)))]
        head = [i for i in range(min(oneID.frame))]
        head.reverse()

        oneID = relabel(oneID)
        oneID = addmissing(oneID)

        wrangled = wrangled.append(oneID)
    
    wrangled = wrangled.sort_values(by=['frame'])
    wrangled.to_csv(path_or_buf = outname + ".csv", na_rep = "NA", index = False)
    
    return 0

def main(argv):
    if len(sys.argv) == 2:
        #if argv[1] == '.':
        #    files = [f for f in os.listdir('.') if f[-4:-1] == '.MP']
        #    for filename in files:
        #        wrangle(filename)
        #    return 0
        #else:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]
    filename = '../Data/R1D7R2A1_trimmed.MP4'
    outname = os.path.splitext(os.path.basename(filename))[0]
    return wrangle(outname)

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)