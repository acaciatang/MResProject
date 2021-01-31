#!/usr/bin/env python3

"""Takes in raw .mat files from BEETag tracking data belonging to the same video and puts it into long format for down-stream analysis"""

__appname__ = 'WrangleData'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

## imports ##
import sys
import scipy.io
import numpy
import pandas
import math
import subprocess
import re

## constants ##

## functions ##
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

def removeraretag(dataframe, n):
    table = dataframe["ID"].value_counts()
    removedRare = dataframe[dataframe["ID"].isin(table.index[0:n])]
    return(removedRare)

def speed(dataset, id):
    subset = dataset[dataset["ID"] == id]
    index = list(subset.index)
    
    time = [subset.loc[index[j+1], 'frame'] - subset.loc[index[j], 'frame'] for j in range(subset.shape[0]-1)]
    time.insert(0, numpy.NaN)
    time = pandas.DataFrame({"time": time})

    distance = [math.sqrt((dataset.loc[index[j+1], 'centroidX'] - dataset.loc[index[j], 'centroidX'])**2 + (dataset.loc[index[j+1], 'centroidY'] - dataset.loc[index[j], 'centroidY'])**2) for j in range(subset.shape[0]-1)]
    distance.insert(0, numpy.NaN)
    distance = pandas.DataFrame({"distance": distance})
    
    tds = pandas.concat([time, distance], axis=1)
    tds[['speed']] = tds[['distance']].div(tds.time, axis=0)
    tds.index = index

    subset = pandas.concat([subset, tds], axis=1)
        
    return(subset)

def mergeSpeed(dataset, IDs):
    merged = pandas.DataFrame()
    for i in IDs:
        merged = merged.append(speed(dataset, i))
    return(merged)

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