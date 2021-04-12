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
def getCoor(csvfile, id, thres = 60):
    all = pd.read_csv(csvfile)
    subset = all.loc[all["ID"] == id, ["frame", "centroidX", "centroidY"]]
    subset
    missing = pd.DataFrame()
    subset = subset.append(missing, ignore_index=True)

    for i in range(subset.shape[0]-1):
        if 1 < subset["frame"][i+1] - subset["frame"][i] < thres:
            addframe = pd.DataFrame(list(range(subset["frame"][i] + 1, subset["frame"][i+1])))
            addX = pd.DataFrame([subset["centroidX"][i] + (subset["centroidX"][i+1]-subset["centroidX"][i]) *(j+1)/len(addframe) for j in range(len(addframe))])
            addY = pd.DataFrame([subset["centroidY"][i] + (subset["centroidY"][i+1]-subset["centroidY"][i]) *(j+1)/len(addframe) for j in range(len(addframe))])
            addme = pd.concat([addframe, addX, addY], axis = 1)
            missing = missing.append(addme)

    if missing.shape != (0, 0):
        missing.columns = ["frame", "centroidX", "centroidY"]    
        subset = subset.append(missing)
        subset = subset.sort_values("frame", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)

    test1 = subset.frame.diff().fillna(thres*10) == 1
    test2 = abs(subset.frame.diff(periods=-1).fillna(thres*10)) == 1
    keep = test1|test2
    subset['keep'] = keep
    subset = subset.loc[subset["keep"] == True, ["frame", "centroidX", "centroidY"]]
    
    return subset.reset_index(drop=True)

def getallCoor(csvfile, thres = 60):
    all = pd.read_csv(csvfile)
    IDs = pd.DataFrame(all["ID"].value_counts())
    IDs.reset_index(level=0, inplace=True)
    IDs.columns = ["ID", "freq"]
    IDs = IDs.loc[IDs["freq"] > 1, ["ID", "freq"]]
    allCoors = [getCoor(csvfile, i, thres = 50) for i in IDs["ID"]]
    allCoors = [i for i in allCoors if i.empty == False]
    
    return allCoors

def chooseColour(i, gap):
    fullList = mcolors.CSS4_COLORS
    names = list(fullList.keys())
    colour = [i for i in fullList[names[i*gap]]]
    r = int(colour[1] + colour[2], 16)
    g = int(colour[3] + colour[4], 16)
    b = int(colour[5] + colour[6], 16)
    return (b, g, r) #opencv uses BGR for some god forsaken reason
# FRAME = cv2.imread('hive_999.png',1)
def drawLines(allCoors, FRAME, frameNum):
    frame = FRAME
    for i in range(len(allCoors)): #for each ID
        isClosed = False #don't want polygon
        # choose colour 
        color = chooseColour(i, math.floor(148/len(allCoors)))
        # Line thickness of 2 px 
        thickness = 3

        df = allCoors[i]
        df = df[df['frame'] <= frameNum]
        if df.empty == True:
            continue
        if frameNum != df['frame'].max():
            continue
        test = frameNum - int(df[df['frame'] == frameNum].index.values[0]) #fix me in Wrangled: should only have one entry per ID per frame!
        df['gap'] = [int(df['frame'][i] - i) != test for i in df.index]
        df = df.loc[df['gap'] == False, ["centroidX", "centroidY"]]

        pts = np.array(df)
        pts = pts.reshape((-1, 1, 2)) 
        
        drew = cv2.polylines(frame, np.int32([pts]), isClosed, color, thickness)
        frame = drew

    return drew
def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]
    outname = os.path.splitext(os.path.basename(filename))[0]
    allCoors = getallCoor(outname + ".csv", thres = 50)
    
    cap = cv2.VideoCapture(filename)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outname + "_tracks.mp4", fourcc, 20.0, (3840,2160))

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
                break
        frame = drawLines(allCoors, frame, i)

        # write the flipped frame
        out.write(frame)
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