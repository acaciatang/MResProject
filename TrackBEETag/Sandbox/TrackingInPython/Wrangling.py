#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'RemoveBackground.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import cv2
import os
import numpy as np
import pandas as pd
import math
import copy
import collections
#import matplotlib.pyplot as plt

def reshapeframe(raw, frameNum, taglist):
    frameData = raw[raw["frame"] == frameNum]
    iterables = [taglist, [frameNum]]
    nested = pd.MultiIndex.from_product(iterables, names=["ID", "frame"])
    frameSeries = pd.DataFrame(index = nested, dtype = 'float64', columns = ['X', 'Y'])
    for row in range(frameData.shape[0]):
        frameSeries.loc[(frameData.iloc[row]['ID'], frameNum), "X"] = frameData.iloc[row]['centroidX']
        frameSeries.loc[(frameData.iloc[row]['ID'], frameNum), "Y"] = frameData.iloc[row]['centroidY']
        
    return frameSeries

def reshape(raw, taglist):
    reshaped = [reshapeframe(raw, f, taglist) for f in list(pd.unique(raw['frame']))]
    reshaped = pd.concat(reshaped, ignore_index=False)
    return reshaped

def IDdistance(found, ID, thres = 50):
    subset = found.loc[ID]
    if subset.shape[0] < 2:
        return None
    frames = list(subset.index)
    distance = [math.sqrt((subset.loc[frames[f], 'X'] - subset.loc[frames[f-1], 'X'])**2 + (subset.loc[frames[f], 'Y'] - subset.loc[frames[f-1], 'Y'])**2) for f in range(len(frames))]
    distance[0] = 0
    time = [(frames[f] - frames[f-1]) for f in range(len(frames))]
    time[0] = 1
    speed = [distance[f]/time[f] for f in range(len(frames))]
    base = 0
    cat = []
    for i in speed:
        if i > thres:
            base = base + 1
        cat.append(base)
    
    categories = list(set(cat))
    end = [subset.iloc[cat.index(c+1)-1] for c in categories if c != max(cat)]
    noZero = copy.deepcopy(categories)
    noZero.pop(0)
    for c in noZero:
        start = subset.iloc[cat.index(c)]
        test = [math.sqrt((start[0] - end[i][0])**2 + (start[1] - end[i][1])**2)/(c-i) for i in range(c)]
        if min(test) < thres:
            categories[c] = categories[test.index(min(test))]
            cat = [categories[c] if i == c else i for i in cat]

    table = collections.Counter(cat)
    counts = [table[i] for i in range(len(table))]
    cat = [c == counts.index(max(counts)) for c in cat]
    #subset['cat'] = cat
    #subset.to_csv(path_or_buf = "324.csv", na_rep = "NA", index = True)
            
    return [frames, cat]

def wrangle(filename, thres = 20):
    raw = pd.read_csv(filename)
    outname = os.path.splitext(os.path.basename(filename))[0]
    if filename[0] == 'A':
        taglist = [68,118,137,173,289,304,325,365,392,420,437,512,559,596,613,666,696,765,862,1112,1150,1203,1492,1730,1966,2091,2327,2452,2511,2932,2992,3067,3261,3360,3415,3486,3570,3757,3908,4015]
    elif filename[0] == 'B':
        taglist = [31,46,69,180,222,270,311,330,347,393,542,598,651,697,792,813,875,1062,1085,1227,1368,1498,1585,1744,1947,1986,2056,2158,2281,2332,2460,2607,2835,2908,2945,3375,3488,3581,3783,3926]
    elif filename[0] == 'C':
        taglist = [52,74,103,209,226,274,312,331,354,427,455,476,502,544,574,601,634,661,707,770,881,1028,1180,1243,1465,1543,1704,1759,1797,1846,1896,2118,2340,2413,2488,2523,2915,2954,3134,3832]
    elif filename[0] == 'D':
        taglist = [59,75,104,135,211,237,324,341,361,377,413,436,456,510,579,609,637,664,681,720,802,844,910,1074,1104,1403,1620,1718,1799,1903,2006,2072,2192,2242,2355,2856,2880,3163,3358,3388]
    elif filename[0] == 'P':
        taglist = [59,68,74,75,103,104,135,137,180,211,274,304,311,312,324,325,330,331,392,393,413,502,544,613,637,651,696,707,792,1104,1112,1465,1543,1759,1846,1903,2056,2856,2945,3163]
    
    wrangled = copy.deepcopy(raw)
    reshaped = reshape(raw, taglist)
    holder = copy.deepcopy(reshaped)
    found = reshaped.dropna(axis = 0)
    foundtags = set([i[0] for i in found.index])
    for id in foundtags:
        if IDdistance(found, id) != None:
            frames = IDdistance(found, id)[0]
            cat = IDdistance(found, id)[1]
            for f in range(len(frames)):
                holder.loc[(id,frames[f]),'cat'] = cat[f]
    
    #havespeed = holder.dropna(axis = 0) # plotting for 
    noNA = holder.dropna(axis = 0)
    good = noNA[noNA.loc[:, 'cat']]
    sus = noNA[noNA.loc[:, 'cat'] == False]
    
    for r in range(sus.shape[0]):
        row = sus.iloc[r]
        changeme = wrangled[wrangled["frame"] == row.name[1]][wrangled["centroidX"] == row[0]][wrangled["centroidY"] == row[1]]
        previous = good.loc[(slice(None), [i for i in range(int(row.name[1]))]), slice(None)]
        ids = list(set([i[0] for i in previous.index]))
        testpts = [previous.loc[id].iloc[-1] if len(previous.loc[id].shape) > 1 else previous.loc[id] for id in ids]
        test = [math.sqrt((row[0] - testpts[i][0])**2 + (row[1] - testpts[i][1])**2)/(row.name[1]-testpts[i].name) for i in range(len(testpts))]
        if len(test) == 0:
            wrangled.iat[changeme.index[0], 1] = None
        elif min(test) < thres:
            wrangled.iat[changeme.index[0], 1] = ids[test.index(min(test))]
        else:
            wrangled.iat[changeme.index[0], 1] = None
    
    wrangled = wrangled.dropna(axis = 0)
    wrangled.to_csv(path_or_buf = outname + ".csv", na_rep = "NA", index = True)
    #raw.to_csv(path_or_buf = outname + "raw.csv", na_rep = "NA", index = True)
    print("Done!")
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
        print(filename)
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]

    return wrangle(filename)

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)