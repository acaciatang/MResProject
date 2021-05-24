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
    frameSeries = pd.DataFrame(index = nested, dtype = 'float64', columns = ['X', 'Y', 'OGindex'])
    for row in range(frameData.shape[0]):
        print(row)
        frameSeries.loc[(frameData.iloc[row]['ID'], frameNum), "X"] = frameData.iloc[row]['centroidX']
        frameSeries.loc[(frameData.iloc[row]['ID'], frameNum), "Y"] = frameData.iloc[row]['centroidY']
        frameSeries.loc[(frameData.iloc[row]['ID'], frameNum), "OGindex"] = frameData.iloc[row].name
        
    return frameSeries

def reshape(raw, taglist):
    reshaped = [reshapeframe(raw, float(f), taglist) for f in list(pd.unique(raw['frame']))]
    reshaped = pd.concat(reshaped, ignore_index=False)
    return reshaped

def IDdistance(found, ID, thresf = 40, thres = 50, thres2 = 100):
    subset = found.loc[ID]
    frames = list(subset.index)
    if subset.shape[0] < thresf:
        cat = [False for i in range(subset.shape[0])]
        return [frames, cat]
    distance = [math.sqrt((subset.loc[frames[f], 'X'] - subset.loc[frames[f-1], 'X'])**2 + (subset.loc[frames[f], 'Y'] - subset.loc[frames[f-1], 'Y'])**2) for f in range(len(frames))]
    distance[0] = 0
    time = [(frames[f] - frames[f-1]) for f in range(len(frames))]
    time[0] = 1
    speed = [distance[f]/time[f] for f in range(len(frames))]
    base = 0
    cat = []
    for i in range(len(speed)):
        if speed[i] > thres or distance[i] > thres2:
            base = base + 1
        cat.append(base)
    
    categories = list(set(cat))
    end = [subset.iloc[cat.index(c+1)-1] for c in categories if c != max(cat)]
    noZero = copy.deepcopy(categories)
    noZero.pop(0)
    for c in noZero:
        start = subset.iloc[cat.index(c)]
        test = [math.sqrt((start[0] - end[i][0])**2 + (start[1] - end[i][1])**2)/(c-i) for i in range(c)]
        minimum = test.index(min(test))
        if test[minimum] < thres and math.sqrt((start[0] - end[minimum][0])**2 + (start[1] - end[minimum][1])**2) < thres2:
            categories[c] = categories[test.index(min(test))]
            cat = [categories[c] if i == c else i for i in cat]

    table = collections.Counter(cat)
    counts = [table[i] for i in range(len(table))]
    cat = [c == counts.index(max(counts)) for c in cat] # True is good, False is sus
    #subset['cat'] = cat
    #subset.to_csv(path_or_buf = "324.csv", na_rep = "NA", index = True)
            
    return [frames, cat]

def wrangle(outname, thres = 50, thres2 = 100):
    print('Reading file...')
    raw = pd.read_csv('../Results/' + outname + '.csv')
    
    if outname[6] == 'A':
        taglist = [237,74,121,137,151,180,181,220,222,311,312,341,402,421,427,456,467,596,626,645,664,681,696,697,765,781,794,862,985,1077,1419,1846,1947,1966,2908,2915]
    elif outname[6] == 'B':
        taglist = [180,74,121,137,151,181,186,220,222,237,311,312,341,393,421,427,467,534,574,596,626,645,664,681,696,697,765,781,862,985,1077,1419,1846,1947,1966,2908,2915]
    elif outname[6] == 'C':
        taglist = [862,121,137,151,180,181,186,220,222,237,341,393,402,421,456,467,534,574,596,626,645,664,681,696,697,765,781,794,985,1077,1419,1846,1947,1966,2908,2915]
    elif outname[6] == 'D':
        taglist = [534,74,121,137,151,186,220,222,237,311,312,341,393,402,421,427,456,467,574,596,626,645,664,681,696,697,781,794,862,985,1077,1419,1846,1947,1966,2908]
    
    wrangled = copy.deepcopy(raw)
    print('Done!')
    print('Reshaping file...')
    reshaped = reshape(raw, taglist)
    print('Done!')
    print('Wrangling...')
    holder = copy.deepcopy(reshaped)
    found = reshaped.dropna(axis = 0)
    foundtags = set([i[0] for i in found.index])
    for id in foundtags:
        if IDdistance(found, id) != None:
            print('ID: ' + str(id))
            frames, cat = IDdistance(found, id)
            if id == 'X':
                cat = [False]*len(cat)
            else:
                cat = [True]*len(cat)
            for f in range(len(frames)):
                holder.loc[(id,frames[f]),'cat'] = cat[f]
                print('frame: ' + str(f))

    #havespeed = holder.dropna(axis = 0) # plotting for 
    noNA = holder.dropna(axis = 0)
    good = noNA[noNA.loc[:, 'cat']]
    sus = noNA[noNA.loc[:, 'cat'] == False]
    
    for r in range(sus.shape[0]): #for each potentially mislabelled entry
        row = sus.iloc[r]
        previous = good.loc[(slice(None), [i for i in range(int(row.name[1]))]), slice(None)]
        goodframes = list(set([i[0] for i in good.index]))
        if int(row.name[1]) in goodframes:
            current = good.loc[(slice(None), int(row.name[1])), slice(None)]
            ids = list(set([i[0] for i in previous.index])-set([i[0] for i in current.index]))
        else:
            ids = list(set([i[0] for i in previous.index]))
        
        if len(ids) == 0:
            wrangled.iat[int(row[2]), 1] = None
        else:
            testpts = [previous.loc[id].iloc[-1] if len(previous.loc[id].shape) > 1 else previous.loc[id] for id in ids]
            test = [math.sqrt((row[0] - testpts[i][0])**2 + (row[1] - testpts[i][1])**2)/(row.name[1]-testpts[i].name) for i in range(len(testpts))]
            minimum = test.index(min(test))
            if test[minimum] < thres and math.sqrt((row[0] - testpts[minimum][0])**2 + (row[1] - testpts[minimum][1])**2) < thres2:
                wrangled.iat[int(row[2]), 1] = ids[test.index(min(test))]
            else:
                wrangled.iat[int(row[2]), 1] = None
    
    wrangled = wrangled.dropna(axis = 0)
    wrangled.to_csv(path_or_buf = '../Results/' + outname + "_final.csv", na_rep = "NA", index = True)
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