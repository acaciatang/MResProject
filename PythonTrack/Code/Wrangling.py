#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'RemoveBackground.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import os
import pandas as pd
import math
import copy

def caldis(row1, row2):
    return math.sqrt((row1['centroidX']-row2['centroidX'])**2 + (row1['centroidY']-row2['centroidY'])**2)

def remove(oneID, thres1):
    if len(oneID.index) < max(oneID.frame)/100:
        removeme = oneID
        input = pd.DataFrame()
        print(set(oneID.ID))
        return input, removeme
    removeme = pd.DataFrame()
    oneID = oneID.sort_values("frame", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)
    phydis = [caldis(oneID.loc[oneID.index[r-1]], oneID.loc[oneID.index[r]]) for r in range(1, oneID.shape[0])]
    if len(phydis) > 1:
        removeindex = [oneID.index[i+1] for i in range(len(phydis)-2) if phydis[i] > thres1*2 and phydis[i+1] > thres1*2]
    else:
        removeindex = list()
    removeme = oneID.loc[removeindex]
    removeme.ID = 'X'
    input = oneID.drop(removeme.index)
    return input, removeme

def relabel(id, oneID, noID, thres1, thres2):
    #trial 1: from top
    addme = pd.DataFrame()
    h = min(oneID.frame)
    t = max(oneID.frame)
    trial = pd.DataFrame()
    model = oneID[oneID["frame"] == h]
    f1 = h+1
    while f1 < t:
        if f1 not in list(oneID.frame):
            candidates = noID[noID["frame"] == f1]
            if len(candidates.index) != 0:
                distances = [caldis(model, candidates.loc[i]) for i in candidates.index]
                if min(distances) < thres1 and f1-h < thres2:
                    model = copy.deepcopy(candidates.loc[candidates.index[distances.index(min(distances))]]).append(pd.Series([f1-h]))
                    model[1] = id
                    trial = trial.append([model], ignore_index=True)
                    h = model['frame']
                    f1 = f1+1
                else:
                    if f1-h > thres2:
                        nextones = [i for i in oneID.frame if i > f1]
                        h = nextones[0]
                        model = oneID[oneID["frame"] == h]
                        f1 = f1+1
                    else:
                        f1 = f1+1
            else:
                f1 = f1+1
        else:
            h = f1
            model = oneID[oneID["frame"] == h]
            f1 = f1+1
    #trial 2: from bottom
    h = min(oneID.frame)
    t = max(oneID.frame)
    model = oneID[oneID["frame"] == t]
    f2 = t-1
    while f2 > h:
        if f2 not in list(oneID.frame):
            candidates = noID[noID["frame"] == f2]
            if len(candidates.index) != 0:
                distances = [caldis(model, candidates.loc[i]) for i in candidates.index]
                if min(distances) < thres1 and t-f2 < thres2:
                    model = copy.deepcopy(candidates.loc[candidates.index[distances.index(min(distances))]]).append(pd.Series([abs(t-f2)]))
                    model[1] = id
                    trial = trial.append([model], ignore_index=True)
                    t = model['frame']
                    f2 = f2-1
                else:
                    if t-f2 > thres2:
                        nextones = [i for i in oneID.frame if i < f2]
                        h = nextones[-1]
                        model = oneID[oneID["frame"] == h]
                        f2 = f2-1
                    else:
                        f2 = f2-1
            else:
                f2 = f2-1
        else:
            t = f2
            model = oneID[oneID["frame"] == t]
            f2 = f2-1

    #combine the two: keep one that is closer to a found tag if different
    if len(trial.index) > 0:
        frames = set(trial.frame)
        for f in frames:
            subset = trial[trial['frame'] == f]
            if len(subset.index) == 1:
                addme = addme.append(subset.iloc[0][0:7])
            elif len(set(subset.centroidX)) == 1 and len(set(subset.centroidY)) == 1:
                addme = addme.append(subset.iloc[0][0:7])
            elif len(set(subset[0])) == 1:
                absdiff = [abs(f-frame) for frame in oneID.frame]
                diff = [f-frame for frame in oneID.frame]
                test = diff[absdiff.index(min(absdiff))]
                if test < 0:
                    addme.append(subset.loc[max(subset.index)][0:7])
                else:
                    addme.append(subset.loc[min(subset.index)][0:7])

            else:
                addme = addme.append(subset.loc[subset[0].idxmin()][0:7])
        
    #add to top
    h = min(oneID.frame)
    t = max(oneID.frame)
    if h > 0:
        model = oneID[oneID["frame"] == h]
        f3 = h-1
        while f3 >= 0:
            candidates = noID[noID["frame"] == f3]
            if len(candidates.index) != 0:
                distances = [caldis(model, candidates.loc[i]) for i in candidates.index]
                if min(distances) < thres1 and f3-h < thres2:
                    model = copy.deepcopy(candidates.loc[candidates.index[distances.index(min(distances))]])
                    model[1] = id
                    oneID = oneID.append([model], ignore_index=True)
                    h = model['frame']
                    f3 = f3-1
                else:
                    f3 = f3-1
            else:
                f3 = f3-1
    #add to bottom    
    h = min(oneID.frame)
    t = max(oneID.frame)
    if t < max(noID.frame):
        model = oneID[oneID["frame"] == t]
        f4 = t+1
        while f4 <= max(noID.frame):
            candidates = noID[noID["frame"] == f4]
            if len(candidates.index) != 0:
                distances = [caldis(model, candidates.loc[i]) for i in candidates.index]
                if min(distances) < thres1 and t-f4 < thres2:
                    model = copy.deepcopy(candidates.loc[candidates.index[distances.index(min(distances))]])
                    model[1] = id
                    oneID = oneID.append([model], ignore_index=True)
                    t = model['frame']
                    f4 = f4+1
                else:
                    f4 = f4+1
            else:
                f4 = f4+1
    oneID = oneID.append(addme, ignore_index=True)
    return oneID.sort_values("frame", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)
    
def addmissing(oneID, thres3, thres4):
    missing = pd.DataFrame()
    for i in range(oneID.shape[0]-1): #for each row except the last
        if 1 < oneID["frame"][i+1] - oneID["frame"][i] < thres3 and math.sqrt((oneID["centroidX"][i+1] - oneID["centroidX"][i])**2 + (oneID["centroidY"][i+1] - oneID["centroidY"][i])**2) < thres4: #threshold by time and distance
            addframe = pd.DataFrame(list(range(int(oneID["frame"][i] + 1), int(oneID["frame"][i+1]))))
            addX = pd.DataFrame([oneID["centroidX"][i] + (oneID["centroidX"][i+1]-oneID["centroidX"][i]) *(j+1)/len(addframe) for j in range(len(addframe))])
            addY = pd.DataFrame([oneID["centroidY"][i] + (oneID["centroidY"][i+1]-oneID["centroidY"][i]) *(j+1)/len(addframe) for j in range(len(addframe))])
            addme = pd.concat([addframe, addX, addY], axis = 1)
            missing = missing.append(addme)
    
    if missing.shape != (0, 0):
        missing.columns = ["frame", "centroidX", "centroidY"] # 0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm', 6:'score'  
        missing['ID'] = oneID.ID[0]
        missing['dir'] = 'X'
        missing['1cm'] = 'X'
        missing['score'] = 'X'

        oneID = oneID.append(missing)
        oneID = oneID.sort_values("frame", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)

    return oneID

def wrangle(outname, thres1 = 30, thres2 = 30, thres3 = 200, thres4 = 150):
    #split data by ID, find gaps
    raw = pd.read_csv(outname + '_raw.csv')
    noID = pd.read_csv(outname + '_noID.csv')
    IDs = set(raw.ID)
    wrangled = pd.DataFrame()
    Input = pd.DataFrame()
    print('Removing problems')
    for id in IDs:
        oneID = raw[raw["ID"] == id]
        input, removeme = remove(oneID, thres3)
        noID = noID.append(removeme)
        Input = Input.append(input)
    
    print('Relabelling and Joining')
    InputIDs = set(Input.ID)
    for id in InputIDs:
        print(id)
        oneIDInput = Input[Input["ID"] == id]
        oneID = relabel(id, oneIDInput, noID, thres1, thres2)
        oneID = addmissing(oneID, thres3, thres4)

        wrangled = wrangled.append(oneID)
    
    wrangled = wrangled.sort_values(by=['frame'])
    wrangled = wrangled.dropna(axis = 0)
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
        outname = '../Results/' + os.path.splitext(os.path.basename(filename))[0]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/Replicate1/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/Replicate1/Data')]
        filename = files[int(iter)-1]
        outname = os.path.splitext(os.path.basename(filename))[0]
    #filename = '../Data/R1D7R2A1_trimmed.MP4'
    
    return wrangle(outname)

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)

