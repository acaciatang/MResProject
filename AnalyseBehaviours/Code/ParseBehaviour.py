#!/usr/bin/env python3

"""Takes in wrangled data and determines behaviours"""

__appname__ = 'ParseBehaviour'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'


## imports ##
import sys
import scipy.io
import pandas
import math
import subprocess
import re
import numpy

## constants ##

#polygon = feed
#point = numpy.array([wrangled.loc[0, 'centroidX'], wrangled.loc[0, 'centroidY']])

## functions ##
def positive(angle):
    if angle < 0:
        return(360+angle)
    else:
        return(angle)

def InOut(polygon, point): #determines if point is in polygon, False = out, True = in
    #if on convex vertex will be taken as out, if vertex is concave will be in
    dirs = [math.degrees(math.atan2(polygon[i,0]-point[0], point[1]-polygon[i,1])) for i in range(len(polygon))]
    dirs2 = list(dirs)
    for i in range(len(dirs2)):
        dirs2[i] = positive(dirs2[i])

    angdiff1 = pandas.DataFrame([abs(dirs[i-1] - dirs[i]) for i in range(len(dirs))])
    angdiff2 = pandas.DataFrame([abs(dirs2[i-1] - dirs2[i]) for i in range(len(dirs2))])

    AngDiff = angdiff1.combine(angdiff2, numpy.minimum)
    test = float(AngDiff.sum() - AngDiff.max())
    if test < 180:
        return(False)
    else:
        return(True)

def Movement(speed):
    min = 0.1 # can change
    max = 10 #can change
    if min < speed < max:
        return(True)
    else:
        return(False)

def Interact(frameNum, dataset):
    subset = dataset[dataset["frame"] == frameNum]
    #threshold = subset.mean()['1cm']
    threshold = 150

    IDs = numpy.array(subset.index)

    for i in IDs:
        for j in IDs[IDs > i]:
            test = math.sqrt((sub.loc[i, 'centroidX']-sub.loc[j, 'centroidX'])**2 + (sub.loc[i, 'centroidY']-sub.loc[j, 'centroidY'])**2)
            if test < threshold:
                dataset.loc[i, 'interacting'] = j

def main(argv):
    """ Main entry point of the program """
    VideoID = argv[1]
    nest = scipy.io.loadmat("../../TrackBEETag/Results/" + VideoID + "nest.mat")['nest']
    feed = scipy.io.loadmat("../../TrackBEETag/Results/" + VideoID + "feed.mat")['feed']
    wrangled = pandas.read_csv("../Data/" + VideoID + "_Wrangled.csv")

    nursing = [InOut(nest, numpy.array([wrangled.loc[i, 'centroidX'], wrangled.loc[i, 'centroidY']])) for i in range(wrangled.shape[0])]
    nursing = pandas.DataFrame({"nursing": nursing})
    
    foraging = [InOut(feed, numpy.array([wrangled.loc[i, 'centroidX'], wrangled.loc[i, 'centroidY']])) for i in range(wrangled.shape[0])]
    foraging = pandas.DataFrame({"foraging": foraging})
    
    moving = [Movement(wrangled.loc[i, 'speed']) for i in range(wrangled.shape[0])]
    moving = pandas.DataFrame({"moving": moving})

    Behaviours = pandas.concat([wrangled, nursing, foraging, moving], axis=1)
        
    Behaviours[['interacting']] = None
    for i in Behaviours["frame"].unique():
        Interact(i, Behaviours)
    
    Behaviours.to_csv(path_or_buf = "../Results/" + VideoID + "_Behaviours.csv", na_rep = "NA", index = False)


    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)