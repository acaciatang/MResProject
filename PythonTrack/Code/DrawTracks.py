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
def getCoor(outname, id):
    """ Reads csvfile and fills in missing data within a certain time range for specified ID. """
    csvfile = '../Results/' + outname + ".csv"
    all = pd.read_csv(csvfile)
    subset = all.loc[all["ID"] == id, ["frame", "centroidX", "centroidY"]] # read in frame, centroid coordinates
    
    #if len(subset.index) > 0:
    #    subset.to_csv(path_or_buf = outname + "_" + str(id) + ".csv", na_rep = "NA", index = False)
    
    return subset.reset_index(drop=True)

def getallCoor(outname):
    """Fills in coordinates for all IDs."""
    csvfile = '../Results/' + outname + ".csv"
    all = pd.read_csv(csvfile)
    IDs = pd.DataFrame(all["ID"].value_counts())
    IDs.reset_index(level=0, inplace=True)
    IDs.columns = ["ID", "freq"]
    IDs = IDs.loc[IDs["freq"] > 1, ["ID", "freq"]]
    allCoors = [getCoor(outname, i) for i in IDs["ID"]]
    allCoors = [i for i in allCoors if i.empty == False]
    
    return allCoors

def chooseColour(i):
    """ Choose unique colour for each ID. """
    fullList = mcolors.CSS4_COLORS
    names = ['lightcoral', 'silver', 'royalblue', 'pink', 'plum', 'orangered', 'navy', 'lightgreen', 'purple', 'mediumvioletred', 'tomato', 'maroon','slateblue', 'red', 'saddlebrown', 'sandybrown', 'peru', 'palegreen', 'burlywood', 'goldenrod', 'lime', 'darkkhaki', 'orange', 'yellow', 'yellowgreen', 'olivedrab', 'green', 'darkgreen', 'darkseagreen', 'turquoise', 'teal', 'aqua', 'steelblue', 'dodgerblue', 'blue', 'deepskyblue', 'blueviolet', 'magenta', 'deeppink', 'crimson']
    colour = [c for c in fullList[names[i]]]
    r = int(colour[1] + colour[2], 16)
    g = int(colour[3] + colour[4], 16)
    b = int(colour[5] + colour[6], 16)
    return (b, g, r) #opencv uses BGR for some god forsaken reason

def drawLines(allCoors, FRAME, frameNum):
    """ Draws all lines that should be there for one frame """
    frame = cv2.cvtColor(FRAME[:,:,2],cv2.COLOR_GRAY2RGB)
    for i in range(len(allCoors)): #for each ID
        isClosed = False #don't want polygon
        # choose colour 
        color = chooseColour(i)
        # Line thickness of 2 px 
        thickness = 3

        df = allCoors[i]
        df = df[df['frame'] <= frameNum]
        if df.empty == True: # nothing to plot in this frame because there is nothing
            drew = frame
            continue
        if frameNum != df['frame'].max(): # nothing to plot in this frame because tracks have stopped before this frame
            drew = frame
            continue
        test = frameNum - int(df[df['frame'] == frameNum].index.values[0]) #fix me in Wrangled: should only have one entry per ID per frame!
        df['gap'] = [int(df['frame'][i] - i) != test for i in df.index]
        toPlot = df.loc[df['gap'] == False, ["centroidX", "centroidY"]]

        pts = np.array(toPlot)
        pts = pts.reshape((-1, 1, 2)) 
        
        drew = cv2.polylines(frame, np.int32([pts]), isClosed, color, thickness)
        frame = drew
    return drew

def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        if argv[1] == '.':
            files = [f for f in os.listdir('.') if f[-4:-1] == '.MP']
            for filename in files:
                outname = '../Results/' + os.path.splitext(os.path.basename(filename))[0]
                allCoors = getallCoor(outname, thres = 10)
                
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
        else:
            filename = argv[1]
            outname = '../Results/' + os.path.splitext(os.path.basename(filename))[0]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/Replicate1/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/Replicate1/Data')]
        filename = files[int(iter)-1]
        outname = os.path.splitext(os.path.basename(filename))[0]
    
    allCoors = getallCoor(outname)
    
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