#!/usr/bin/env python3

"""Tests algorithm for tag reading based on training data."""

__appname__ = 'Training.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import cv2
import subprocess
import os
import PythonTracking

#code

def main(argv):
    for (dir, subdir, files) in subprocess.os.walk("../Data/Test"):
        for file in files:
            if file != '.DS_Store':
                print('Tracking ' + file)
                img = cv2.imread(dir + '/' + file)
                outname = os.path.splitext(file)[0]
                
                if outname[6] == 'A':
                    taglist = [237,74,121,137,151,180,181,220,222,311,312,341,402,421,427,456,467,596,626,645,664,681,696,697,765,781,794,862,985,1077,1419,1846,1947,1966,2908,2915]
                elif outname[6] == 'B':
                    taglist = [180,74,121,137,151,181,186,220,222,237,311,312,341,393,421,427,467,534,574,596,626,645,664,681,696,697,765,781,862,985,1077,1419,1846,1947,1966,2908,2915]
                elif outname[6] == 'C':
                    taglist = [862,121,137,151,180,181,186,220,222,237,341,393,402,421,456,467,534,574,596,626,645,664,681,696,697,765,781,794,985,1077,1419,1846,1947,1966,2908,2915]
                elif outname[6] == 'D':
                    taglist = [534,74,121,137,151,186,220,222,237,311,312,341,393,402,421,427,456,467,574,596,626,645,664,681,696,697,781,794,862,985,1077,1419,1846,1947,1966,2908]
                
                Coordinates, Cropped, bkgd = PythonTracking.findtags(img, outname)
                a = 0
                while a < len(Coordinates):
                    print(a)
                    bkgd, row = PythonTracking.drawtag(Coordinates[a], Cropped[a], bkgd, outname, a, taglist) #[results[2], 'centroidX', 'centroidY', results[1], 'OneCM', results[0]]
                    a = a+1
                cv2.imwrite('../Results/Test/' + outname + "_foundtags.png", bkgd)
                print('Finished ' + file)

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)