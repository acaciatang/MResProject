#!/usr/bin/env python3

"""Analyses behaviour based on algorithms from Crall et al"""

__appname__ = 'AnalyseBehaviour'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'


## imports ##
import sys # module to interface our program with the operating system
import scipy.io
import pandas
import numpy

## constants ##

## functions ##
def transformresults(file):
    rawdata = scipy.io.loadmat(file)['trackingData'][0]
    # a frame: rawdata[0][FrameNumber-1]
    # a code: rawdata[0][FrameNumber-1][0][CodeNumber]
    # rawdata[0][FrameNumber-1][0][CodeNumber][1] = centroid
    # rawdata[0][FrameNumber-1][0][CodeNumber][6] = code identity
    # direction: 



def main(argv):
    """ Main entry point of the program """
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)