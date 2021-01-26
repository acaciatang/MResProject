#!/usr/bin/env python3

"""Prints 'This is a boilerplate'."""

__appname__ = 'RunMatLab.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

## imports ##
import sys # module to interface our program with the operating system
import matlab.engine
import os
#import time

## constants ##


## functions ##
def GetMov(file):
    eng = matlab.engine.start_matlab()
    filename = os.path.abspath(file)
    outname = os.path.splitext(filename)[0]
    eng.GetMov(filename, outname, nargout=0)
    eng.quit()

def main(argv):
    """ Main entry point of the program """
    directory = os.path.abspath(argv[1])
    for i in os.listdir(directory):
        if ".mp4" in i:
            f = directory + "/" + i
            GetMov(f) 
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)

###########################