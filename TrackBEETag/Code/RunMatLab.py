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
def track(file):
    eng = matlab.engine.start_matlab()
    filename = os.path.abspath(file)
    outname = os.path.splitext(filename)[0]
    eng.Tracking(filename, outname, nargout=0)
    eng.quit()

def main(argv):
    """ Main entry point of the program """
    track(argv[1])
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)

###########################