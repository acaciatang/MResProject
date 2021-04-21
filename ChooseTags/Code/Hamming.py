#!/usr/bin/env python3

"""Calculates Hamming Distance."""

__appname__ = 'RemoveBackground.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import os
import numpy as np

def drawmodel(id):
    binary = '%012d' %int(bin(id)[2:len(bin(id))])
    tag = np.array([int(bit) for bit in binary])
    tag = np.reshape(tag, (4,3), 'f')
    model = np.ones((6, 6))
    model[1:5, 4] = 0
    model[1:5, 1:4] = tag
    model[1, 4] = sum(model[1:5, 1])%2
    model[2, 4] = sum(model[1:5, 2])%2
    model[3, 4] = sum(model[1:5, 3])%2
    model[4, 4] = np.sum(model[1:5, 1:4])%2

    return model.astype(int)

def calhamming(row):
    model1 = drawmodel(row[0])
    model2 = drawmodel(row[1])
    configs = [model2, np.rot90(model2, k=1, axes = (0,1)), np.rot90(model2, k=2, axes = (0,1)), np.rot90(model2, k=3, axes = (0,1))]
    distances = [np.sum(abs(model1 - i)) for i in configs]

    return min(distances)

def main(argv):
    """ Main entry point of the program """
    pairs = np.genfromtxt('../Data/pairs.csv', dtype=int, delimiter=',')
    distance = np.apply_along_axis(func1d = calhamming, axis = 1, arr = pairs)
    output = np.c_[pairs, distance]
    output = output.astype(int)
    np.savetxt("../Data/HamDis.csv", output, delimiter=",")
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)
    