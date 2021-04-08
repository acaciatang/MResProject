#!/usr/bin/env python3

"""Seperates video into pngs of frames."""

__appname__ = 'RemoveBackground.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import av
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#testdata
#filename = 'P1000073.MP4'
#image = 'D6_1.png'

def makepng(filename):
    container = av.open(filename)
    outname = os.path.splitext(os.path.basename(filename))[0]

    for frame in container.decode(video=0):
        print("Printed frame " + str('%06d' % frame.index) + "!")
        frame.to_image().save(outname + '_%06d.png' % frame.index)
        i = frame.index
    return i

def extremepoints(contour):
    leftmost = contour[contour[:,:,0].argmin()][0]
    rightmost = contour[contour[:,:,0].argmax()][0]
    topmost = contour[contour[:,:,1].argmin()][0]
    bottommost = contour[contour[:,:,1].argmax()][0]
    return [leftmost, rightmost, topmost, bottommost]

def distance(points):
    distances = [math.sqrt((points[p-1][0]-points[p][0])**2 + (points[p-1][1]-points[p][1])**2) for p in points]

def findtags(image):
    #load image and get name of output file
    img = cv2.imread(image, 1)
    outname = os.path.splitext(os.path.basename(image))[0]

    # separate out colour channels of interest: green (G) for detecting tags, red (R) for recognizing tags
    G = img[:, :, 1]
    #cv2.imwrite(outname + "_G.png", G)
    R = img[:, :, 2]
    #cv2.imwrite(outname + "_R.png", R)
    bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)

    #convert to black and white with Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(G,(5,5),0)
    ret3,bw = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    cv2.imwrite(outname + "_BW.png", bw)

    #make blobs more blobby
    kernel = np.ones((6,6),np.uint8)
    close = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(outname + "_close.png", close)
    
    # find blobs
    contours, hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [blob for blob in contours if cv2.contourArea(blob) > 0]
    drawcontours = cv2.drawContours(bkgd, contours, -1, (0,255,255), 1) # all closed contours plotted in yellow
    cv2.imwrite(outname + "_BWcontour.png", drawcontours)

    # filter for blobs of right size based on extreme points
    rightsize = []
    for blob in contours:
        points = extremepoints(blob)
        distances = [math.sqrt((points[p-1][0]-points[p][0])**2 + (points[p-1][1]-points[p][1])**2) for p in range(len(points))]
        maxdist = max(distances)
        mindist = min(distances)
        if 25 < maxdist < 125 and 10 < mindist:
            rightsize.append(blob)
    drawrightsize = cv2.drawContours(bkgd, rightsize, -1, (255,0,0), 1) # contours of right size plotted in blue
    cv2.imwrite(outname + "_BWrightsize.png", drawrightsize)

    # redraw contours to make convex
    redraw = [cv2.convexHull(i) for i in rightsize]
    drawredraw = cv2.drawContours(bkgd, redraw, -1, (0,255,0), 1) 
    cv2.imwrite(outname + "_BWredraw.png", drawredraw)

    #draw rectangles around the points (tilt)    
    rect = [cv2.minAreaRect(blob) for blob in redraw]
    box = [cv2.boxPoints(rectangles) for rectangles in rect]
    # draw rectangles around the rectangle (no tilt)
    rect2 = [cv2.boundingRect(blob) for blob in box]
    for pts in rect2:
        fillrect = cv2.rectangle(bkgd,(pts[0]-1,pts[1]-1),(pts[0]-1 + pts[2]+2, pts[1]-1 + pts[3]+2),(0,0,255),-1)
        fillrect = bkgd
    cv2.imwrite(outname + "_BWfillrect.png", fillrect)

    # crop out each rectangle from red channel
    mask = cv2.inRange(fillrect, np.array([0,0,255]), np.array([0,0,255]))
    bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)
    cropped = cv2.bitwise_and(R,R, mask= mask)
    cv2.imwrite(outname + "_BWcropped.png", cropped)

    # convert to black and white
    bw2 = cv2.adaptiveThreshold(cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    cv2.imwrite(outname + "_BW2.png", bw2)

    return rect2

def IDTags(pts):
#for a in range(len(rect)):
#    pts = rect[a]
    #crop out potential tag region
    cropped = R[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
    croppedbkgd = cv2.cvtColor(cropped,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(outname + "_cropped" + str(a) + ".png", cropped)
    bw = cv2.adaptiveThreshold (cropped,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    cv2.imwrite(outname + "_bw" + str(a) + ".png", bw)

    contrast = cropped//32
    contrast = contrast * 32
    cv2.imwrite(outname + "_contrast" + str(a) + ".png", contrast)
    
    th = 1 + (int(np.max(contrast)) + int(np.min(contrast)))/2
    ret,th1 = cv2.threshold(contrast,max(th, 64),255,cv2.THRESH_BINARY)
    cv2.imwrite(outname + "_bw" + str(a) + ".png", th1)

    contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]
    
    leftmost = largest[largest[:,:,0].argmin()][0]
    rightmost = largest[largest[:,:,0].argmax()][0]
    topmost = largest[largest[:,:,1].argmin()][0]
    bottommost = largest[largest[:,:,1].argmax()][0]
    
    points = [leftmost, rightmost, topmost, bottommost]
    POIs = [int(not (pts[2]-1 in p or 0 in p or pts[3]-1 in p)) for p in points]
    return tagData  

def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]
    print (filename)
    makepng(filename)

    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)