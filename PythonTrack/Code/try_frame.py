import sys
import av
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from matching.games import HospitalResident
import pandas as pd
import copy

#test data
img = cv2.imread('../Data/Training/R1D2R2C3_00000.png')
outname = 'test'

def caldis(pt, corner):
    dis = math.sqrt((pt[0][0]-corner[0])**2 + (pt[0][1]-corner[1])**2)
    return dis

#change to distance to corners
def extremepoints(contour):
    Xs = contour[:, 0][:,0]
    Ys = contour[:, 0][:,1]

    minX = min(Xs)
    maxX = max(Xs)
    minY = min(Ys)
    maxY = max(Ys)

    dis1 = [caldis(pt, [minX, minY]) for pt in contour]
    dis2 = [caldis(pt, [maxX, minY]) for pt in contour]
    dis3 = [caldis(pt, [maxX, maxY]) for pt in contour]
    dis4 = [caldis(pt, [minX, maxY]) for pt in contour]
    
    p1 = contour[dis1.index(min(dis1))][0]
    p2 = contour[dis2.index(min(dis2))][0]
    p3 = contour[dis3.index(min(dis3))][0]
    p4 = contour[dis4.index(min(dis4))][0]

    return [p1, p2, p3, p4]

img = cv2.imread('../Data/Training/R1D2R2C3_00000.png')
outname = 'test'

R = img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
#cv2.imwrite(outname + "_eq.png", R)
bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)

bw = cv2.adaptiveThreshold(R,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#cv2.imwrite(outname + "_bw.png", bw)

blur = cv2.medianBlur(bw,5)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
cv2.imwrite(outname + "_opening.png", opening)

# find blobs
contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = [blob for blob in contours if 2000 > cv2.contourArea(blob) > 300]

drawcontours = cv2.drawContours(bkgd, contours, -1, (0,255,255), 1) # all closed contours plotted in yellow
cv2.imwrite(outname + "_BWcontour.png", drawcontours)

#redraw contours to make convex
redraw = [cv2.convexHull(blob) for blob in contours]
#drawredraw = cv2.drawContours(img, redraw, -1, (0,255,0), 1) 
#cv2.imwrite(outname + "_BWredraw.png", drawredraw)

# filter for blobs of right size based on extreme points
rightsize = []
for blob in redraw:
    points = extremepoints(blob)
    distances1 = [math.sqrt((points[p-1][0]-points[p][0])**2 + (points[p-1][1]-points[p][1])**2) for p in range(len(points))]
    distances2 = [math.sqrt((blob[p-1][0][0]-blob[p][0][0])**2 + (blob[p-1][0][1]-blob[p][0][1])**2) for p in range(len(blob))]
    distances = distances1 + distances2
    maxdist = max(distances)
    if 25 < maxdist < 100:
        rightsize.append(blob)
drawrightsize = cv2.drawContours(img, rightsize, -1, (0,255,0), 1) # contours of right size plotted in blue
cv2.imwrite(outname + "_BWrightsize.png", drawrightsize)

#draw rectangles around the points (tilt)    
rect = [cv2.minAreaRect(blob) for blob in rightsize]
box = [cv2.boxPoints(pts) for pts in rect]

# draw rectangles around the rectangle (no tilt)
rect2 = [cv2.boundingRect(blob) for blob in box if min(cv2.boundingRect(blob))> 0]
fillrect = bkgd
potentialTags = []
for pts in rect2:
    roi = cv2.medianBlur(R,5)[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
    if np.max(roi) - np.min(roi) > 50 and np.max(roi) > 113:
        fillrect = cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,0,255),3)
        fillrect = bkgd
        potentialTags.append(pts)
cv2.imwrite(outname + "_BWfillrect.png", fillrect)
bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)