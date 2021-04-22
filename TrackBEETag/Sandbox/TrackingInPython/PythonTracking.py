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
from matching.games import HospitalResident
import pandas as pd
import copy

def makepng(filename, outname):
    container = av.open(filename)

    for frame in container.decode(video=0):
        print("Printed frame " + str('%06d' % frame.index) + "!")
        frame.to_image().save(outname + '_%06d.png' % frame.index)
        i = frame.index
    return i

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

def lrtb(contour):  
    leftmost = contour[contour[:,:,0].argmin()][0]
    rightmost = contour[contour[:,:,0].argmax()][0]
    topmost = contour[contour[:,:,1].argmin()][0]
    bottommost = contour[contour[:,:,1].argmax()][0]
    
    return [leftmost, topmost, rightmost, bottommost]

def closesttosides(contour):
    Xs = contour[:, 0][:,0]
    Ys = contour[:, 0][:,1]

    p1 = contour[np.argmin(Xs)]
    p2 = contour[np.argmin(Ys)]
    p3 = contour[np.argmax(Xs)]
    p4 = contour[np.argmax(Ys)]

    return [p1, p2, p3, p4]

def convert(vertexes, x, y):
    for i in range(len(vertexes)):
        vertexes[i][0][0] = vertexes[i][0][0] + x
        vertexes[i][0][1] = vertexes[i][0][1] + y

    return vertexes

def findtags(img, outname):
    #load image and get name of output file
    #img = cv2.imread(image, 1)
    #outname = os.path.splitext(os.path.basename(image))[0]

    # separate out colour channels of interest: green (G) for detecting tags, red (R) for recognizing tags
    G = img[:, :, 1]
    #cv2.imwrite(outname + "_G.png", G)
    R = img[:, :, 2]
    #cv2.imwrite(outname + "_R.png", R)
    bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)

    #convert to black and white with Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(G,(5,5),0)
    ret,toodark = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2,bw =  cv2.threshold(blur,(ret-2),255,cv2.THRESH_BINARY)
    #cv2.imwrite(outname + "_BW.png", bw)

    #make blobs more blobby
    kernel = np.ones((5,5),np.uint8)
    close = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite(outname + "_close.png", close)
    
    # find blobs
    #contours, hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [blob for blob in contours if cv2.contourArea(blob) > 0]
    #drawcontours = cv2.drawContours(bkgd, contours, -1, (0,255,255), 1) # all closed contours plotted in yellow
    #cv2.imwrite(outname + "_BWcontour.png", drawcontours)

    # redraw contours to make convex
    redraw = [cv2.convexHull(blob) for blob in contours]
    #drawredraw = cv2.drawContours(bkgd, redraw, -1, (0,255,0), 1) 
    #cv2.imwrite(outname + "_BWredraw.png", drawredraw)

    # make triangles into squares
    notri = []
    for blob in redraw:
        if len(blob) == 3:
            Xs = blob[:, 0][:,0]
            Ys = blob[:, 0][:,1]

            minX = min(Xs)
            maxX = max(Xs)
            minY = min(Ys)
            maxY = max(Ys)

            p1 = np.array([[minX, minY]])
            p2 = np.array([[maxX, minY]])
            p3 = np.array([[maxX, maxY]])
            p4 = np.array([[minX, maxY]])
            
            notri.append(np.array([p1, p2, p3, p4]))
        else:
            notri.append(blob)

    # filter for blobs of right size based on extreme points
    rightsize = []
    for blob in notri:
        points = extremepoints(blob)
        distances1 = [math.sqrt((points[p-1][0]-points[p][0])**2 + (points[p-1][1]-points[p][1])**2) for p in range(len(points))]
        distances2 = [math.sqrt((blob[p-1][0][0]-blob[p][0][0])**2 + (blob[p-1][0][1]-blob[p][0][1])**2) for p in range(len(blob))]
        distances = distances1 + distances2
        maxdist = max(distances)
        mindist = min(distances)
        if 25 < maxdist < 250:
            rightsize.append(blob)
    #drawrightsize = cv2.drawContours(bkgd, rightsize, -1, (255,0,0), 1) # contours of right size plotted in blue
    #cv2.imwrite(outname + "_BWrightsize.png", drawrightsize)

    #draw rectangles around the points (tilt)    
    rect = [cv2.minAreaRect(blob) for blob in rightsize]
    box = [cv2.boxPoints(pts) for pts in rect]
    
    # draw rectangles around the rectangle (no tilt)
    rect2 = [cv2.boundingRect(blob) for blob in box if min(cv2.boundingRect(blob))> 0]
    fillrect = bkgd
    potentialTags = []
    for pts in rect2:
        roi = R[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
        #if np.min(roi) < 113 and min(pts[2],pts[3]) > 25 and np.max(roi) > 113:
        fillrect = cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,0,255),-1)
        fillrect = bkgd
        potentialTags.append(pts)
    #cv2.imwrite(outname + "_BWfillrect.png", fillrect)

    # crop out each rectangle from red channel
    #mask = cv2.inRange(fillrect, np.array([0,0,255]), np.array([0,0,255]))
    #cropped = cv2.bitwise_and(img,img, mask= mask)
    #output = cv2.add(cropped,cv2.cvtColor(R,cv2.COLOR_GRAY2RGB))
    #cv2.imwrite(outname + "_potentialTags.png", output)

    return potentialTags

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

    model = np.rot90(model)*255 # I don't know why the matlab cod is like this but it is
    return model.astype(int)

def drawtag(pts, R, outname, a):
    #pts = potentialTags[a]
    #R = img[:,:,2]
    #crop out potential tag region
    cropped = R[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
    croppedbkgd = cv2.cvtColor(cropped,cv2.COLOR_GRAY2RGB)
    #cv2.imwrite(outname + "_cropped" + str(a) + ".png", cropped)
    bw = cv2.adaptiveThreshold (cropped,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    
    #convert to black and white with Otsu's thresholding
    ret2,bw = cv2.threshold(cropped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    #cv2.imwrite(outname + "_bw" + str(a) + ".png", bw)

    contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]    
    points = lrtb(largest)
    POIs = [int(not (pts[2]-1 in p or 0 in p or pts[3]-1 in p)) for p in points]
    
    if sum(POIs) < 2:
        return None
    
    # get threshold within largest blob
    croppedblur = cv2.GaussianBlur(cropped,(5,5),0)
    bw_2 = cv2.cvtColor(bw,cv2.COLOR_GRAY2RGB)
    croppedblur_2 = cv2.cvtColor(croppedblur,cv2.COLOR_GRAY2RGB)
    mask = cv2.inRange(bw_2, np.array([255,255,255]), np.array([255,255,255]))
    masked = cv2.bitwise_and(croppedblur_2,croppedblur_2, mask= mask)
    #cv2.imwrite(outname + "_mask.png", masked)

    line = np.array([masked[i, j][0] for i in range(masked.shape[0]) for j in range(masked.shape[1]) if masked[i, j][0] != 0])
    ret,bw = cv2.threshold(line,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2, bw2 = cv2.threshold(cropped,(ret-16),255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(bw2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]    
    testarea = cv2.contourArea(largest)
    #transform polygon (polygon should just be the tag)
    epsilon = 0.01*cv2.arcLength(largest,True)
    approx = cv2.approxPolyDP(largest,epsilon,True)
    #polygon = cv2.drawContours(croppedbkgd, [approx], -1, (0,255,0), 1, cv2.LINE_AA)
    #cv2.imwrite(outname + "_approx_" + str(a) + ".png", polygon)

    vertexes = closesttosides(approx)
    test = np.array(vertexes)

    if len(np.unique(test, axis = 0)) < 4:
        return None
    
    edges = [math.sqrt((vertexes[p-1][0][0]-vertexes[p][0][0])**2 + (vertexes[p-1][0][1]-vertexes[p][0][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))

    OneCM = edge/0.3
    rows,cols = cropped.shape
    pts1 = np.float32([vertexes[0],vertexes[2],vertexes[3]])
    pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(cropped,M,(cols,rows))
    tag = dst[0:edge, 0:edge]   

    slopes = [(vertexes[i][0][1]-vertexes[i-1][0][1])/(vertexes[i][0][0]-vertexes[i-1][0][0])for i in range(4)]
    diff = [abs(slopes[i-2] - slopes[i])/((slopes[i-2] + slopes[i])/2) for i in range(2)]
    for d in range(2):
        if diff[d] > 0.1:
            lengths = [math.sqrt((vertexes[i][0][0]-vertexes[(i+1)%4][0][0])**2 + (vertexes[i][0][1]-vertexes[(i+1)%4][0][1])**2) for i in [d, d+2]]
            if lengths[0] > lengths[1]:
                short = d+2
            else:
                short = d
            change1 = copy.deepcopy(vertexes)
            change1[short][0][1] = change1[short-1][0][1] - change1[short-2][0][1] + change1[(short+1)%4][0][1]
            change1[short][0][0] = change1[short-1][0][0] - change1[short-2][0][0] + change1[(short+1)%4][0][0]
            change2 = copy.deepcopy(vertexes)
            s = short + 1
            change2[s][0][1] = change2[s-1][0][1] - change2[s-2][0][1] + change2[(s+1)%4][0][1]
            change2[s][0][0] = change2[s-1][0][0] - change2[s-2][0][0] + change2[(s+1)%4][0][0]
            
            average = []
            tags = []
            for v in [change1, change2]:
                edges = [math.sqrt((v[p-1][0][0]-v[p][0][0])**2 + (v[p-1][0][1]-v[p][0][1])**2) for p in range(len(v))]
                edge = math.floor(min(edges))

                OneCM = edge/0.3
                rows,cols = cropped.shape
                pts1 = np.float32([v[0],v[2],v[3]])
                pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
                M = cv2.getAffineTransform(pts1,pts2)
                dst = cv2.warpAffine(cropped,M,(cols,rows))
                tag = dst[0:edge, 0:edge]
                average.append(np.mean(tag))
                tags.append(tag)

                tag = tags[average.index(max(average))]

    if tag.shape[0] != tag.shape[1]:
        edge = min(tag.shape[0], tag.shape[1]) #was min
        tag = dst[0:edge, 0:edge]
    
    #cv2.imwrite(outname + "_tag_" + str(a) + ".png", tag)
    
    if edge < 6 or edge > 40:
        return None

    #draw tag
    thres,bwtag = cv2.threshold(tag,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imwrite(outname + "_bwtag_" + str(a) + ".png", bwtag)

    #enlarge
    bwtag2 = cv2.resize(bwtag, (bwtag.shape[0]*6, bwtag.shape[1]*6))
    #cv2.imwrite(outname + "_bwtag2_" + str(a) + ".png", bwtag2)

    TAG1 = np.full((6, 6), 255)
    for i in range(6):
        for j in range(6):
            TAG1[i,j] = np.mean(bwtag2[bwtag.shape[0]*i:bwtag.shape[0]*(i+1), bwtag.shape[0]*j:bwtag.shape[0]*(j+1)])
            if TAG1[i, j] < 127:
                TAG1[i,j] = 0
            if i == 0 or i == 5 or j == 0 or j == 5 or TAG1[i, j] > 128:
                TAG1[i,j] = 255
    #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG1)

    TAG2 = np.full((6, 6), 255)
    for i in range(6):
        for j in range(6):
            test = np.mean(bwtag2[edge*i:edge*(i+1), edge*j:edge*(j+1)])
            if test < thres:
                if i == 0 or i == 5 or j == 0 or j == 5:
                    continue
                else:
                    TAG2[i,j] = 0         
    #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG2)

    if np.sum(TAG1) > 8160:
        return None

    corners = copy.deepcopy(vertexes)
    corners = convert(corners, pts[0], pts[1])
    Xs = [corners[i][0][0] for i in range(4)]
    Ys = [corners[i][0][1] for i in range(4)]
    centroidX = sum(Xs)/4
    centroidY = sum(Ys)/4
    
    #check for misalignment, if found, correct
    kernel = np.ones((4,4),np.uint8)
    close = cv2.morphologyEx(bwtag, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite(outname + "_close_" + str(a) + ".png", close)
    
    #correct for margin
    contours, hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    allcontours = [contours[i][j] for i in range(len(contours)) for j in range(len(contours[i]))]
    if len(allcontours) == 0:
        return None
    noedges = np.array([allcontours[i] for i in range(len(allcontours)) if allcontours[i][0][0] != 0 and allcontours[i][0][0] != close.shape[0]-1 and allcontours[i][0][1] != 0 and allcontours[i][0][1] != close.shape[0]-1 ])
    if len(noedges) == 0:
        return None
    left = min(noedges[:,:, 0])[0] + 1
    right = close.shape[0]-max(noedges[:,:, 0])[0]
    top = min(noedges[:,:, 1])[0] + 1
    bottom = close.shape[0]-max(noedges[:,:, 1])[0]

    if left > right:
        close = np.concatenate((close, np.full((close.shape[0], left-right), 255)), axis = 1)
        #tag = np.concatenate((tag, np.full((close.shape[0], left-right), 255)), axis = 1)
    elif right > left:
        close = np.concatenate((np.full((close.shape[0], right-left), 255), close), axis = 1)
        #tag = np.concatenate((np.full((close.shape[0], right-left), 255), tag), axis = 1)

    if top > bottom:
        close = np.concatenate((close, np.full((top-bottom, close.shape[1]), 255)), axis = 0)
        #tag = np.concatenate((tag, np.full((top-bottom, close.shape[1]), 255)), axis = 0)
    elif bottom > top:
        close = np.concatenate((np.full((bottom-top, close.shape[1]), 255), close), axis = 0)
        #tag = np.concatenate((np.full((bottom-top, close.shape[1]), 255), tag), axis = 0)
    
    close = np.array(close, dtype='uint8')
    #cv2.imwrite(outname + "_close_" + str(a) + ".png", close)
    
    #correct for size
    if close.shape[0] > close.shape[1]:
        finalx = close.shape[0] + 6 - (close.shape[0]%6)
        possibleY = [6*i for i in range(int(finalx/6) + 1)]
        diff = [abs((i-close.shape[1])/close.shape[1]) for i in possibleY]
        finaly = possibleY[diff.index(min(diff))]
        close = cv2.resize(close, dsize=(finaly, finalx))
        close = cv2.resize(close, dsize=(finalx, finalx))
    elif close.shape[1] > close.shape[0]:
        finaly = close.shape[1] + 6 - (close.shape[1]%6)
        possibleX = [6*i for i in range(int(finaly/6) + 1)]
        diff = [abs((i-close.shape[0])/close.shape[0]) for i in possibleX]
        finalx = possibleX[diff.index(min(diff))]
        close = cv2.resize(close, dsize=(finaly, finalx))
        close = cv2.resize(close, dsize=(finalx, finalx))

    CLOSE = cv2.resize(close, dsize=(close.shape[0]*6, close.shape[0]*6))
    TAG = cv2.resize(tag, dsize=(close.shape[0]*6, close.shape[0]*6))
    TAG3 = np.full((6, 6), 255)
    for i in range(6):
        for j in range(6):
            TAG3[i,j] = np.mean(CLOSE[close.shape[0]*i:close.shape[0]*(i+1), close.shape[0]*j:close.shape[0]*(j+1)])
            if TAG3[i, j] > 127:
                TAG3[i,j] = 255
            if i == 0 or i == 5 or j == 0 or j == 5:
                TAG3[i,j] = 255        
            elif TAG3[i,j] < 127:
                TAG3[i,j] = 0


    #check for misalignment, if found, add corrected tags
    kernel = np.ones((3,3),np.uint8)
    close = cv2.morphologyEx(bwtag, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite(outname + "_close_" + str(a) + ".png", close)
    
    bw2 = np.full((close.shape[0]+20, close.shape[0]+20), 255, dtype = 'uint8')
    bw2[10:close.shape[0]+10, 10:close.shape[0]+10] = close
    
    contours, hierarchy = cv2.findContours(bw2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(blob) for blob in contours]
    contours.pop(areas.index(max(areas)))
    if len(contours) == 0:
        return None
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]
    #transform polygon (polygon should just be the tag)
    epsilon = 0.02*cv2.arcLength(largest,True)
    approx = cv2.approxPolyDP(largest,epsilon,True)
    #polygon = cv2.drawContours(cv2.cvtColor(bw2,cv2.COLOR_GRAY2RGB), [approx], -1, (0,255,0), 1, cv2.LINE_AA)
    #cv2.imwrite(outname + "_approx_" + str(a) + ".png", polygon)
    
    ###check and correct for tilt
    DISs = [math.sqrt((approx[i][0][0]-approx[i-1][0][0])**2 + (approx[i][0][1]-approx[i-1][0][1])**2) for i in range(len(approx))]
    longest = DISs.index(max(DISs))
    Xdiff = abs(approx[longest][0][0]-approx[longest-1][0][0])
    Ydiff = abs(approx[longest][0][1]-approx[longest-1][0][1])
    if min(Xdiff, Ydiff) > 2:
        angle = math.degrees(math.atan2(approx[longest-1][0][1]-approx[longest][0][1], approx[longest-1][0][0]-approx[longest][0][0]))
        rows,cols = bw2.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(bw2,M,(cols,rows))
        rotated = dst[min(Xdiff, Ydiff)+2:dst.shape[0]-(min(Xdiff, Ydiff)+2), min(Xdiff, Ydiff)+2:dst.shape[0]-(min(Xdiff, Ydiff)+2)]
        thres,bw2 = cv2.threshold(rotated,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imwrite(outname + "_rotated_" + str(a) + ".png", bw2)
        CORNERS = [corners, angle]
    
    ###check and correct for scaling and margin
    contours, hierarchy = cv2.findContours(bw2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(blob) for blob in contours]
    if len(areas) == 0:
        return None
    contours.pop(areas.index(max(areas)))
    if len(contours) == 0:
        return None
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]
    #transform polygon (polygon should just be the tag)
    epsilon = 0.02*cv2.arcLength(largest,True)
    approx = cv2.approxPolyDP(largest,epsilon,True)
    points = np.array(extremepoints(approx))
    Xdis = max(points[:, 0]) - min(points[:,0])
    Ydis = max(points[:, 1]) - min(points[:,1])
    taglength = max(Xdis, Ydis)

    tag2 = bw2[min(points[:, 1]):max(points[:, 1])+1, min(points[:, 0]):max(points[:, 0])+1]
    #correct for size
    if tag2.shape[0] > tag2.shape[1]:
        finalx = close.shape[0] + 4 - (close.shape[0]%4)
        possibleY = [4*i for i in range(1, int(finalx/4) + 1)]
        diff = [abs((i-tag2.shape[1])/tag2.shape[1]) for i in possibleY]
        finaly = possibleY[diff.index(min(diff))]
        tag2 = cv2.resize(tag2, dsize=(finaly, finalx))
        tag2 = cv2.resize(tag2, dsize=(finalx, finalx))
    elif tag2.shape[1] > tag2.shape[0]:
        finaly = tag2.shape[1] + 4 - (tag2.shape[1]%4)
        possibleX = [4*i for i in range(1, int(finaly/4) + 1)]
        diff = [abs((i-tag2.shape[0])/tag2.shape[0]) for i in (possibleX)]
        finalx = possibleX[diff.index(min(diff))]
        tag2 = cv2.resize(tag2, dsize=(finaly, finalx))
        tag2 = cv2.resize(tag2, dsize=(finalx, finalx))

    #cv2.imwrite(outname + "_cropped_" + str(a) + ".png", tag2)
    #add margins
    tag2 = np.concatenate((np.full((int(tag2.shape[0]/4), tag2.shape[1]), 255), tag2, np.full((int(tag2.shape[0]/4), tag2.shape[1]), 255)), axis = 0)
    tag2 = np.concatenate((np.full((tag2.shape[0], int(tag2.shape[1]/4)), 255), tag2, np.full((tag2.shape[0], int(tag2.shape[1]/4)), 255)), axis = 1)
    tag2 = np.array(tag2, dtype='uint8')
    #draw tag
    TAGadj = cv2.resize(tag2, dsize=(tag2.shape[0]*6, tag2.shape[0]*6))
    TAG4 = np.full((6, 6), 255)
    for i in range(6):
        for j in range(6):
            TAG4[i,j] = np.mean(TAGadj[tag2.shape[0]*i:tag2.shape[0]*(i+1), tag2.shape[0]*j:tag2.shape[0]*(j+1)])
    TAG4 = np.array(TAG4, dtype='uint8')
    TAG4 = cv2.adaptiveThreshold(TAG4,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,1)
    #cv2.imwrite(outname + "_bwTAG3_" + str(a) + ".png", TAG3)

    TAGs = [TAG1, TAG2, TAG3, TAG4]
    return [TAGs, corners, centroidX, centroidY, OneCM]

def scoretag(TAG, models):
    #ID tag
    configs = [TAG, np.rot90(TAG, k=1, axes = (0, 1)), np.rot90(TAG, k=2, axes = (0, 1)), np.rot90(TAG, k=3)]
    difference = []
    direction = []
    
    for m in models:
        diff = [np.sum(abs(m - config))/255 for config in configs]
        difference.append(min(diff))
        direction.append(diff.index(min(diff)))

    return [difference, direction]

def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]
    print (filename)

    #filename = 'D6.MP4'
    #filename = 'A1.MP4'
    outname = os.path.splitext(os.path.basename(filename))[0]
    container = av.open(filename)

    if filename[0] == 'A':
        taglist = [68,118,137,173,289,304,325,365,392,420,437,512,559,596,613,666,696,765,862,1112,1150,1203,1492,1730,1966,2091,2327,2452,2511,2932,2992,3067,3261,3360,3415,3486,3570,3757,3908,4015]
    elif filename[0] == 'B':
        taglist = [31,46,69,180,222,270,311,330,347,393,542,598,651,697,792,813,875,1062,1085,1227,1368,1498,1585,1744,1947,1986,2056,2158,2281,2332,2460,2607,2835,2908,2945,3375,3488,3581,3783,3926]
    elif filename[0] == 'C':
        taglist = [52,74,103,209,226,274,312,331,354,427,455,476,502,544,574,601,634,661,707,770,881,1028,1180,1243,1465,1543,1704,1759,1797,1846,1896,2118,2340,2413,2488,2523,2915,2954,3134,3832]
    elif filename[0] == 'D':
        taglist = [59,75,104,135,211,237,324,341,361,377,413,436,456,510,579,609,637,664,681,720,802,844,910,1074,1104,1403,1620,1718,1799,1903,2006,2072,2192,2242,2355,2856,2880,3163,3358,3388]
    elif filename[0] == 'P':
        taglist = [59,68,74,75,103,104,135,137,180,211,274,304,311,312,324,325,330,331,392,393,413,502,544,613,637,651,696,707,792,1104,1112,1465,1543,1759,1846,1903,2056,2856,2945,3163]
    elif filename[0] == 't':
        taglist = [74,121,137,151,162,180,181,186,220,222,237,311,312,341,393,402,421,427,456,467,502,534,574,596,626,645,664,681,696,697,720,765,781,794,824,862,985,1074,1077,1419,1486,1797,1846,1875,1947,1966,2192,2211,2908,2915]
    
    models = [drawmodel(id) for id in taglist]
    wrangled = pd.DataFrame()
    f = 0
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='bgr24')
        #break
        out = outname +  "_" + str(f)

        frameData = pd.DataFrame()
        scores = pd.DataFrame()
        directions = pd.DataFrame()
        potentialTags = findtags(img, out) #potentialTags
        As = list()
        for a in range(len(potentialTags)):
            #raw = drawtag(potentialTags[a], img[:,:,2], out, a) # [[TAG1, TAG2], vertexes, centroidX, centroidY, OneCM]
            raw = drawtag(potentialTags[a], img[:,:,1], out, a)
            if raw == None:
                continue
            frontchoice = [np.array([round((raw[1][i][0][0]+raw[1][(i+1)%4][0][0])/2), round((raw[1][i][0][1]+raw[1][(i+1)%4][0][1])/2)])  for i in range(4)]
            dirchoice = [math.degrees(math.atan2(frontchoice[i][0]-raw[2], raw[3]-frontchoice[i][1])) for i in range(4)]
            row = [str(f), int(a), raw[2], raw[3], None, raw[4], None, None] # [(frameNum, ID, centroidX, centroidY, dir, OneCM, test, score)]

            TAG1score = scoretag(raw[0][0], models)[0]
            TAG1dir = scoretag(raw[0][0], models)[1]
            TAG2score = scoretag(raw[0][1], models)[0]
            TAG2dir = scoretag(raw[0][1], models)[1]
            if len(raw[0]) > 2:
                TAG3score = scoretag(raw[0][2], models)[0]
                TAG3dir = scoretag(raw[0][2], models)[1]
                if len(raw[0]) > 3:
                    TAG4score = scoretag(raw[0][3], models)[0]
                    TAG4dir = scoretag(raw[0][3], models)[1]
                    score = [min(TAG1score[i], TAG2score[i], TAG3score[i], TAG4score[i]) for i in range(len(models))]
                else:
                    score = [min(TAG1score[i], TAG2score[i], TAG3score[i]) for i in range(len(models))]
            else:
                score = [min(TAG1score[i], TAG2score[i]) for i in range(len(models))]

#fix me!
            dir = []
            for i in range(len(models)):
                if TAG1score[i] < TAG2score[i]:
                    dir.append(TAG1dir[i])
                else:
                    dir.append(TAG2dir[i])
            dir = [dirchoice[d] for d in dir]

            if min(score) < 2:
                As.append(a)
                row[1] = str(int(taglist[score.index(min(score))]))
                row[4] = dir[score.index(min(score))]
                row[7] = str(int(min(score)))
                
                frameData = frameData.append([tuple(row)], ignore_index=True)
                print(a)
        frameData[6] = As
        wrangled = wrangled.append(frameData, ignore_index=True)
        print("Finished frame " + str(f))
        f = f+1

    output = wrangled.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm', 6:'test'})
    output.to_csv(path_or_buf = outname + "_raw.csv", na_rep = "NA", index = False)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)