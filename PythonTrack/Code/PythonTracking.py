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
import statistics
import pandas as pd
import copy

#test data
#img = cv2.imread('../Data/Training/R1D2R2C3_00000.png')
#outname = 'test'

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

def closesttosides(contour):
    Xs = contour[:, 0][:,0]
    Ys = contour[:, 0][:,1]

    p1 = contour[np.argmin(Xs)][0]
    p2 = contour[np.argmin(Ys)][0]
    p3 = contour[np.argmax(Xs)][0]
    p4 = contour[np.argmax(Ys)][0]

    return [p1, p2, p3, p4]

def lrtb(contour):  
    leftmost = contour[contour[:,:,0].argmin()][0]
    rightmost = contour[contour[:,:,0].argmax()][0]
    topmost = contour[contour[:,:,1].argmin()][0]
    bottommost = contour[contour[:,:,1].argmax()][0]
    
    return [leftmost, topmost, rightmost, bottommost]

def convert(vertexes, x, y):
    for i in range(len(vertexes)):
        vertexes[i][0][0] = vertexes[i][0][0] + x
        vertexes[i][0][1] = vertexes[i][0][1] + y

    return vertexes

def findthres(cl1, thres):
    img = copy.deepcopy(cl1)
    img[img > thres] = 255
    img[img < 255] = 0

    #cv2.imwrite('test_' + str(thres) + '.png', img)
    contours = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [blob for blob in contours if cv2.contourArea(blob) > 0 and 2500 > cv2.contourArea(cv2.convexHull(blob)) > 500]
    lefts = [tuple(c[c[:,:,0].argmin()][0]) for c in contours]
    leftcolours = [img[l[1], l[0]+1] for l in lefts]
    white = [contours[i] for i in range(len(contours)) if (leftcolours[i] == 255)]

    return white

def findtags(img):
    #img = cv2.imread("../Data/Training/R1D7R2A1_00000.png")
    #separate out red channel
    R = copy.deepcopy(img[:, :, 2])
    #cv2.imwrite("test_eq.png", R)
    bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)
    
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(R)
    #cv2.imwrite('test_equalised.png', cl1)

    #thresholding
    white = findthres(cl1, 223)
    for t in [95, 111, 143, 175]:
        white2 = findthres(cl1, t)
        [white.append(blob) for blob in white2]

    #redraw contours to make convex
    redraw = [cv2.convexHull(blob) for blob in white]
    cv2.drawContours(bkgd, redraw, -1, (255,0,0), 1) 
    #cv2.imwrite("test_BWredraw.png", bkgd)

    #draw rectangles around the points (tilt)    
    rect = [cv2.minAreaRect(blob) for blob in redraw]
    box = [cv2.boxPoints(pts) for pts in rect]

    # draw rectangles around the rectangle (no tilt)
    checkOverlap = pd.DataFrame([cv2.boundingRect(blob) for blob in box if min(cv2.boundingRect(blob))> 0])
    removeIndex = list()
    for i in range(1, len(checkOverlap.index)):
        checkAgainst = [j for j in checkOverlap.index if j != i]
        for j in checkAgainst:
            if checkOverlap.loc[i, 0] >= checkOverlap.loc[j, 0] and checkOverlap.loc[i, 1] >= checkOverlap.loc[j, 1]:
                if checkOverlap.loc[i, 0] + checkOverlap.loc[i, 2] <= checkOverlap.loc[j, 0] + checkOverlap.loc[j, 2] and checkOverlap.loc[i, 1] + checkOverlap.loc[i, 3] <= checkOverlap.loc[j, 1] + checkOverlap.loc[j, 3]:
                    removeIndex.append(i)
    
    checkOverlap = checkOverlap.drop(set(removeIndex))
    
    #check for rectangles that have significant overlap
    removeIndex2 = list()
    checkOverlap['centroidX'] = checkOverlap[0]+(checkOverlap[2]/2)
    checkOverlap['centroidY'] = checkOverlap[1]+(checkOverlap[3]/2)
    checkOverlap = checkOverlap.sort_values("centroidX", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)
    for i in range(1, len(checkOverlap.index)):
        checkAgainst = [j for j in checkOverlap.index if j != i]
        for j in checkAgainst:
            if math.sqrt((checkOverlap.loc[i, 'centroidX'] - checkOverlap.loc[j, 'centroidX'])**2 + (checkOverlap.loc[i, 'centroidY'] - checkOverlap.loc[j, 'centroidY'])**2) < 10:
                if abs(checkOverlap.loc[i, 2]/checkOverlap.loc[i, 3]-1) > abs(checkOverlap.loc[j, 2]/checkOverlap.loc[j, 3]-1):
                    removeIndex2.append(i)
    checkOverlap = checkOverlap.drop(set(removeIndex2))
    checkOverlap = checkOverlap.drop(['centroidX', 'centroidY'], axis = 1)

    #check for shape
    Coordinates = list()
    CroppedTags = list()
    for i in checkOverlap.index:
        pts = tuple(checkOverlap.loc[i])
        #print(pts[2]/pts[3])
        if 30 < max(pts[2], pts[3]) < 100 and min(pts[2], pts[3]) > 20 and 0.4 < pts[2]/pts[3] < 2.5:
            cropped = img[(pts[1]-2):(pts[1]+pts[3]+2), (pts[0]-2):(pts[0]+pts[2]+2)]
            if min(cropped.shape) > 0:
                cv2.rectangle(bkgd, (pts[0],pts[1]), (pts[0]+pts[2], pts[1]+pts[3]), (0,0,255), 1)
                Coordinates.append((pts[0], pts[1], pts[2], pts[3]))   
    cv2.imwrite("test_foundtags.png", bkgd)

    return [Coordinates, bkgd]

def scoretag(TAG, taglist):
    #ID tag
    configs = [TAG, np.rot90(TAG, k=1, axes = (0, 1)), np.rot90(TAG, k=2, axes = (0, 1)), np.rot90(TAG, k=3)]
    difference = []
    direction = []
    models = [drawmodel(id) for id in taglist]
    
    for m in models:
        diff = [np.sum(abs(m - config))/255 for config in configs]
        difference.append(min(diff))
        direction.append(diff.index(min(diff)))

    return [min(difference), direction[difference.index(min(difference))], taglist[difference.index(min(difference))]]

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

def drawtag(pts, bkgd, img, taglist):
    cropped = img[(pts[1]-2):(pts[1]+pts[3]+2), (pts[0]-2):(pts[0]+pts[2]+2)]
    #cv2.imwrite(outname + "_area_" + str(a) + ".png", cropped)
    grey = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    #convert to black and white with Otsu's thresholding
    #grey = cv2.bilateralFilter(grey,9,75,75)
    
    blur = cv2.blur(grey,(5,5))
    if np.max(blur)-np.min(blur) < 50:
        print('not tag')
        return [bkgd, ['not tag', 'not tag', 'not tag', 'not tag', 'not tag', 'not tag']]        
    
    #bordervalue = (np.sum(bw[0, :]) + np.sum(bw[1:bw.shape[0]-1, 0]) + np.sum(bw[bw.shape[0]-1, :]) + np.sum(bw[1:bw.shape[0]-1, bw.shape[1]-1]))/255
    #borderlen = len(bw[0, :]) + len(bw[1:bw.shape[0]-1, 0]) + len(bw[bw.shape[0]-1, :]) + len(bw[1:bw.shape[0]-1, bw.shape[1]-1])
    
    #if bordervalue/borderlen > 0.1 and (np.sum(bw)/255)/(bw.shape[0]*bw.shape[1]) < 0.5:
    #    centroidX = statistics.mean([vertexes[0][0][0], vertexes[1][0][0], vertexes[2][0][0], vertexes[3][0][0]]) + pts[0]
    #    centroidY = statistics.mean([vertexes[0][0][1], vertexes[1][0][1], vertexes[2][0][1], vertexes[3][0][1]]) + pts[1]
    #    cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,255),2)
    #    print('cannot read')
    #    return [bkgd, ['X', centroidX, centroidY, 'dir', 'X', results[0]]]

    #cv2.imwrite(outname + "_area_" + str(a) + ".png", cropped)
    #cv2.imwrite(outname + "_bw_" + str(a) + ".png", bw)

    #convert to black and white with Otsu's thresholding
    bw = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]    
    #transform polygon (polygon should just be the tag)
    epsilon = 0.01*cv2.arcLength(largest,True)
    approx = cv2.approxPolyDP(largest,epsilon,True)
    #polygon = cv2.drawContours(cropped, [approx], -1, (0,255,0), 1, cv2.LINE_AA)
    #cv2.imwrite(outname + "_approx_" + str(a) + ".png", polygon)
#trial 1
    print('first try')
    vertexes = extremepoints(approx)  
    edges = [math.sqrt((vertexes[p-1][0]-vertexes[p][0])**2 + (vertexes[p-1][1]-vertexes[p][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))

    if edge == 0:
        vertexes = closesttosides(approx)  
        edges = [math.sqrt((vertexes[p-1][0]-vertexes[p][0])**2 + (vertexes[p-1][1]-vertexes[p][1])**2) for p in range(len(vertexes))]
        edge = math.floor(min(edges))

    if edge == 0:
        print('not tag')
        return [bkgd, ['not tag', 'not tag', 'not tag', 'not tag', 'not tag', 'not tag']]   
    
    OneCM = edge/0.3
    rows,cols = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY).shape
    pts1 = np.float32([vertexes[0],vertexes[2],vertexes[3]])
    pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY),M,(cols,rows))
    tag = dst[0:edge, 0:edge]   

    #draw tag
    bwtag = cv2.threshold(tag,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #cv2.imwrite(outname + "_bwtag_" + str(a) + ".png", bwtag)

    #enlarge
    bwtag2 = cv2.resize(bwtag, (bwtag.shape[0]*6, bwtag.shape[1]*6))
    #cv2.imwrite(outname + "_bwtag2_" + str(a) + ".png", bwtag2)

    TAG1 = np.full((6, 6), 255)
    for i in range(6):
        for j in range(6):
            TAG1[i,j] = np.mean(bwtag2[bwtag.shape[1]*i:bwtag.shape[1]*(i+1), bwtag.shape[0]*j:bwtag.shape[0]*(j+1)])
    
    #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG1)
    results = scoretag(TAG1, taglist) # score, dir, id
    if results[0] < 2:
        centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
        centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
        cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(bkgd,(centroidX,centroidY), 5, (255,255,0), -1)
        print('Found it! ID: ' + str(results[2]))
        return [bkgd, [results[2], centroidX, centroidY, results[1], OneCM, results[0]]] 
#trial 2
    print('trial 2')
    #check for misalignment, if found, correct
    kernel = np.ones((3,3),np.uint8)
    close = cv2.morphologyEx(bwtag, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite(outname + "_close_" + str(a) + ".png", close)
    
    #correct for margin
    contours = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    allcontours = [contours[i][j] for i in range(len(contours)) for j in range(len(contours[i]))]
    if len(allcontours) == 0:
        centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
        centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,255),2)
        print('cannot read')
        return [bkgd, ['X', centroidX, centroidY, 'dir', 'X', 'X']] 
    noedges = np.array([allcontours[i] for i in range(len(allcontours)) if allcontours[i][0][0] != 0 and allcontours[i][0][0] != close.shape[0]-1 and allcontours[i][0][1] != 0 and allcontours[i][0][1] != close.shape[0]-1 ])
    if len(noedges) > 0:
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
        
        TAG3 = np.full((6, 6), 255)
        for i in range(6):
            for j in range(6):
                TAG3[i,j] = np.mean(CLOSE[close.shape[1]*i:close.shape[1]*(i+1), close.shape[0]*j:close.shape[0]*(j+1)])

        #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG1)
        results = scoretag(TAG3, taglist) # score, dir, id
        if results[0] < 1:
            centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
            centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
            cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
            cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
            cv2.circle(bkgd,(centroidX,centroidY), 5, (255,255,0), -1)
            print('Found it! ID: ' + str(results[2]))
            return [bkgd, [results[2], centroidX, centroidY, results[1], OneCM, results[0]]]
#trial 3
    print("third time's the charm?")
    #check for misalignment, if found, add corrected tags
    kernel = np.ones((3,3),np.uint8)
    close = cv2.morphologyEx(bwtag, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite(outname + "_close_" + str(a) + ".png", close)
    
    bw2 = np.full((close.shape[0]+20, close.shape[1]+20), 255, dtype = 'uint8')
    bw2[10:close.shape[0]+10, 10:close.shape[1]+10] = close
    
    contours = cv2.findContours(bw2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    areas = [cv2.contourArea(blob) for blob in contours]
    contours.pop(areas.index(max(areas)))
    if len(contours) == 0:
        print('not tag')
        return [bkgd, ['not tag', 'not tag', 'not tag', 'not tag', 'not tag', 'not tag']]  
    else:
        print(len(contours))
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
        #CORNERS = [corners, angle]
    
    ###check and correct for scaling and margin
    contours = cv2.findContours(bw2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    areas = [cv2.contourArea(blob) for blob in contours]
    if len(areas) == 0:
        centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
        centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,255),2)
        print('cannot read')
        return [bkgd, ['X', centroidX, centroidY, 'dir', 'X', 'X']]
    contours.pop(areas.index(max(areas)))
    if len(contours) == 0:
        centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
        centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,255),2)
        print('cannot read')
        return [bkgd, ['X', centroidX, centroidY, 'dir', 'X', 'X']] 
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
            TAG4[i,j] = np.mean(TAGadj[tag2.shape[1]*i:tag2.shape[1]*(i+1), tag2.shape[0]*j:tag2.shape[0]*(j+1)])
    
    #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG1)
    results = scoretag(TAG4, taglist) # score, dir, id
    if results[0] < 1:
        centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
        centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
        cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(bkgd,(centroidX,centroidY), 5, (255,255,0), -1)
        print('Found it! ID: ' + str(results[2]))
        return [bkgd, [results[2], centroidX, centroidY, results[1], OneCM, results[0]]]

#done, still cannot read
    centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
    centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
    cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,255),2)
    print('cannot read')
    return [bkgd, ['X', centroidX, centroidY, 'dir', 'X', 'X']]

def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
        base = os.path.splitext(os.path.basename(filename))[0]
        outname = '../Results/' + base
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/Replicate1/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/Replicate1/Data')]
        filename = files[int(iter)-1]
        base = os.path.splitext(os.path.basename(filename))[0]
        outname = base
    print (filename)

    #filename = '../Data/R1D7R2A1_trimmed.MP4'
    container = av.open(filename)

    if base[6] == 'A':
        taglist = [237,74,121,137,151,180,181,220,222,311,312,341,402,421,427,456,467,596,626,645,664,681,696,697,765,781,794,862,985,1077,1419,1846,1947,1966,2908,2915]
    elif base[6] == 'B':
        taglist = [180,74,121,137,151,181,186,220,222,237,311,312,341,393,421,427,467,534,574,596,626,645,664,681,696,697,765,781,862,985,1077,1419,1846,1947,1966,2908,2915]
    elif base[6] == 'C':
        taglist = [862,121,137,151,180,181,186,220,222,237,341,393,402,421,456,467,534,574,596,626,645,664,681,696,697,765,781,794,985,1077,1419,1846,1947,1966,2908,2915]
    elif base[6] == 'D':
        taglist = [534,74,121,137,151,186,220,222,237,311,312,341,393,402,421,427,456,467,574,596,626,645,664,681,696,697,781,794,862,985,1077,1419,1846,1947,1966,2908]
    wrangled = pd.DataFrame()
    noID = pd.DataFrame()
    f = 0
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outname + "_raw.mp4", fourcc, 20.0, (3840,2160))

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='bgr24')
        
        #img = cv2.imread("../Data/Training/R1D7R2A1_00000.png")
        frameData = pd.DataFrame()
        cannotRead = pd.DataFrame()
        print('Finding tags...')
        Coordinates, bkgd = findtags(img)
        print('Done!')
        print('Reading tags...')
        a = 0
        while a < len(Coordinates):
            print(a)
            bkgd, row = drawtag(Coordinates[a], bkgd, img, taglist) #[results[2], 'centroidX', 'centroidY', results[1], 'OneCM', results[0]]
            if row[0] != 'not tag':
                if row[0] != 'X':
                    completerow = [f] + row
                    frameData = frameData.append([tuple(completerow)], ignore_index=True)
                    #cv2.imwrite(outname + '_' + str(f)  + '_' + str(a) + "_foundtags.png", bkgd)
                    a = a+1
                else:
                    completerow = [f] + row
                    cannotRead = cannotRead.append([tuple(completerow)], ignore_index=True)
                    #cv2.imwrite(outname + '_' + str(f)  + '_' + str(a) + "_foundtags.png", bkgd)
                    a = a+1
            else:
                #cv2.imwrite(outname + '_' + str(f)  + '_' + "a" + "_foundtags.png", bkgd)
                a = a+1
        frameData = frameData.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm', 6:'score'})
        cannotRead = cannotRead.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm', 6:'score'})
        if len(frameData.index) > 1:
            #if there is more than one match for an ID
            if len(set(frameData.ID)) < len(frameData.ID):
                doubled = set(frameData.ID[frameData.duplicated(subset='ID')])
                for d in doubled:
                    problem = frameData[frameData.ID == d]
                    keep = problem[problem.score == min(problem.score)]
                    where = problem.loc[keep.index[0]]
                    problem = problem.drop(keep.index[0])
                    frameData = frameData.drop(problem.index)
                    
                    test1 = problem.index[problem.centroidX == where.centroidX]
                    test2 = problem.index[problem.centroidY == where.centroidY]
                    dontAdd = [i for i in problem.index if i in test1 and i in test2]
                    problem = problem.drop(dontAdd)
                    frameData.append(keep, ignore_index=True)
                    
                    cannotRead.append(problem, ignore_index=True)

            wrangled = wrangled.append(frameData, ignore_index=True)
            noID = noID.append(cannotRead, ignore_index=True)
        out.write(bkgd)
        #cv2.imwrite(outname + '_' + str(f) + "_foundtags.png", bkgd)
        print("Finished frame " + str(f))
        f = f+1
    out.release()
    cv2.destroyAllWindows()
    output = wrangled.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm', 6:'score'})
    output.to_csv(path_or_buf = outname + "_raw.csv", na_rep = "NA", index = False)
    output2 = noID.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm', 6:'score'})
    output2.to_csv(path_or_buf = outname + "_noID.csv", na_rep = "NA", index = False)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)

    #4, 7, 19, [23], 26!, 29, 42, 51, 53, 60, 61, 67, 80, 85, 87, [88], 89, 91