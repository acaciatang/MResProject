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
import PythonTracking

filename = '../Data/Training/R1D2R2C1_00000.png'
equalised = cv2.imread('../Data/Training/R1D2R2C1_00000.png')
img = cv2.imread('../Data/Training/R1D2R2C1_00000.png')
outname = 'test'
potentialTags = PythonTracking.findtags(img, outname)[0]
bkgd = PythonTracking.findtags(img, outname)[1]

if filename[-12] == 'A':
    taglist = [237,74,121,137,151,180,181,220,222,311,312,341,402,421,427,456,467,596,626,645,664,681,696,697,765,781,794,862,985,1077,1419,1846,1947,1966,2908,2915]
elif filename[-12] == 'B':
    taglist = [180,74,121,137,151,181,186,220,222,237,311,312,341,393,421,427,467,534,574,596,626,645,664,681,696,697,765,781,862,985,1077,1419,1846,1947,1966,2908,2915]
elif filename[-12] == 'C':
    taglist = [862,121,137,151,180,181,186,220,222,237,341,393,402,421,456,467,534,574,596,626,645,664,681,696,697,765,781,794,985,1077,1419,1846,1947,1966,2908,2915]
elif filename[-12] == 'D':
    taglist = [534,74,121,137,151,186,220,222,237,311,312,341,393,402,421,427,456,467,574,596,626,645,664,681,696,697,781,794,862,985,1077,1419,1846,1947,1966,2908]

models = [drawmodel(id) for id in taglist]
for i in range(3):
    equalised[:, :, i] = cv2.equalizeHist(equalised[:, :, i])

a = 0
for pts in potentialTags:
    print(a)
    #pts = potentialTags[a]
    #R = img[:,:,2]
    #crop out potential tag region
    cropped = equalised[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
    #cv2.imwrite(outname + "_cropped_" + str(a) + ".png", cropped)
    
    grey = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    
    #convert to black and white with Otsu's thresholding
    ret2,bw = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.medianBlur(bw,3)
    #cv2.imwrite(outname + "_bw_" + str(a) + ".png", bw)
    #cv2.imwrite(outname + "_blur_" + str(a) + ".png", blur)
    
    contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]    
    testarea = cv2.contourArea(largest)
    #transform polygon (polygon should just be the tag)
    epsilon = 0.01*cv2.arcLength(largest,True)
    approx = cv2.approxPolyDP(largest,epsilon,True)
    polygon = cv2.drawContours(cropped, [approx], -1, (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(outname + "_approx_" + str(a) + ".png", polygon)

    #first try
    vertexes = extremepoints(approx)  
    edges = [math.sqrt((vertexes[p-1][0]-vertexes[p][0])**2 + (vertexes[p-1][1]-vertexes[p][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))

    OneCM = edge/0.3
    rows,cols = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY).shape
    pts1 = np.float32([vertexes[0],vertexes[2],vertexes[3]])
    pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY),M,(cols,rows))
    tag = dst[0:edge, 0:edge]   

    #draw tag
    thres,bwtag = cv2.threshold(tag,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(outname + "_bwtag_" + str(a) + ".png", bwtag)

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
    results = scoretag(TAG1, models, taglist) # score, dir, id
    if results[0] < 2:
        print('first try worked!')
        print([a] + results)
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
        cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(bkgd,(447,63), 1, (255,0,0), -1)
        a = a+1
        continue
        #return [frameNum, results[2], centroidX, centroidY, results[1], OneCM, results[0]]

    #second try
    print('second try')
    vertexes = closesttosides(approx)  
    edges = [math.sqrt((vertexes[p-1][0][0]-vertexes[p][0][0])**2 + (vertexes[p-1][0][1]-vertexes[p][0][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))

    if edge == 0:
        
        a = a+1
    else:
        OneCM = edge/0.3
        rows,cols = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY).shape
        pts1 = np.float32([vertexes[0],vertexes[2],vertexes[3]])
        pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY),M,(cols,rows))
        tag = dst[0:edge, 0:edge]     

        #draw tag
        thres,bwtag = cv2.threshold(tag,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(outname + "_bwtag_" + str(a) + ".png", bwtag)

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
        results = scoretag(TAG1, models, taglist) # score, dir, id
        if results[0] < 2:
            print([a] + results)
            cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
            cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
            cv2.circle(bkgd,(447,63), 1, (255,0,0), -1)
            a = a+1
            continue
            #return [frameNum, results[2], centroidX, centroidY, results[1], OneCM, results[0]]

cv2.imwrite(outname + "_tags.png", bkgd)



def drawtag(pts, equalised, outname, a):
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

