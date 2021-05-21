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

    p1 = contour[np.argmin(Xs)]
    p2 = contour[np.argmin(Ys)]
    p3 = contour[np.argmax(Xs)]
    p4 = contour[np.argmax(Ys)]

    return [p1, p2, p3, p4]

def scoretag(TAG, models, taglist):
    #ID tag
    configs = [TAG, np.rot90(TAG, k=1, axes = (0, 1)), np.rot90(TAG, k=2, axes = (0, 1)), np.rot90(TAG, k=3)]
    difference = []
    direction = []
    
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

    if filename[6] == 'A':
        taglist = [237,74,121,137,151,180,181,220,222,311,312,341,402,421,427,456,467,596,626,645,664,681,696,697,765,781,794,862,985,1077,1419,1846,1947,1966,2908,2915]
    elif filename[6] == 'B':
        taglist = [180,74,121,137,151,181,186,220,222,237,311,312,341,393,421,427,467,534,574,596,626,645,664,681,696,697,765,781,862,985,1077,1419,1846,1947,1966,2908,2915]
    elif filename[6] == 'C':
        taglist = [862,121,137,151,180,181,186,220,222,237,341,393,402,421,456,467,534,574,596,626,645,664,681,696,697,765,781,794,985,1077,1419,1846,1947,1966,2908,2915]
    elif filename[6] == 'D':
        taglist = [534,74,121,137,151,186,220,222,237,311,312,341,393,402,421,427,456,467,574,596,626,645,664,681,696,697,781,794,862,985,1077,1419,1846,1947,1966,2908]
    models = [drawmodel(id) for id in taglist]

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