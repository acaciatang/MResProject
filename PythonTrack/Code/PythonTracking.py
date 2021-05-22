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
img = cv2.imread('../Data/Training/R1D2R2C3_00000.png')
outname = 'test'

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

    p1 = contour[np.argmin(Xs)]
    p2 = contour[np.argmin(Ys)]
    p3 = contour[np.argmax(Xs)]
    p4 = contour[np.argmax(Ys)]

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

def findthres(G):
    converted = [[G[G > thres] == 0] for thres in range(255)]
    for thres in range(255):
        G[G > (255-thres)] == 0
        converted.append(G)
    del converted[0:255]
    blurs = [cv2.GaussianBlur(g,(5,5),0) for g in converted]
    rets = [cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0] for blur in blurs]
    bws =  [cv2.threshold(blurs[i],rets[i],255,cv2.THRESH_BINARY)[1] for i in range(len(blurs))]
    average = [np.average(bw) for bw in bws]

    return bws[average.index(max(average))-10]

def findtags(img, outname):
    for i in range(3):
        img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    #separate out red channel
    R = img[:, :, 2]
    #cv2.imwrite(outname + "_eq.png", R)
    bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)

    bw = cv2.adaptiveThreshold(R,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.imwrite(outname + "_bw.png", bw)

    blur = cv2.medianBlur(bw,9)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    #cv2.imwrite(outname + "_opening.png", opening)

    # find blobs
    contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [blob for blob in contours if 2000 > cv2.contourArea(blob) > 500 and cv2.arcLength(blob, True) < 200]
    #drawcontours = cv2.drawContours(bkgd, contours, -1, (0,255,255), 1) # all closed contours plotted in yellow
    #cv2.imwrite(outname + "_BWcontour.png", drawcontours)

    #get white blobs
    lefts = [tuple(c[c[:,:,0].argmin()][0]) for c in contours]
    leftcolours = [opening[l[1], l[0]+1] for l in lefts]
    tops = [tuple(c[c[:,:,1].argmin()][0]) for c in contours]
    topcolours = [opening[t[1]+1, t[0]] for t in tops]
    white = [contours[i] for i in range(len(contours)) if (leftcolours[i] == 255 and topcolours[i] == 255)]

    #redraw contours to make convex
    redraw = [cv2.convexHull(blob) for blob in white]
    #drawredraw = cv2.drawContours(bkgd, redraw, -1, (0,255,0), 1) 
    #cv2.imwrite(outname + "_BWredraw.png", drawredraw)

    # filter for blobs of right size based on extreme points
    #rightsize = []
    #for blob in redraw:
    #    points = extremepoints(blob)
    #    distances1 = [math.sqrt((points[p-1][0]-points[p][0])**2 + (points[p-1][1]-points[p][1])**2) for p in range(len(points))]
    #    distances2 = [math.sqrt((blob[p-1][0][0]-blob[p][0][0])**2 + (blob[p-1][0][1]-blob[p][0][1])**2) for p in range(len(blob))]
    #    distances = distances1 + distances2
    #    maxdist = max(distances)
    #    if 25 < maxdist < 100:
    #        rightsize.append(blob)
    #drawrightsize = cv2.drawContours(img, rightsize, -1, (0,255,0), 1) # contours of right size plotted in blue
    #cv2.imwrite(outname + "_BWrightsize.png", drawrightsize)

    #draw rectangles around the points (tilt)    
    rect = [cv2.minAreaRect(blob) for blob in redraw]
    box = [cv2.boxPoints(pts) for pts in rect]

    # draw rectangles around the rectangle (no tilt)
    rect2 = [cv2.boundingRect(blob) for blob in box if min(cv2.boundingRect(blob))> 0]
    fillrect = bkgd
    Coordinates = list()
    CroppedTags = list()
    for pts in rect2:
        cropped = img[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
        #weed out blobs that are too dark
        blur = np.ones(cropped.shape)
        for i in range(3):
            blur[:, :, i] = cv2.medianBlur(cropped[:, :, i],9)
        test = np.sum(blur, 2)
        if len(np.where(test > 100)[0])/(test.shape[0]*test.shape[1]) == 1.0 or len(np.where(test > 100)[0])/(test.shape[0]*test.shape[1]) < 0.5:
            continue
        grey = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
        
        #convert to black and white with Otsu's thresholding
        ret2,bw = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if len(np.where(bw == 0)[0])/len(np.where(bw == 255)[0]) > 5:
            continue
        
        Coordinates.append(pts)
        CroppedTags.append(cropped)        
    #cv2.imwrite(outname + "_BWfillrect.png", fillrect)

    # crop out each rectangle from red channel
    #mask = cv2.inRange(fillrect, np.array([0,0,255]), np.array([0,0,255]))
    #cropped = cv2.bitwise_and(img,img, mask= mask)
    #output = cv2.add(cropped,cv2.cvtColor(R,cv2.COLOR_GRAY2RGB))
    #cv2.imwrite(outname + "_potentialTags.png", output)

    return [Coordinates, CroppedTags, bkgd]

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

def drawtag(pts, cropped, bkgd, outname, a, taglist):
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
    #cv2.imwrite(outname + "_approx_" + str(a) + ".png", polygon)

    #first try
    vertexes = extremepoints(approx)  
    edges = [math.sqrt((vertexes[p-1][0]-vertexes[p][0])**2 + (vertexes[p-1][1]-vertexes[p][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))

    if edge == 0:
        return [bkgd, ['not tag', 'not tag', 'not tag', 'not tag', 'not tag', 'not tag']]

    OneCM = edge/0.3
    rows,cols = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY).shape
    pts1 = np.float32([vertexes[0],vertexes[2],vertexes[3]])
    pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY),M,(cols,rows))
    tag = dst[0:edge, 0:edge]   

    if np.sum(np.isnan(tag)) !=0:
        return [bkgd, ['not tag', 'not tag', 'not tag', 'not tag', 'not tag', 'not tag']]        

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
            if TAG1[i, j] > 128: #this is on purpose middle values are tricky
                TAG1[i,j] = 255
    
    #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG1)
    results = scoretag(TAG1, taglist) # score, dir, id
    if results[0] < 2:
        centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
        centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
        cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(bkgd,(centroidX,centroidY), 5, (255,255,0), -1)
        taglist.remove(results[2])
        return [bkgd, [results[2], centroidX, centroidY, results[1], OneCM, results[0]]]

    #second try
    vertexes = closesttosides(approx)  
    edges = [math.sqrt((vertexes[p-1][0][0]-vertexes[p][0][0])**2 + (vertexes[p-1][0][1]-vertexes[p][0][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))

    if edge == 0:
        vertexes = extremepoints(approx)
        centroidX = statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0]
        centroidY = statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1]
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,0,255),3)
        return [bkgd, ['X', centroidX, centroidY, 'dir', 'X', 'X']]
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
        #cv2.imwrite(outname + "_bwtag_" + str(a) + ".png", bwtag)

        #enlarge
        bwtag2 = cv2.resize(bwtag, (bwtag.shape[0]*6, bwtag.shape[1]*6))
        #cv2.imwrite(outname + "_bwtag2_" + str(a) + ".png", bwtag2)
        
        TAG2 = np.full((6, 6), 255)
        for i in range(6):
            for j in range(6):
                TAG2[i,j] = np.mean(bwtag2[bwtag.shape[0]*i:bwtag.shape[0]*(i+1), bwtag.shape[0]*j:bwtag.shape[0]*(j+1)])
                if TAG2[i, j] < 127:
                    TAG2[i,j] = 0
                if TAG1[i, j] > 128: #this is on purpose middle values are tricky
                    TAG2[i,j] = 255
        
        #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG1)
        results = scoretag(TAG2, models, taglist) # score, dir, id
        if results[0] < 2:
            centroidX = statistics.mean([vertexes[0][0][0], vertexes[1][0][0], vertexes[2][0][0], vertexes[3][0][0]]) + pts[0]
            centroidY = statistics.mean([vertexes[0][0][1], vertexes[1][0][1], vertexes[2][0][1], vertexes[3][0][1]]) + pts[1]
            cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
            cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
            cv2.circle(bkgd,(centroidX,centroidY), 5, (255,255,0), -1)
            return [bkgd, [results[2], centroidX, centroidY, results[1], OneCM, results[0]]]
        centroidX = statistics.mean([vertexes[0][0][0], vertexes[1][0][0], vertexes[2][0][0], vertexes[3][0][0]]) + pts[0]
        centroidY = statistics.mean([vertexes[0][0][1], vertexes[1][0][1], vertexes[2][0][1], vertexes[3][0][1]]) + pts[1]
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,0,255),3)
        return [bkgd, ['X', centroidX, centroidY, 'dir', 'X', 'X']]

def main(argv):
    """ Main entry point of the program """
    if len(sys.argv) == 2:
        filename = argv[1]
    else:
        iter = os.getenv('PBS_ARRAY_INDEX')
        files = ['/rds/general/user/tst116/home/TrackBEETag/Data' + "/" + i for i in os.listdir('/rds/general/user/tst116/home/TrackBEETag/Data')]
        filename = files[int(iter)-1]
    print (filename)

    #filename = '../Data/R1D7R2A1.MP4'
    base = os.path.splitext(os.path.basename(filename))[0]
    outname = '../Results/' + base
    container = av.open(filename)

    if base[6] == 'A':
        taglist = [237,74,121,137,151,180,181,220,222,311,312,341,402,421,427,456,467,596,626,645,664,681,696,697,765,781,794,862,985,1077,1419,1846,1947,1966,2908,2915]
    elif base[6] == 'B':
        taglist = [180,74,121,137,151,181,186,220,222,237,311,312,341,393,421,427,467,534,574,596,626,645,664,681,696,697,765,781,862,985,1077,1419,1846,1947,1966,2908,2915]
    elif base[6] == 'C':
        taglist = [862,121,137,151,180,181,186,220,222,237,341,393,402,421,456,467,534,574,596,626,645,664,681,696,697,765,781,794,985,1077,1419,1846,1947,1966,2908,2915]
    elif base[6] == 'D':
        taglist = [534,74,121,137,151,186,220,222,237,311,312,341,393,402,421,427,456,467,574,596,626,645,664,681,696,697,781,794,862,985,1077,1419,1846,1947,1966,2908]
    models = [drawmodel(id) for id in taglist]
    wrangled = pd.DataFrame()
    f = 0
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='bgr24')
        #break

        frameData = pd.DataFrame()
        scores = pd.DataFrame()
        directions = pd.DataFrame()
        Coordinates, Cropped, bkgd = findtags(img, outname+str(f))
        a = 0
        while a < len(Coordinates):
            if drawtag(Coordinates[a], Cropped[a], bkgd, outname, a, models, taglist) != None:
                bkgd, row = drawtag(Coordinates[a], Cropped[a], bkgd, outname, a, models, taglist) #[results[2], 'centroidX', 'centroidY', results[1], 'OneCM', results[0]]
                if row[0] != 'not tag':
                    completerow = [f] + row
                    frameData = frameData.append([tuple(completerow)], ignore_index=True)
                    print(row)
                    a = a+1
            else:
                a = a+1
        wrangled = wrangled.append(frameData, ignore_index=True)
        #cv2.imwrite(outname + '_' + str(f) + "_foundtags.png", bkgd)
        print("Finished frame " + str(f))
        f = f+1

    output = wrangled.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm', 6:'score'})
    output.to_csv(path_or_buf = outname + "_raw.csv", na_rep = "NA", index = False)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)