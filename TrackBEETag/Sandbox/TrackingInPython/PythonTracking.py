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
    #cv2.imwrite(outname + "_BW.png", bw)

    #make blobs more blobby
    kernel = np.ones((10,10),np.uint8)
    close = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite(outname + "_close.png", close)
    
    # find blobs
    contours, hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [blob for blob in contours if cv2.contourArea(blob) > 0]
    #drawcontours = cv2.drawContours(bkgd, contours, -1, (0,255,255), 1) # all closed contours plotted in yellow
    #cv2.imwrite(outname + "_BWcontour.png", drawcontours)

    #get white blobs (too slow not worth it)
    #whites = []
    #mask = np.zeros(G.shape,np.uint8)
    #for blob in contours:
    #    cv2.drawContours(mask,[blob],0,255,-1)
    #    pixelpoints = np.transpose(np.nonzero(mask))
    #    meanVal = cv2.mean(bw,mask = mask)[0]
    #    if meanVal > 50:
    #        whites.append(blob)

    # redraw contours to make convex
    redraw = [cv2.convexHull(blob) for blob in contours]
    #drawredraw = cv2.drawContours(bkgd, redraw, -1, (0,255,0), 1) 
    #cv2.imwrite(outname + "_BWredraw.png", drawredraw)

    # filter for blobs of right size based on extreme points
    rightsize = []
    for blob in redraw:
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
        if np.min(roi) < 113 and min(pts[2],pts[3]) > 25 and np.max(roi) > 113:
                    fillrect = cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,0,255),-1)
                    fillrect = bkgd
                    potentialTags.append(pts)
    #cv2.imwrite(outname + "_BWfillrect.png", fillrect)

    # crop out each rectangle from red channel
    mask = cv2.inRange(fillrect, np.array([0,0,255]), np.array([0,0,255]))
    cropped = cv2.bitwise_and(img,img, mask= mask)
    output = cv2.add(cropped,img)
    cv2.imwrite(outname + "_potentialTags.png", output)

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

    model = np.rot90(model) # I don't know why the matlab cod is like this but it is
    return model*255
    
def IDTags(pts, R, outname):
for a in range(len(potentialTags)):
    pts = potentialTags[a]
    #crop out potential tag region
    cropped = R[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
    croppedbkgd = cv2.cvtColor(cropped,cv2.COLOR_GRAY2RGB)
    #cv2.imwrite(outname + "_cropped" + str(a) + ".png", cropped)
    #bw = cv2.adaptiveThreshold (cropped,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    
    #convert to black and white with Otsu's thresholding
    ret2,bw = cv2.threshold(cropped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    #cv2.imwrite(outname + "_bw" + str(a) + ".png", bw)

    #contrast = cropped//32
    #contrast = contrast * 32
    #cv2.imwrite(outname + "_contrast" + str(a) + ".png", contrast)
    
    #th = 1 + (int(np.max(contrast)) + int(np.min(contrast)))/2
    #ret,th1 = cv2.threshold(contrast,max(th, 64),255,cv2.THRESH_BINARY)
    #cv2.imwrite(outname + "_bw" + str(a) + ".png", th1)

    contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]
    
    leftmost = largest[largest[:,:,0].argmin()][0]
    rightmost = largest[largest[:,:,0].argmax()][0]
    topmost = largest[largest[:,:,1].argmin()][0]
    bottommost = largest[largest[:,:,1].argmax()][0]
    
    points = [leftmost, rightmost, topmost, bottommost]
    POIs = [int(not (pts[2]-1 in p or 0 in p or pts[3]-1 in p)) for p in points]
    
    if sum(POIs) < 2:
        continue
        #drawlargest = cv2.drawContours(croppedbkgd, largest, -1, (125,255,255), 1)
        #cv2.imwrite(outname + "_largest" + str(a) + ".png", drawlargest)
    arclen = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, arclen*0.01, True)
    #polygon = cv2.drawContours(croppedbkgd, [approx], -1, (0,0,255), 1, cv2.LINE_AA)
    #cv2.imwrite(outname + "_polygon" + str(a) + ".png", polygon)
    test = [0]
    while True:
        edgeLens = [math.sqrt((approx[i-1][0][0]-approx[i][0][0])**2 + (approx[i-1][0][1]-approx[i][0][1])**2) for i in range(len(approx))]
        Closest = []
        DisToClosest = []
        ActualDis = []
        
        for i in range(len(approx)):
            Js = [j for j in range(len(approx)) if j != i]
            distances = [math.sqrt((approx[i][0][0]-approx[j][0][0])**2 + (approx[i][0][1]-approx[j][0][1])**2) for j in Js]
            index = distances.index(min(distances))
            indices = [index, i]
            Closest.append(Js[index])
            DisToClosest.append(distances[index])
            actual1 = sum([edgeLens[(x+1)%len(approx)] for x in range(min(indices), max(indices))])
            actual2 = sum([edgeLens[(x+1)%len(approx)] for x in range(len(approx)) if x not in range(min(indices), max(indices))])
            ActualDis.append(min([actual1, actual2]))
        
        test = [min(abs(Closest[x] - x), abs(len(approx) - abs(Closest[x] - x))) for x in range(len(approx))]
        
        #check that all points are closest to adjacent points
        if max(test) == 1:
            break
        start = test.index(max(test))
        end = Closest[test.index(max(test))]
        remove1 = [i for i in range(min(start, end) + 1, max(start, end))]

        start2 = test.index(max(test)) - len(approx)
        remove2 = [i for i in range(min(start2, end) + 1, max(start2, end))]
        end2 = Closest[test.index(max(test))] - len(approx)
        remove3 = [i for i in range(min(start, end2) + 1, max(start, end2))]

        minlen = min(len(remove1), len(remove2), len(remove3))
        if len(remove1) == minlen:
            removeIndex = remove1
        elif len(remove2) == minlen:
            removeIndex = remove2
        else:
            removeIndex = remove3
        if len(removeIndex) == 0:
            break

        removeIndex = [i%len(approx) for i in removeIndex]
        
        approx = np.array([approx[i] for i in range(len(approx)) if i not in removeIndex])
        print("Removed:")
        print(removeIndex)

    if len(approx) < 2:
        continue
    
    moments = cv2.moments(approx)
    centroidX = int(moments['m10']/moments['m00'])
    centroidY = int(moments['m01']/moments['m00'])
    #polygon = cv2.drawContours(croppedbkgd, [approx], -1, (0,255,0), 1, cv2.LINE_AA)
    #cv2.imwrite(outname + "_polygon2_" + str(a) + ".png", polygon)

# do transformation better, this is completely unacceptable

    #####get centroid here before transformation!######

    #transform polygon
    vertexes = extremepoints(approx)
    edges = [math.sqrt((vertexes[p-1][0]-vertexes[p][0])**2 + (vertexes[p-1][1]-vertexes[p][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))
    if edge < 6:
        continue
    OneCM = edge/0.3 ##measure the tag to get actual edge length!!
    rows,cols = cropped.shape
    pts1 = np.float32([leftmost,rightmost,bottommost])
    pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(cropped,M,(cols,rows))
    tag = dst[0:edge, 0:edge]
    #cv2.imwrite(outname + "_tag_" + str(a) + ".png", tag)
    
#add gate for too little contrast

    #draw tag
    ret2,bwtag = cv2.threshold(tag,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imwrite(outname + "_bwtag_" + str(a) + ".png", bwtag)

    TAG1 = np.full((6, 6), 255)
    thres = np.mean(bwtag)
    celledge = edge/6
    e = 0
    for i in range(6):
        for j in range(6):
            test = np.mean(bwtag[round(celledge*i):round(celledge*(i+1)), round(celledge*j):round(celledge*(j+1))])
            if test < thres:
                if i == 0 or j == 0:
                    e = e+1
                else:
                    TAG1[i,j] = 0
                    
    #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG1)

    TAG2 = np.full((6, 6), 255)
    celledge = round(edge/6)
    thres = np.mean(bwtag[celledge:celledge*5, celledge:celledge*5])
    for i in range(6):
        for j in range(6):
            TAG2[i,j] = np.median(bwtag[round(celledge*i):round(celledge*(i+1)), round(celledge*j):round(celledge*(j+1))])
            if TAG2[i,j] == 0 and (i == 0 or j == 0):
                TAG2[i,j] = 255
    #cv2.imwrite(outname + "_bwTAG2_" + str(a) + ".png", TAG2)

    #ID tag
    configs = [TAG1, np.rot90(TAG1, 1), np.rot90(TAG1, 2), np.rot90(TAG1, 3), TAG2, np.rot90(TAG2, 1), np.rot90(TAG2, 2), np.rot90(TAG2, 3)]
    if image[0] == 'A':
        taglist = [68,118,137,173,289,304,325,365,392,420,437,512,559,596,613,666,696,765,862,1112,1150,1203,1492,1730,1966,2091,2327,2452,2511,2932,2992,3067,3261,3360,3415,3486,3570,3757,3908,4015]
    elif image[0] == 'B':
        taglist = [31,46,69,180,222,270,311,330,347,393,542,598,651,697,792,813,875,1062,1085,1227,1368,1498,1585,1744,1947,1986,2056,2158,2281,2332,2460,2607,2835,2908,2945,3375,3488,3581,3783,3926]
    elif image[0] == 'C':
        taglist = [52,74,103,209,226,274,312,331,354,427,455,476,502,544,574,601,634,661,707,770,881,1028,1180,1243,1465,1543,1704,1759,1797,1846,1896,2118,2340,2413,2488,2523,2915,2954,3134,3832]
    elif image[0] == 'D':
        taglist = [59,75,104,135,211,237,324,341,361,377,413,436,456,510,579,609,637,664,681,720,802,844,910,1074,1104,1403,1620,1718,1799,1903,2006,2072,2192,2242,2355,2856,2880,3163,3358,3388]

    models = [drawmodel(id) for id in taglist]
    difference = [np.sum(abs(m - config))/255 for config in configs for m in models]
    test = min(difference)
    bestfits = [d for d in difference if d == test]
    if len(bestfits) > 1:
        continue
    #if test >= 4:
    #    continue
    id = taglist[difference.index(min(difference))%40]
    cv2.imwrite(outname + "_bestfit_" + str(a) + "_" + str(id) + ".png",drawmodel(id))
    cv2.imwrite(outname + "_cropped" + str(a) + ".png", cropped)

    #return [(frameNum, ID, centroidX, centroidY, dir, OneCM)]

# add script to figure out best fit given that other tags may have taken the match with a higher score
# try choosing based on best fit to each model?

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

#testdata
#filename = 'P1000073.MP4'
#image = 'D6_0.png'
#image = 'B.png'
findtags('A_000000.png')
findtags('B_000000.png')
findtags('C_000000.png')
findtags('D_000000.png')