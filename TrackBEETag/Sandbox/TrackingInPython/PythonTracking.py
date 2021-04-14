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

def convert(vertexes, x, y):
    for i in range(len(vertexes)):
        vertexes[i][0] = vertexes[i][0] + x
        vertexes[i][1] = vertexes[i][1] + y

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
    contours, hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
        if np.min(roi) < 113 and min(pts[2],pts[3]) > 25 and np.max(roi) > 113:
                    fillrect = cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,0,255),-1)
                    fillrect = bkgd
                    potentialTags.append(pts)
    #cv2.imwrite(outname + "_BWfillrect.png", fillrect)

    # crop out each rectangle from red channel
    mask = cv2.inRange(fillrect, np.array([0,0,255]), np.array([0,0,255]))
    cropped = cv2.bitwise_and(img,img, mask= mask)
    output = cv2.add(cropped,cv2.cvtColor(R,cv2.COLOR_GRAY2RGB))
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

    model = np.rot90(model)*255 # I don't know why the matlab cod is like this but it is
    return model.astype(int)
    
def drawtag(pts, R, outname, a):
    #pts = potentialTags[a]
    #R = img[:,:,2]
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
        return None
    #drawlargest = cv2.drawContours(croppedbkgd, largest, -1, (125,255,255), 1)
    #cv2.imwrite(outname + "_largest" + str(a) + ".png", drawlargest)
    arclen = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, arclen*0.01, True)
    #polygon = cv2.drawContours(croppedbkgd, [approx], -1, (0,0,255), 1, cv2.LINE_AA)
    #cv2.imwrite(outname + "_polygon" + str(a) + ".png", polygon)
    test = [0]
#this is bad, try something else
# find mean within polygon and transform, better yet do ozru on polygon
# alternatively but this is harder, find parallel lines
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
        #print("Removed:")
        #print(removeIndex)

    if len(approx) < 2:
        return None
    
    moments = cv2.moments(approx)
    centroidX = int(moments['m10']/moments['m00']) + pts[0]
    centroidY = int(moments['m01']/moments['m00']) + pts[1]
    #polygon = cv2.drawContours(croppedbkgd, [approx], -1, (0,255,0), 1, cv2.LINE_AA)
    #cv2.imwrite(outname + "_polygon2_" + str(a) + ".png", polygon)

    #transform polygon (polygon should just be the tag)
    vertexes = extremepoints(approx)
    corners = convert(vertexes, pts[0], pts[1])
    edges = [math.sqrt((vertexes[p-1][0]-vertexes[p][0])**2 + (vertexes[p-1][1]-vertexes[p][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))

    OneCM = edge/0.3 ##measure the tag to get actual edge length!!
    rows,cols = cropped.shape
    pts1 = np.float32([leftmost,rightmost,bottommost])
    pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(cropped,M,(cols,rows))
    tag = dst[0:edge, 0:edge]

    if tag.shape[0] != tag.shape[1]:
        edge = min(tag.shape[0], tag.shape[1])
        tag = dst[0:edge, 0:edge]
    
    if edge < 6:
        return None
    #draw tag
    ret2,bwtag = cv2.threshold(tag,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imwrite(outname + "_bwtag_" + str(a) + ".png", bwtag)

    #enlarge
    bwtag = cv2.resize(bwtag, (edge*6, edge*6))

    TAG1 = np.full((6, 6), 255)
    thres = np.mean(bwtag)
    for i in range(6):
        for j in range(6):
            test = np.mean(bwtag[edge*i:edge*(i+1), edge*j:edge*(j+1)])
            if i == 5:
                test = np.mean(bwtag[edge*i:edge*(i+1), edge*j:edge*(j+1)])
                if j == 5: 
                    test = np.mean(bwtag[edge*i:edge*(i+1), edge*j:edge*(j+1)])
            if j == 5:
                test = np.mean(bwtag[edge*i:edge*(i+1), edge*j:edge*(j+1)])
            if test < thres:
                if i == 0 or j == 0:
                    continue
                else:
                    TAG1[i,j] = 0
                    
    #cv2.imwrite(outname + "_bwTAG1_" + str(a) + ".png", TAG1)

    TAG2 = np.full((6, 6), 255)
    thres = np.mean(bwtag[edge:edge*5, edge:edge*5])
    for i in range(6):
        for j in range(6):
            TAG2[i,j] = np.median(bwtag[edge*i:edge*(i+1), edge*j:edge*(j+1)])
            if i == 0 or j == 0:
                TAG2[i,j] = 255
    #cv2.imwrite(outname + "_bwTAG2_" + str(a) + ".png", TAG2)
    return [[TAG1, TAG2], corners, centroidX, centroidY, OneCM]

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

def matchtags(df): #df is a pandas dataframe that contains the scores, columns are models, rows are tags
    # generate dictionary of preferences based on all tags returned from IDTags
    modellist = df.columns.to_list()
    taglist = df.index.to_list()
    
    modelscores = [df[x].values.tolist() for x in df.columns]
    modelranks = [[sorted(model).index(i) for i in model]for model in modelscores]
    tagsranked = []
    for i in range(len(modellist)):
        zipped_lists = zip(modelranks[i], taglist)
        sorted_zipped_lists = sorted(zipped_lists)
        tagsranked.append([element for _, element in sorted_zipped_lists])
    modelpref = {modellist[i]: tagsranked[i] for i in range(len(modellist))}

    tagscores = df.values.tolist()
    tagranks = [[sorted(tag).index(i) for i in tag]for tag in tagscores]
    modelsranked = []
    for i in range(len(taglist)):
        zipped_lists = zip(tagranks[i], modellist)
        sorted_zipped_lists = sorted(zipped_lists)
        modelsranked.append([element for _, element in sorted_zipped_lists])
    tagpref = {taglist[i]: modelsranked[i] for i in range(len(taglist))}

    capacity = {h: 1 for h in modelpref}

    game = HospitalResident.create_from_dictionaries(tagpref, modelpref, capacity)
    solution = game.solve()
    assert game.check_validity(), "No valid solution"
    assert game.check_stability(), "No stable solution"

    #matched_tags = [str(tags) for _, [tags] in solution.items()]
    #unmatched_tags = set(tagpref.keys()) - set(matched_tags)
    #print("Tags without matches: ", unmatched_tags)

    return solution

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
    filename = 'A1.MP4'
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

    models = [drawmodel(id) for id in taglist]
    wrangled = pd.DataFrame()
    f = 0
    for frame in container.decode(video=0):
        if f > 20:
            break
        img = frame.to_ndarray(format='bgr24')
        #break
        out = outname +  "_" + str(f)

        frameData = pd.DataFrame()
        scores = pd.DataFrame()
        directions = pd.DataFrame()
        potentialTags = findtags(img, out) #potentialTags
        As = list()
        for a in range(len(potentialTags)):
            raw = drawtag(potentialTags[a], img[:,:,2], out, a) # [[TAG1, TAG2], vertexes, centroidX, centroidY, OneCM]
            if raw == None:
                continue
            frontchoice = [np.array([round((raw[1][i][0]+raw[1][(i+1)%4][0])/2), round((raw[1][i][1]+raw[1][(i+1)%4][1])/2)])  for i in range(4)]
            dirchoice = [math.degrees(math.atan2(frontchoice[i][0]-raw[2], raw[3]-frontchoice[i][1])) for i in range(4)]
            row = [(f, a, raw[2], raw[3], None, raw[4])] # [(frameNum, ID, centroidX, centroidY, dir, OneCM)]
            frameData = frameData.append(row, ignore_index=True)

            TAG1score = scoretag(raw[0][0], models)[0]
            TAG1dir = scoretag(raw[0][0], models)[1]
            TAG2score = scoretag(raw[0][1], models)[0]
            TAG2dir = scoretag(raw[0][1], models)[1]

            score = [[min(TAG1score[i], TAG2score[i]) for i in range(len(models))]]
            
            dir = []
            for i in range(len(models)):
                if TAG1score[i] < TAG2score[i]:
                    dir.append(TAG1dir[i])
                else:
                    dir.append(TAG2dir[i])
            dir = [dirchoice[d] for d in dir]
            dir = [dir]

            As.append(a)

            scores = scores.append(score, ignore_index=True)
            directions = directions.append(dir, ignore_index=True)
            print(a)
            
        scores = scores.rename(index = {h: As[h] for h in scores.index}, columns = {h: taglist[h] for h in scores.columns})
        directions = directions.rename(index = {h: As[h] for h in directions.index}, columns = {h: taglist[h] for h in directions.columns})

        matchResults = matchtags(scores)
        values = []
        D = list(matchResults.values())
        for d in D:
            if d != []:
                values.append(str(d[0]))
    
        frameData[1] = [list(matchResults.keys())[values.index(str(i))] for i in frameData[1]]
        frameData[4] = [directions[int(str(frameData[1][i]))][As[i]] for i in frameData.index]

        wrangled = wrangled.append(frameData, ignore_index=True)
        print("Finished frame " + str(f))
        f = f+1

    output = wrangled.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm'})
    output.to_csv(path_or_buf = outname + ".csv", na_rep = "NA", index = False)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)