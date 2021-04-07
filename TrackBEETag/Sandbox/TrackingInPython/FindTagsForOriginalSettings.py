#imports
import sys
import av
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#load image and get name of output file
image = "P1000023.png"
img = cv2.imread(image, 1)
outname = os.path.splitext(os.path.basename(image))[0]

# separate out colour channels of interest: green (G) for detecting tags, red (R) for recognizing tags
G = img[:, :, 1]
#cv2.imwrite(outname + "_G.png", G)
R = img[:, :, 2]
#cv2.imwrite(outname + "_R.png", R)
bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)

#convert to black and white with mean thresholding
bw = cv2.adaptiveThreshold(G,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
cv2.imwrite(outname + "_BW.png", bw)

#removes white points that are noise, which gives us nice black borders of blobs
kernel = np.ones((6,6),np.uint8) # larger kernel preserves less details
opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
cv2.imwrite(outname + "_BWopen.png", opening)

# find blobs
contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contour = cv2.drawContours(bkgd, contours, -1, (255,255,255), 1) # all contours plotted in green
#cv2.imwrite(outname + "_BWopencontour.png", contour)

# filter for blobs of right size
areas = [cv2.contourArea(blob) for blob in contours]
rightsized = [contours[i] for i in range(len(areas)) if 1000 < areas[i] < 5000]
#rightsize = cv2.drawContours(bkgd, rightsized, -1, (255,0,0), 1) # contours of right size plotted in blue
#cv2.imwrite(outname + "_BWrightsize.png", rightsize)
rightsizeareas = [cv2.contourArea(blob) for blob in rightsized]

# filter for blobs of right perimeter to area ratio
perimeters = [cv2.arcLength(blob,True) for blob in rightsized]
ratio = [perimeters[i]/rightsizeareas[i] for i in range(len(perimeters))]
rightratios = [rightsized[i] for i in range(len(rightsized)) if ratio[i] < 0.3] #0.2?
#rightratio = cv2.drawContours(bkgd, rightratios, -1, (255,255,0), 1) # contours of right size plotted in blue
#cv2.imwrite(outname + "_BWrightratio.png", rightratio)
#rightones = [cv2.arcLength(blob,True)/cv2.contourArea(blob) for blob in rightratios]

# redraw contours
epsilon = [0.025*cv2.arcLength(blob,True) for blob in rightratios]
redraw = [cv2.approxPolyDP(rightratios[i],epsilon[i],True) for i in range(len(rightratios))]
nonzero = [blob for blob in redraw if cv2.contourArea(blob) > 0]
#redrawed = cv2.drawContours(bkgd, nonzero, -1, (0,255,0), 1)
#cv2.imwrite(outname + "_BWredraw.png", redrawed)

# filter for blobs of right perimeter to area ratio
ratio2 = [cv2.arcLength(blob,True)/cv2.contourArea(blob) for blob in nonzero]
rightratios2 = [nonzero[i] for i in range(len(nonzero)) if ratio2[i] < 0.15]
#rightratio2 = cv2.drawContours(bkgd, rightratios2, -1, (0,255,255), 1) # contours of right size plotted in blue
#cv2.imwrite(outname + "_BWrightratio2.png", rightratio2)
#rightones2 = [cv2.arcLength(blob,True)/cv2.contourArea(blob) for blob in rightratios2]

# draw rectangles around the points (no tilt)
rect = [cv2.boundingRect(blob) for blob in rightratios2]
for pts in rect:
    fillrect = cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0] + pts[2], pts[1] + pts[3]),(0,0,255),-1)
    fillrect = bkgd
#cv2.imwrite(outname + "_BWfillrect.png", fillrect)

# draw rectangles around the points (with tilt)
#rect = [cv2.minAreaRect(blob) for blob in rightratios2]
#box = [cv2.boxPoints(rectangles) for rectangles in rect]
#for boxpts in box:
#    boxpts = np.int0(boxpts)
#    boxpts = boxpts.reshape((-1, 1, 2)) 
#    drawrect = cv2.drawContours(bkgd,[boxpts],-1,(0,0,255),2)
#    drawrect = bkgd
#cv2.imwrite(outname + "_BWdrawrect.png", drawrect)

# crop out each rectangle, use red channel and convert to black and white
#for boxpts in box:
#    boxpts = np.int0(boxpts)
#    fillrect = cv2.fillPoly(bkgd, pts =[boxpts], color=(0,0,255))
#    fillrect = bkgd
#cv2.imwrite(outname + "_BWfillrect.png", fillrect)

# crop out each rectangle from red channel
mask = cv2.inRange(fillrect, np.array([0,0,255]), np.array([0,0,255]))
bkgd = cv2.cvtColor(R,cv2.COLOR_GRAY2RGB)
cropped = cv2.bitwise_and(R,R, mask= mask)
cv2.imwrite(outname + "_BWcropped.png", cropped)

# convert to black and white
bw2 = cv2.adaptiveThreshold(cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imwrite(outname + "_BW2.png", bw2)
#kernel2 = np.ones((3,3),np.uint8)
#closing = cv2.morphologyEx(bw2, cv2.MORPH_CLOSE, kernel2)
#cv2.imwrite(outname + "_BW2closing.png", closing)
#opening2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, kernel2)
#cv2.imwrite(outname + "_BW2opening.png", opening2)

#cropped = cropped.astype('float')
#cropped[cropped == 0] = 135
#blur = cv2.GaussianBlur(cropped,(3,3),0)
#ret3,bw2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# draw rectangles around the points (no tilt)
rect = [cv2.boundingRect(blob) for blob in rightratios2]
ranges = []
for a in range(len(rect)):
    pts = rect[a]
    #crop out potential tag region
    cropped = R[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
    croppedbkgd = cv2.cvtColor(cropped,cv2.COLOR_GRAY2RGB)
    #cv2.imwrite(outname + "_cropped" + str(a) + ".png", cropped)
    bw = cv2.adaptiveThreshold (cropped,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    #cv2.imwrite(outname + "_bw" + str(a) + ".png", bw)

    contrast = cropped//32
    contrast = contrast * 32
    #cv2.imwrite(outname + "_contrast" + str(a) + ".png", contrast)
    
    th = 1 + (int(np.max(contrast)) + int(np.min(contrast)))/2
    ret,th1 = cv2.threshold(contrast,max(th, 64),255,cv2.THRESH_BINARY)
    #cv2.imwrite(outname + "_bw" + str(a) + ".png", th1)

    contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]
    
    leftmost = largest[largest[:,:,0].argmin()][0]
    rightmost = largest[largest[:,:,0].argmax()][0]
    topmost = largest[largest[:,:,1].argmin()][0]
    bottommost = largest[largest[:,:,1].argmax()][0]
    
    points = [leftmost, rightmost, topmost, bottommost]
    POIs = [int(not (pts[2]-1 in p or 0 in p or pts[3]-1 in p)) for p in points]
    
    if sum(POIs) >= 2:
        #drawlargest = cv2.drawContours(croppedbkgd, largest, -1, (125,255,255), 1)
        #cv2.imwrite(outname + "_largest" + str(a) + ".png", drawlargest)
        arclen = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, arclen*0.01, True)
        polygon = cv2.drawContours(croppedbkgd, [approx], -1, (0,0,255), 1, cv2.LINE_AA)
        #cv2.imwrite(outname + "_polygon" + str(a) + ".png", polygon)
        test = [0]
        if 2 < len(approx) < 5: #will not be able to fit ellipse
            polygon = cv2.drawContours(croppedbkgd, [approx], -1, (0,255,0), 1, cv2.LINE_AA)
            cv2.imwrite(outname + "_polygon2_" + str(a) + ".png", polygon)
        else:
            (x,y),(MA,ma),angle = cv2.fitEllipse(approx)
            MaxForRemoval = min(MA, ma)*2/3
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
                
                #do not remove anything if points are too far away
                #if DisToClosest[test.index(max(test))] > MaxForRemoval:
                #    Closest[test.index(max(test))] = 1
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

            if len(approx) > 2:    
                polygon = cv2.drawContours(croppedbkgd, [approx], -1, (0,255,0), 1, cv2.LINE_AA)
                cv2.imwrite(outname + "_polygon2_" + str(a) + ".png", polygon)
    print("Done " + str(a) + "!")