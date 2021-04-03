#imports
import sys
import av
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

#load image and get name of output file
image = "P1000027.png"
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
kernel = np.ones((4,4),np.uint8) # larger kernel preserves less details, use for images that are less well-lit
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
i = 0
ranges = []
for pts in rect:
    #crop out potential tag region
    cropped = R[pts[1]:pts[1]+pts[3], pts[0]:pts[0]+pts[2]]
    cv2.imwrite(outname + "_cropped" + str(i) + ".png", cropped)
    ranges.append(cropped.max() - cropped.min())
    if cropped.max() - cropped.min() > 125:
        #locate corners
        croppedbkgd = cv2.cvtColor(cropped,cv2.COLOR_GRAY2RGB)
                
        corners = cv2.goodFeaturesToTrack(cropped,25,0.01,10)
        corners = np.int0(corners)

        for j in corners:
            x,y = j.ravel()
            cv2.circle(croppedbkgd,(x,y),3,255,-1)
                
        cv2.imwrite(outname + "_corners" + str(i) + ".png", croppedbkgd)

#find squares: edges based on corners
#largest square that is white = tag
#transform to square
#compare to known tags
#give identity or throw out
#get direction and number of pixels to 1cm
#refer to original position to get centroid coordinates
#output as pandas series


        ######################
        #turn into black and white
        #bw = cv2.adaptiveThreshold(cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        bw = cv2.adaptiveThreshold(cropped,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        cv2.imwrite(outname + "_bw" + str(i) + ".png", bw)
        #open
        kernel = np.ones((2,2),np.uint8) # larger kernel preserves less details
        opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        #cv2.imwrite(outname + "_BWopen" + str(i) + ".png", opened)
        #draw contours
        bkgd = cv2.cvtColor(cropped,cv2.COLOR_GRAY2RGB)
        contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #contour = cv2.drawContours(bkgd, contours, -1, (0,0,255), 1) # all contours plotted in green
        #cv2.imwrite(outname + "_BWopencontour" + str(i) + ".png", contour)

        epsilon = [0.05*cv2.arcLength(blob,True) for blob in contours]
        redraw = [cv2.approxPolyDP(contours[i],epsilon[i],True) for i in range(len(contours))]
        nonzero = [blob for blob in redraw if cv2.contourArea(blob) > 0]
        redrawed = cv2.drawContours(bkgd, nonzero, -1, (0,255,0), 1)
        cv2.imwrite(outname + "_BWredraw" + str(i) + ".png", redrawed)
#######################        
    i = i+1