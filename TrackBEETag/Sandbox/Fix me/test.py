import cv2
import numpy as np
from matplotlib import pyplot as plt

overexposed = cv2.imread('overexposed.png')
good = cv2.imread('good.png')
tags = cv2.imread('tags.png')

color = ('b','g','r')
hist = plt.figure()
for i,col in enumerate(color):
    histr = cv2.calcHist([tags],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

hist.savefig("tags.pdf", bbox_inches='tight')

color = ('b','g','r')
for i in range(len(color)):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cv2.imwrite(color[i]+'.png',tags[:, :, i])
    tags[:, :, i] = clahe.apply(tags[:, :, i])

cv2.imwrite('tags_equalised.png',tags)


tags[:, :, i] = cv2.equalizeHist(tags[:, :, i])

frame = tags

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([0,100,100])
upper_blue = np.array([130,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(frame, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)

cv2.imwrite('tags_res.png',res)