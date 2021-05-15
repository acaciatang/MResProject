import cv2
import numpy as np
from matplotlib import pyplot as plt

overexposed = cv2.imread('overexposed.png')
good = cv2.imread('good.png')

color = ('b','g','r')
hist = plt.figure()
for i,col in enumerate(color):
    histr = cv2.calcHist([good],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

hist.savefig("good.pdf", bbox_inches='tight')

color = ('b','g','r')
for i in range(len(color)):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cv2.imwrite(color[i]+'.png',overexposed[:, :, i])
    overexposed[:, :, i] = clahe.apply(overexposed[:, :, i])

cv2.imwrite('good_equalised.png',good)


overexposed[:, :, i] = cv2.equalizeHist(overexposed[:, :, i])