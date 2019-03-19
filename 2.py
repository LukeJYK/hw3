import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
dir = 'C:\\Users\\lenovo\\Desktop\\3\\images'
for files in os.listdir(dir):
    #print(files)
    img = cv2.imread('C:\\Users\\lenovo\\Desktop\\3\\images\\'+files,0)
    res = cv2.equalizeHist(img)
    arr = img.flatten()
    arr1 = res.flatten()
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.hist(arr, bins=256, density=1, facecolor='green')
    plt.subplot(223), plt.imshow(res,'gray')
    plt.subplot(224), plt.hist(arr1, bins=256, density=1, facecolor='green')
    plt.show()
