import cv2
from matplotlib import pyplot as plt
import os
dir = 'C:\\Users\\lenovo\\Desktop\\3\\images'
for files in os.listdir(dir):
    img = cv2.imread('C:\\Users\\lenovo\\Desktop\\3\\images\\'+ files,0)
    arr = img.flatten()
    plt.hist(arr, bins=256, density=1, facecolor='green')
    plt.show()
