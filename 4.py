import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
dir=["C:\\Users\\lenovo\\Desktop\\3\\images\\elain.bmp","C:\\Users\\lenovo\\Desktop\\3\\images\\lena.bmp"]
for files in dir:
    img = cv2.imread(files,0)
    img_array = np.array(img)
    img1 = cv2.copyMakeBorder(img,3,3,3,3,cv2.BORDER_CONSTANT)
    img2 = Image.fromarray(img1)
    array1 = np.zeros([7,7])

    #img2.show()
    for i in range(0,512):
        for j in range(0,512):
            img3 = cv2.equalizeHist(img1[i:i+7,j:j+7])
            img1[i-1,j-1] = img3[3,3]
    arr = img.flatten()
    arr1 = img1.flatten()
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.hist(arr, bins=256, density=1, facecolor='green')
    plt.subplot(223), plt.imshow(img1,'gray')
    plt.subplot(224), plt.hist(arr1, bins=256, density=1, facecolor='green')
    plt.show()
