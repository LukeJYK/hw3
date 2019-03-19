import cv2
from matplotlib import pyplot as plt
import numpy as np
def arrayToHist(grayArray,nums):
    if(len(grayArray.shape) != 2):
        print("length error")
        return None
    w,h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if(hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    #normalize
    n = w*h
    for key in hist.keys():
        hist[key] = float(hist[key])/n
    return hist
def drawHist(hist,name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist)-1#x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    #plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys),tuple(values))#绘制直方图
    #plt.show()
def histMatch(grayArray,h_d):
    #计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray,256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    #计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des


img = cv2.imread('C:\\Users\\lenovo\\Desktop\\3\\images\\lena.bmp',0)
#原始图和直方图
plt.subplot(2,3,1)
plt.title("original")
plt.imshow(img,cmap='gray')
plt.subplot(2,3,4)
hist_s = arrayToHist(img,256)
drawHist(hist_s,"original_hist")
im_match = cv2.imread('C:\\Users\\lenovo\\Desktop\\3\\images\\lena4.bmp',0)
#match图和其直方图
plt.subplot(2,3,2)
plt.title("match")
plt.imshow(im_match,cmap='gray')

plt.subplot(2,3,5)
hist_m = arrayToHist(im_match,256)
drawHist(hist_m,"match_hist")
#match后的图片及其直方图
im_d = histMatch(img,hist_m)#将目标图的直方图用于给原图做均衡，也就实现了match
plt.subplot(2,3,3)
plt.title("matched")
plt.imshow(im_d,cmap='gray')

plt.subplot(2,3,6)
hist_d = arrayToHist(im_d,256)
drawHist(hist_d,"matched_hist")



plt.show()