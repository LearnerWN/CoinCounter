
import numpy as np
import cv2
import skimage.filters as filter
from scipy import ndimage as ndi
import math
from matplotlib import pyplot as plt


def showImg(image, titlestr=""):
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.title(titlestr)
    plt.show()

def detectCircles(image, i, isSave= True):
    # 输出图像大小，方便根据图像大小调节minRadius和maxRadius
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=100, param2=30, minRadius=5, maxRadius=400)
    # 输出检测到圆的个数
    # print(len(circles[0]))
    # 根据检测到圆的信息，画出每一个圆
    image_circles = image
    for circle in circles[0]:
        # 圆的基本信息
        # print(circle[2])
        # 坐标行列
        x = int(circle[0])
        y = int(circle[1])
        # 半径
        r = int(circle[2])
        # 在原图用指定颜色标记出圆的位置
        image_circles = cv2.circle(image, (x, y), r, (0, 0, 255), -1)

        if isSave:
            imageName = "image_circles" + str(i)
            saveImage(image_circles, imageName)

    return circles, image_circles

def countValue(circles):
    count = 0
    # 根据圆的半径区分面额并计数
    for circle in circles[0]:
        radius = int(circle[2])
        if radius >= 290:
            count = count + 1
        elif radius >= 240:
            count = count + 0.5
        elif radius >= 220:
            count = count + 0.1
    return count

def saveImage(image, str):
    cv2.imwrite('pic/' + str +'.jpg', image)

def detectGold(image_BGR):
    image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([14, 43, 46])
    upper_hsv = np.array([34, 255, 255])
    mask = cv2.inRange(image_HSV, lower_hsv, upper_hsv)
    return mask


def coinCount(coinImage, i, isVisualize=False, isSave=True):
    if isVisualize:
        showImg(coinImage, 'the image')
    circles, img_circles = detectCircles(coinImage, i, isSave=isSave)
    print("num of circles:",circles)
    if isVisualize:
        showImg(img_circles)
    value = countValue(circles)
    return round(value, 1)

# 测试代码
img = cv2.imread('pic/10.jpg')
mask = detectGold(img)
showImg(mask)
# value = coinCount(img,10)
# print("value of the image:{value}".format(value=value))