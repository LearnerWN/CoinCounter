
import numpy as np
import cv2
from matplotlib import pyplot as plt

def loadImage(imagePath):
    '''
    加载图片
    :param imagePath: 图片路径
    :return: 返回加载到的图片
    '''
    img = cv2.imread(imagePath+'.jpg')
    return img

def showImg(image, titlestr=""):
    '''
    展示图片
    :param image: 图片对象
    :param titlestr: 图片标题，默认为空
    '''
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.title(titlestr)
    plt.show()

def saveImage(image, str, path='pic/'):
    '''
    保存图片
    :param image: 图片对象
    :param str: 文件名
    :param path: 保存路径
    '''
    cv2.imwrite(path+ str +'.jpg', image)

def detectCircles(image, i, circles_gold=None, isSave=True, name=""):
    '''
    检测图片中圆的个数
    :param image: 待检测的图片对象
    :param i: 图像的序号
    :param circles_gold: 五毛钱的位置信息
    :param isSave: 是否保存图像
    :param name: 需要保存的图片名称
    :return: 返回检测到的圆形信息和图片对象
    '''
    # 输出图像大小，方便根据图像大小调节minRadius和maxRadius
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=100, param2=30, minRadius=5, maxRadius=400)
    # 输出检测到圆的个数
    print(circles)
    # 根据检测到圆的信息，画出每一个圆
    image_circles = image
    # 判断有没有检测到圆
    if circles is None:
        return None, None
    else:
        # 判断有没有黄金圆（五毛钱）
        if circles_gold is None:
            for circle in circles[0]:
                # 圆的基本信息
                # print(circle[2])
                # 坐标行列
                x = int(circle[0])
                y = int(circle[1])
                # 半径
                r = int(circle[2])
                print("radius:", r)
                # 在原图用指定颜色标记出不同面额的位置
                if r >= 290:
                    image_circles = cv2.circle(image, (x, y), r, (0, 0, 255), 15)
                    cv2.putText(image_circles, "1", (x, y), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 255), 10)
                elif r >= 220:
                    image_circles = cv2.circle(image, (x, y), r, (255, 255, 0), 15)
                    cv2.putText(image_circles, "0.1", (x, y), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 255), 10)

                if isSave:
                    imageName = "image_circles" + name + str(i)
                    saveImage(image_circles, imageName)
        else:
            for circle in circles[0]:
                x = int(circle[0])
                y = int(circle[1])
                r = int(circle[2])
                print("radius:", r)
                for circle_gold in circles_gold[0]:
                    # 将掩膜前与掩膜后的圆心做差，横纵坐标相差不超过20时，该圆形为五毛
                    if abs(circle[0] - circle_gold[0]) <= 20 and abs(circle[1] - circle_gold[1]) <= 20:
                        image_circles = cv2.circle(image, (x, y), r, (0, 255, 0), 15)
                        cv2.putText(image_circles, "0.5", (x, y), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 255), 10)
                        # 绘制完五毛钱区域以后将其半径置0 防止对后续的半径判断产生影响
                        r = 0
                if r >= 290:
                    image_circles = cv2.circle(image, (x, y), r, (0, 0, 255), 15)
                    cv2.putText(image_circles, "1", (x, y), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 255), 10)
                elif r >= 220:
                    image_circles = cv2.circle(image, (x, y), r, (255, 255, 0), 15)
                    cv2.putText(image_circles, "0.1", (x, y), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 255), 10)

                if isSave:
                    imageName = "image_circles" + name + str(i)
                    saveImage(image_circles, imageName)
        return circles, image_circles

def countValue(circles,circles_gold=None):
    '''
    计算金额总数
    :param circles: 整个图片中所有的圆形信息
    :param circles_gold: 五毛钱的圆形信息
    :return: 金额
    '''
    count = 0
    # 根据硬币的半径和颜色区分面额并计数
    # 判断有没有五毛钱
    if circles_gold is None:
        for circle in circles[0]:
            radius = int(circle[2])
            if radius >= 290:
                count = count + 1
            elif radius >= 220:
                count = count + 0.1
    else:
        for circle in circles[0]:
            for circle_gold in circles_gold[0]:
                if abs(circle[0]-circle_gold[0])<=20 and abs(circle[1]-circle_gold[1])<=20:
                    count = count + 0.5
                    circle[2] = 0
            radius = int(circle[2])
            if radius >= 290:
                count = count + 1
            elif radius >= 220:
                count = count + 0.1
    return count

def detectGold(image_BGR):
    '''
    对图像做掩膜处理去除掉五毛钱意外的图像信息
    :param image_BGR: 待检测的图像
    :return: 进行掩膜处理后的图像
    '''
    # 将bgr图像转换为HSV
    image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
    # 设置颜色过滤区间
    lower_hsv = np.array([0, 180, 80])
    upper_hsv = np.array([255, 255, 170])
    # 掩膜
    mask = cv2.inRange(image_HSV, lower_hsv, upper_hsv)
    res = cv2.bitwise_and(image_BGR, image_BGR, mask=mask)
    return res


def coinCount(coinImage, i, isVisualize=False, isSave=True):
    '''
    检测图像中的金额
    :param coinImage: 原图像
    :param i: 图像序号
    :param isVisualize: 是否要进行可视化
    :param isSave: 是否要保存
    :return: 金额
    '''
    if isVisualize:
        showImg(coinImage, 'the image')
    img_Gold= detectGold(coinImage)
    circles_gold, img_circles_glod = detectCircles(img_Gold, i, isSave=isSave, name="gold")
    circles, img_circles = detectCircles(coinImage, i, circles_gold=circles_gold, isSave=isSave)
    if isVisualize:
        showImg(img_circles)
        if img_circles_glod:
            showImg(img_circles_glod)
    if circles_gold is None:
        value = countValue(circles)
    else:
        value = countValue(circles, circles_gold=circles_gold)

    return round(value, 1)
