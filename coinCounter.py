import numpy as np
import cv2
from matplotlib import pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk


def loadImage(image_Path):
    '''
    加载图片
    :param image_Path: 图片路径
    :return: 返回加载到的图片
    '''
    img = cv2.imread(image_Path)
    return img


def showImg(image, title_str=""):
    '''
    展示图片
    :param image: 图片对象
    :param title_str: 图片标题，默认为空
    '''
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.title(title_str)
    plt.show()


def saveImage(image, str, path='pic/'):
    '''
    保存图片
    :param image: 图片对象
    :param str: 文件名
    :param path: 保存路径
    '''
    cv2.imwrite(path + str + '.jpg', image)


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
    # 设置霍夫圆检测参数（根据情况进行参数调整）
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


def countValue(circles, circles_gold=None):
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
                if abs(circle[0] - circle_gold[0]) <= 20 and abs(circle[1] - circle_gold[1]) <= 20:
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
    # 设置颜色过滤区间（根据情况进行参数设置）
    lower_hsv = np.array([0, 180, 80])
    upper_hsv = np.array([255, 255, 170])
    # 掩膜
    mask = cv2.inRange(image_HSV, lower_hsv, upper_hsv)
    res = cv2.bitwise_and(image_BGR, image_BGR, mask=mask)
    return res


def coinCount(coin_image, i, isVisualize=False, isSave=True):
    '''
    检测图像中的金额
    :param coin_image: 原图像
    :param i: 图像序号
    :param isVisualize: 是否要进行可视化
    :param isSave: 是否要保存
    :return: 金额
    '''
    if isVisualize:
        showImg(coin_image, 'the image')
    img_Gold = detectGold(coin_image)
    circles_gold, img_circles_glod = detectCircles(img_Gold, i, isSave=isSave, name="gold")
    circles, img_circles = detectCircles(coin_image, i, circles_gold=circles_gold, isSave=isSave)
    if isVisualize:
        showImg(img_circles)
        if img_circles_glod:
            showImg(img_circles_glod)
    if circles_gold is None:
        value = countValue(circles)
    else:
        value = countValue(circles, circles_gold=circles_gold)
    return round(value, 1)


# 界面
window = tk.Tk()
window.title('硬币检测')
window.geometry('1200x600')


def paintResource():
    canvas_img_resource = tk.Canvas(window, bg='white', height=460, width=345)
    canvas_img_resource.place(x=200, y=10, anchor='nw')
    img_resource = Image.open(image_path.get())
    img_resource_resize = img_resource.resize((345, 460))
    photo = ImageTk.PhotoImage(img_resource_resize)
    image = canvas_img_resource.create_image(0, 0, anchor='nw', image=photo)
    window.mainloop()


def paintProcess():
    canvas_img_precess = tk.Canvas(window, bg='white', height=460, width=345)
    canvas_img_precess.place(x=680, y=10, anchor='nw')
    wifi_img = Image.open(image_path.get().split('/')[0] + '/image_circles' + image_path.get().split('/')[1])
    wifi_img1 = wifi_img.resize((345, 460))
    photo = ImageTk.PhotoImage(wifi_img1)
    image = canvas_img_precess.create_image(0, 0, anchor='nw', image=photo)
    window.mainloop()


def hit_me():
    global on_hit
    # 从 False 状态变成 True 状态
    if on_hit == False:
        on_hit = True
        i = image_path.get().split('/')[1].split('.')[0]
        img = loadImage(image_path.get())
        count_value = coinCount(img, i)
        # 显示value数据
        value.set('总金额：' + str(count_value))
        paintProcess()
    else:
        # 从 True 状态变成 False 状态
        on_hit = False
        # 设置文字为空
        value.set('')


# 图像路径输入框
image_path = tk.Entry(window,
                      font=('Arial', 15))
image_path.place(x=330, y=470)

# 图像路径输入框标签
image_path_label = tk.Label(window,
                            text=" 图像存储路径",
                            font=('Arial', 12))

image_path_label.place(x=220, y=470)

button_resource = tk.Button(window,
                            # 显示在按钮上的文字
                            text='导入原始图像',
                            width=18, height=2,
                            # 点击按钮式执行的命令
                            command=paintResource)
button_resource.place(x=220, y=510, anchor='nw')

# 文字变量储存器
value = tk.StringVar()
value_area = tk.Label(window,
                      # 使用 textvariable 替换 text, 因为这个可以变化
                      textvariable=value,
                      bg='pink', font=('Arial', 20), width=15, height=2)
value_area.place(x=750, y=490, anchor='nw')

button_detect = tk.Button(window,
                          # 显示在按钮上的文字
                          text='硬币检测',
                          width=18, height=2,
                          # 点击按钮式执行的命令
                          command=hit_me)
# 按钮位置
button_detect.place(x=400, y=510, anchor='nw')
# 默认初始状态为 False
on_hit = False

window.mainloop()
