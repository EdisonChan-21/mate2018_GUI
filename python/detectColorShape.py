from scipy.spatial import distance as dist
import imutils
import cv2
import numpy as np
import time

def colorDetect(image):
    # red
    lower_red = np.array([0,100,100])
    upper_red = np.array([13,255,255])
    lower_red1 = np.array([150,100,100])
    upper_red1 = np.array([180,255,255])
    # yellow
    lower_yellow = np.array([15, 0, 0])
    upper_yellow = np.array([45, 255, 255])
    # blue
    lower_blue = np.array([85,80,50])
    upper_blue = np.array([135,255,255])
    mask = []
    color = ["red","yellow","blue"]
    resized = imutils.resize(image, width=600)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    cntsList = []
    couterList = []

    #mask
    mask.append(cv2.inRange(hsv, lower_red, upper_red)+cv2.inRange(hsv, lower_red1, upper_red1))
    mask.append(cv2.inRange(hsv, lower_yellow, upper_yellow))
    mask.append(cv2.inRange(hsv, lower_blue, upper_blue))
##    cv2.imshow("red", mask[0])
##    cv2.waitKey(0)
##    cv2.imshow("yellow", mask[1])
##    cv2.waitKey(0)
    cv2.imshow("blue", mask[2])
    cv2.waitKey(0)

    for i in range(3):
        kernelOpen=np.ones((5,5))
        kernelClose=np.ones((20,20))
        maskOpen=cv2.morphologyEx(mask[i],cv2.MORPH_OPEN,kernelOpen)
        maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
        blurred = cv2.GaussianBlur(maskClose, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        im2,contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        try:
            c = max(contours, key = cv2.contourArea)
            cntsList.append(c)
            couterList.append(i)
        except:
            pass
    print(cntsList)
    if (cntsList == []):
        return "unidentified",[]
    else:
        cMax = max(cntsList, key = cv2.contourArea)
        return color[couterList[cntsList.index(cMax)]],cMax

def shapeDetect(self, c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "rectangle"

    # return the name of the shape
    return shape

def detectColorShape(image):
    colorList = ["red","yellow","blue"]
    colorValue = [(0, 0, 255),(0, 255, 255),(255, 0, 0)]
    shapeList = ["triangle","rectangle"]
    labelList = [["A","B","C"],["D","E","F"]]
    shape = "unidentified"
    resized = imutils.resize(image, width=600)
    ratio = image.shape[0] / float(resized.shape[0])
    color,cnts = colorDetect(image)
    print("color is ",color)
    if (color=="unidentified"):
        return False
##    print(color)
##    print(cnts)
    # kernelOpen=np.ones((5,5))
    # kernelClose=np.ones((20,20))
    # maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    # maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    # blurred = cv2.GaussianBlur(maskClose, (5, 5), 0)
    # thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE)
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
    shape = shapeDetect(resized,cnts)
    print("shape is ",shape)
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    if(shape in shapeList):
        cnts = cnts.astype("float")
        cnts *= ratio
        cnts = cnts.astype("int")
        cv2.drawContours(image, [cnts], -1, (0, 255, 0), 2)
        cv2.putText(image, labelList[shapeList.index(shape)][colorList.index(color)], (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
            4, colorValue[colorList.index(color)], 2)
    if(shape=="unidentified"):
            return False
    imageName = str('photo/' + time.strftime("%Y_%m_%d_%H_%M")) + '.jpg'
    cv2.imwrite(imageName,image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return True
