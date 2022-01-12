import numpy as np
import cv2
from PIL import ImageGrab
import detectColorShape as dcs
import time

# fourcc = cv2.VideoWriter_fourcc('X','V','I','D') #you can use other codecs as well.
# vid = cv2.VideoWriter('record.avi', fourcc, 8, (500,490))
while(True):
    img = ImageGrab.grab(bbox=(400, 200, 1600, 900)) #x, y, w, h
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    clone = frame.copy()
    processSuccessful = dcs.detectColorShape(frame)
    if processSuccessful:
        imageNameA = str('photo/' + time.strftime("%Y_%m_%d_%H_%M")) +'_raw' + '.jpg'
        cv2.imwrite(imageNameA,clone)
        break
    time.sleep(0.5)

# vid.release()
cv2.destroyAllWindows()
