import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from PIL import ImageGrab

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
Pt= []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, Pt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)
		print (refPt[0], refPt[1])
		cv2.imshow("image", image)

	elif event == cv2.EVENT_RBUTTONDOWN:
		if (len(refPt) == 2):
			refPt.append((x, y))
			cv2.line(image, refPt[1], refPt[2], (0, 0, 255), 2)
			refD = dist.euclidean(refPt[0], refPt[1])
			D = dist.euclidean(refPt[1], refPt[2])
			refDx = refPt[1][0]-refPt[0][0]
			Dx = refPt[2][0]-refPt[1][0]
			print("Reference Distance: ",refD)
			realDis = 42/refD*D
			realDis = round(realDis,2)
			print("Length: ",realDis,"cm")
			distanceText = "Length: "+str(realDis)+"cm"
			cv2.putText(image,distanceText,refPt[2],cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

img = ImageGrab.grab(bbox=(400, 200, 1600, 900)) #x, y, w, h
img_np = np.array(img)
frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
# image = cv2.imread(frame)
image = frame
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
	# if the 'c' key is pressed, break from the loop
	elif key == ord("m"):
		if (len(refPt) == 2) and (len(Pt) == 1):
			print("fullfiled")
	elif key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it

# close all open windows
cv2.destroyAllWindows()
