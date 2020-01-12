import cv2
import numpy as np

from linalghelpers import lineEndPointsOnImage

# #The pink
# TOP_HUE = 350
# TOP_RANGE = 30
# TOP_UPPER_SATURATION = 0.2
# TOP_LOWER_SATURATION = 0.03
# TOP_UPPER_BRIGHTNESS = 1.0
# TOP_LOWER_BRIGHTNESS = 0.9

#The pink aura
TOP_HUE = 345 #310-30
TOP_RANGE = 35
TOP_UPPER_SATURATION = 0.2
TOP_MIDDLE_SATURATION = 0.03
TOP_LOWER_SATURATION = 0.0
TOP_UPPER_BRIGHTNESS = 1.0
TOP_MIDDLE_BRIGHTNESS = 0.97
TOP_LOWER_BRIGHTNESS = 0.85

# # The yellow
# BOTTOM_HUE = 130
# BOTTOM_RANGE = 40
# BOTTOM_UPPER_SATURATION = 0.2
# BOTTOM_LOWER_SATURATION = 0.03
# BOTTOM_UPPER_BRIGHTNESS = 1.0
# BOTTOM_LOWER_BRIGHTNESS = 0.9

# The green aura
BOTTOM_HUE = 120 # 80-160
BOTTOM_RANGE = 40
BOTTOM_UPPER_SATURATION = 0.2
BOTTOM_MIDDLE_SATURATION = 0.03
BOTTOM_LOWER_SATURATION = 0.0
BOTTOM_UPPER_BRIGHTNESS = 1.0
BOTTOM_MIDDLE_BRIGHTNESS = 0.97
BOTTOM_LOWER_BRIGHTNESS = 0.85


# Hue (what colour), Saturation (white to colour) Value (dark to vibrant)
def maskColor(img_hsv, hue_center, hue_radius,
	upper_sat=1.0, lower_sat=0.3, upper_val=1.0, lower_val=0.3):
	#TODO: hue_rad > 180 = bad, darkness bounds, whiteness bounds
	# handle color rollaround
	if hue_center+hue_radius > 360:
		hue_center = hue_center-360
	if hue_center-hue_radius < 0:
		#above zero
		lower = (0,255*lower_sat,255*lower_val)
		upper = (int((hue_center+hue_radius)/2),255*upper_sat,255*upper_val)
		mask_part_1 = cv2.inRange(img_hsv,np.array(lower,dtype = "uint8"),np.array(upper,dtype = "uint8"))
		#below zero
		lower = (int((360+hue_center-hue_radius)/2),255*lower_sat,255*lower_val)
		upper = (179,255*upper_sat,255*upper_val)
		mask_part_2 = cv2.inRange(img_hsv,np.array(lower,dtype = "uint8"),np.array(upper,dtype = "uint8"))
		mask = mask_part_1 | mask_part_2
	else:  
		lower = (int((hue_center-hue_radius)/2),255*lower_sat,255*lower_val)
		upper = (int((hue_center+hue_radius)/2),255*upper_sat,255*upper_val)
		mask = cv2.inRange(img_hsv,np.array(lower,dtype = "uint8"),np.array(upper,dtype = "uint8"))
	return mask

def maskRod(hsv_img):
	#TODO: Tune "simplify", this is the amount of erosion/dilation on the color mask
	SIMPLIFY = 2
	# Run the color mask
	mask = maskColor(hsv_img,ROD_HUE,ROD_RANGE,
		ROD_UPPER_SATURATION,ROD_LOWER_SATURATION,ROD_UPPER_BRIGHTNESS,ROD_LOWER_BRIGHTNESS)
	# Simplify shapes
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(SIMPLIFY,SIMPLIFY))
	mask = cv2.erode(mask, kernel, iterations=1)
	mask = cv2.dilate(mask, kernel, iterations=1)
	return mask

def maskBall(hsv_img,hue,hue_range,upper_sat,lower_sat,upper_val,lower_val):
	#TODO: Tune "simplify", this is the amount of erosion/dilation on the color mask
	SIMPLIFY = 30

	mask = maskColor(hsv_img,hue,hue_range,upper_sat,lower_sat,upper_val,lower_val)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(SIMPLIFY,SIMPLIFY))
	mask = cv2.dilate(mask, kernel, iterations=1)
	mask = cv2.erode(mask, kernel, iterations=1)
	return mask

def maskTopBall(hsv_img):
	return maskBall(hsv_img,TOP_HUE,TOP_RANGE,
		TOP_UPPER_SATURATION,TOP_LOWER_SATURATION,TOP_UPPER_BRIGHTNESS,TOP_LOWER_BRIGHTNESS)

def maskBottomBall(hsv_img):
	return maskBall(hsv_img,BOTTOM_HUE,BOTTOM_RANGE,
		BOTTOM_UPPER_SATURATION,BOTTOM_LOWER_SATURATION,BOTTOM_UPPER_BRIGHTNESS,BOTTOM_LOWER_BRIGHTNESS)

def findRodLines(hsv_img):
	mask = maskRod(hsv_img)
	# Find edges and lines
	edges = cv2.Canny(mask,50,100,apertureSize = 3)
	lines = cv2.HoughLines(edges,1,np.pi/180,100)
	output = []
	try:
		for line in lines:
			rho,theta = line[0]
			end_points = lineEndPointsOnImage(rho, theta, (img.shape[0],img.shape[1]))
			output.append(end_points)
	except:
		pass
	return output

def findCircles(mask):
	MAX_BALL_RAD = 35
	CIRCULARITY_SENSITIVITY = 5 #1-5ish range

	circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, CIRCULARITY_SENSITIVITY, 50,
		param1=100,param2=30,minRadius=0,maxRadius=MAX_BALL_RAD)
	
	if circles is not None:
		circles = np.round(circles[0, :]).astype("int")
	return circles

def findTopCircle(hsv_img):
	mask = maskTopBall(hsv_img)
	circles = findCircles(mask)
	return circles

def findBottomCircle(hsv_img):
	mask = maskBottomBall(hsv_img)
	circles = findCircles(mask)
	return circles

def drawCircles(img,circles):
	if circles is not None:
		for (x, y, r) in circles:
			cv2.circle(img, (x, y), r, (0, 255, 0), 6)
			#cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

def drawLines(img,lines):
	if lines is not None:
		for line in lines:
			cv2.line(img,line[0],line[1],(0,0,255),2)
