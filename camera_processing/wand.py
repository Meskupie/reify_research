import cv2
import numpy as np

from linalghelpers import lineEndPointsOnImage

class Marker():
	def __init__(self):
		pass

	def detect(self,img):
		raise NotImplementedError

	def visualizeMask(self,img):
		raise NotImplementedError
	
	# Hue (what colour), Saturation (white to colour) Value (dark to vibrant)
	def maskColor(self,img_hsv,hue_center,hue_radius,
		upper_sat=1.0,lower_sat=0.3,upper_val=1.0,lower_val=0.3):
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

#The pink aura
class ActiveBallMarker(Marker):
	def __init__(self, hue, hue_range, u_sat, m_sat, l_sat, u_bright, m_bright, l_bright):
		self.IMAGE_BLUR = 3
		self.MAX_BALL_RAD = 20
		self.CIRCULARITY_SENSITIVITY = 5 #1-5ish range
		self.PUPIL_BLUR = 5 # must be odd

		self.hue = hue
		self.hue_range = hue_range
		self.u_sat = u_sat
		self.m_sat = m_sat
		self.l_sat = l_sat
		self.u_bright = u_bright
		self.m_bright = m_bright
		self.l_bright = l_bright
		super().__init__()

	def prepareImage(self,img):
		blurred_img = cv2.GaussianBlur(img, (self.IMAGE_BLUR, self.IMAGE_BLUR), 0)
		hsv_img = cv2.cvtColor(blurred_img,cv2.COLOR_BGR2HSV)
		return hsv_img

	def getPupilMask(self,hsv_img):
		mask = self.maskColor(hsv_img,self.hue,self.hue_range,self.m_sat,self.l_sat,self.u_bright,self.m_bright)
		
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.PUPIL_BLUR,self.PUPIL_BLUR))
		mask = cv2.dilate(mask, kernel, iterations=1)
		mask = cv2.erode(mask, kernel, iterations=1)
		return mask

	def getCoronaMask(self,hsv_img):
		# Setting the kernel to be the max size of the ball ensures 
		# that the inner saturated ball gets gellied when dilating
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.MAX_BALL_RAD*2+1,self.MAX_BALL_RAD*2+1))
		mask = self.maskColor(hsv_img,self.hue,self.hue_range,self.u_sat,self.m_sat,self.m_bright,self.l_bright)
		mask = cv2.dilate(mask, kernel, iterations=1)
		mask = cv2.erode(mask, kernel, iterations=1)
		return mask

	def visualizeMask(self,img):
		hsv_img = self.prepareImage(img)
		pupil_mask = self.getPupilMask(hsv_img)
		corona_mask = self.getCoronaMask(hsv_img)
		has_mask = np.logical_or(pupil_mask > 0, corona_mask > 0)
		img[has_mask] = np.dstack((corona_mask, np.zeros(pupil_mask.shape), pupil_mask))[has_mask]

	def detect(self,img):
		hsv_img = self.prepareImage(img)
		pupil_mask = self.getPupilMask(hsv_img)
		corona_mask = self.getCoronaMask(hsv_img)
		marker_mask = np.minimum(pupil_mask,corona_mask)
		marker_mask = cv2.GaussianBlur(marker_mask, (self.PUPIL_BLUR, self.PUPIL_BLUR), 0)

		circles = cv2.HoughCircles(marker_mask, cv2.HOUGH_GRADIENT, self.CIRCULARITY_SENSITIVITY, 50,
			param1=100,param2=30,minRadius=0,maxRadius=self.MAX_BALL_RAD)
		if circles is not None:
			if len(circles) == 1:
				circles = np.round(circles[0, :]).astype("int")
				return circles[0]
		return None


class BallWand():
	def __init__(self):

		#NOTE: where should these hard codes be located and how should they be referenced?
		self.marker_distance = 0.4
		self.topMarker = ActiveBallMarker(hue=345, hue_range=35, 
			u_sat=0.2, m_sat=0.04, l_sat=0, u_bright=1, m_bright=0.96, l_bright=0.85)
		self.bottomMarker = ActiveBallMarker(hue=120, hue_range=40, 
			u_sat=0.2, m_sat=0.04, l_sat=0, u_bright=1, m_bright=0.96, l_bright=0.85)

	def detect(self,img):
		top_keypoints = self.topMarker.detect(img)
		bottom_keypoints = self.bottomMarker.detect(img)
		if top_keypoints is None or bottom_keypoints is None:
			return None

		keypoints = {'top':{'x':top_keypoints[0],'y':top_keypoints[1],'r':top_keypoints[2]},
					 'bottom':{'x':bottom_keypoints[0],'y':bottom_keypoints[1],'r':bottom_keypoints[2]}}
		return keypoints

	def draw(self,img,key):
		cv2.circle(img, (key['top']['x'],key['top']['y']), key['top']['r'], (0, 0, 255), 5)
		cv2.circle(img, (key['bottom']['x'],key['bottom']['y']), key['bottom']['r'], (0, 255, 0), 5)
		cv2.line(img,(key['top']['x'],key['top']['y']),(key['bottom']['x'],key['bottom']['y']), (255, 0, 0), 5)

	def visualizeMask(self,img):
		self.topMarker.visualizeMask(img)
		self.bottomMarker.visualizeMask(img)



