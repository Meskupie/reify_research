import cv2
import numpy as np
import glob
import os

class IntrinsicCalibrator():
	def __init__(self,checker_shape=(9,6)):
		self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		
		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		self.checker_shape = checker_shape
		self.checker_points = np.zeros((self.checker_shape[0]*self.checker_shape[1],3), np.float32)
		self.checker_points[:,:2] = np.mgrid[0:self.checker_shape[1],0:self.checker_shape[0]].T.reshape(-1,2)

	def checkerPointsFromImageList(self,imgs):
		#TODO: Check if all the images are the same size, if they are not, calibration will fail
		#TODO: check the number of images, need at least 3
		object_points = [] # 3d point in real world space
		image_points = [] # 2d points in image plane.
		for img in imgs:
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (CHECKER_GRID[1],CHECKER_GRID[0]),None)
		    if ret == True:
		        corners_refined = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.term_criteria)
		        object_points.append(self.checker_points)
		        image_points.append(corners_refined)

		#TODO: Check if enough images were correctly processed for calibration to suceed
        #print('Processed {}of{} checkerboard images'.format(len(object_points),len(imgs)))
        return object_points,image_points

    def calculate
    	object_keypoints, image_keypoints = self.checkerPointsFromImageList(imgs)

    def paramFromImageList(self,imgs):


	def paramFromImageDir(self,dir_path):
		img_paths = glob.glob(dir_path) #dirpath = os.path.join(BASEPATH,'checker_*.jpg')
		imgs = []
		for img_path in img_paths:
			imgs.append(cv2.imread(img_path))

		paramFromImageList(imgs)
		
		
		img = cv2.drawChessboardCorners(img, (self.checker_shape[1],self.checker_shape[0]), corners_refined,ret)



def applyIntrinsicTransform(img,mapx,mapy,roi):
    # undistort and crop the image
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst





import time

img = cv2.imread(os.path.join(BASEPATH,'pose_0.jpg'))

start_time = time.time()
for i in range(1000):
    result = applyIntrinsicTransform(img,mapx,mapy,roi)
print("Average processing time: {}s".format((time.time()-start_time)/1000))






class ExtrinsicCalibrator():
	def __init__(self):
		pass