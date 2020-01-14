import cv2
import numpy as np
import glob
import os
import yaml

class IntrinsicCalibrator():
	def __init__(self,checker_shape=(9,6)):
		self.loaded = False
		self.resolution = None
		
		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		self.checker_shape = checker_shape
		self.checker_points = np.zeros((self.checker_shape[0]*self.checker_shape[1],3), np.float32)
		self.checker_points[:,:2] = np.mgrid[0:self.checker_shape[0],0:self.checker_shape[1]].T.reshape(-1,2)

		self.termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		

	def getResolution(self,imgs):
		for img in imgs:
			if self.resolution is None:
				self.resolution = img.shape
			elif self.resolution == img.shape:
				continue
			else:
				return False
		print(self.resolution)
		return True

	def findCorners(self,img,refine=True):
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (self.checker_shape[0],self.checker_shape[1]),None)
		if refine & ret:
			corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.termination_criteria)
		return ret, corners

	def checkerPointsFromImageList(self,imgs):
		#TODO: Check if all the images are the same size, if they are not, calibration will fail
		#TODO: check the number of images, need at least 3
		object_points = [] # 3d point in real world space
		image_points = [] # 2d points in image plane.
		for img in imgs:
			ret, corners = self.findCorners(img)
			if ret:
				object_points.append(self.checker_points)
				image_points.append(corners)

		#TODO: Check if enough images were correctly processed for calibration to suceed
		#print('Processed {}of{} checkerboard images'.format(len(object_points),len(imgs)))
		return object_points,image_points

	def calculateError (self,image_keypoints,object_keypoints,mtx,dist,rvecs,tvecs):
		total_error = 0
		for i in range(len(object_keypoints)):
			image_keypoints2, _ = cv2.projectPoints(object_keypoints[i], rvecs[i], tvecs[i], mtx, dist)
			error = cv2.norm(image_keypoints[i],image_keypoints2, cv2.NORM_L2)/len(image_keypoints2)
			total_error += error
		return total_error/len(object_keypoints)

	def saveCameraParams(self,param_path,camera_matrix,new_camera_matrix,distortion_params,fov,calibration_error):
		#TODO: Add resolution to file and add code to verify that the same resolution is being used as intended
		param_object = {'cam_matrix':camera_matrix.tolist(),'new_cam_matrix':new_camera_matrix.tolist(),
			'distortion_params':distortion_params.tolist(),'fov':fov,'calibration_error':calibration_error}
		with open(param_path, 'w') as file:
			documents = yaml.dump(param_object, file)

	def loadCameraParams(self,param_path):
		#TODO: Do actaull checking on this file to make sure that we loaded it correctly
		try:
			with open(param_path) as file:
				param_object = yaml.full_load(file)

			for key, value in param_object.items():
				if key == 'cam_matrix': cam_matrix = np.array(value)
				elif key == 'new_cam_matrix': new_cam_matrix = np.array(value)
				elif key == 'distortion_params': distortion_params = np.array(value)
				elif key == 'fov': self.fov = value
				elif key == 'calibration_error': calibration_error = value
				else: print('Unknown key when reading camera YAML')

			self.mapx, self.mapy = cv2.initUndistortRectifyMap(cam_matrix,distortion_params,None,new_cam_matrix,self.resolution[:2],5)
			self.loaded = True
			return True

		except:
			print('File reading is broken')
			return False

	def paramsFromImageList(self,param_path,imgs):
		#TODO add a return true or false piped from various steps
		self.getResolution(imgs)
		object_keypoints, image_keypoints = self.checkerPointsFromImageList(imgs)
		_, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_keypoints, image_keypoints, (self.resolution[1],self.resolution[0]),None,None)
		# 4th arg is the free scaleing param, 0 means scale in, 1 means scale out
		newcameramtx, fov = cv2.getOptimalNewCameraMatrix(mtx,dist,(self.resolution[1],self.resolution[0]),0)
		calibration_error = self.calculateError(image_keypoints,object_keypoints,mtx,dist,rvecs,tvecs)
		self.saveCameraParams(param_path,mtx,newcameramtx,dist,fov,calibration_error)
		self.loadCameraParams(param_path)
	
	def applyIntrinsicTransform(self,img):
		# TODO: need to look into the ordering of self.fov comming from 'cv2.getOptimalNewCameraMatrix'
		# undistort and crop the image
		print(img.shape)
		if self.loaded:
			out = cv2.remap(img,self.mapx,self.mapy,cv2.INTER_LINEAR)
			print(out.shape)
			x,y,w,h = self.fov
			out = out[y:y+h, x:x+w]
			print(out.shape)
			return out
		else:
			return None

	def paramsFromImageDir(self,param_path,dir_path):
		dir_path = os.path.join(dir_path,'checker_*.jpg')
		img_paths = glob.glob(dir_path)
		imgs = []
		for img_path in img_paths:
			imgs.append(cv2.imread(img_path))

		self.paramsFromImageList(param_path,imgs)

	def drawCheckerboard(self,img):
		ret, corners = self.findCorners(img)
		cv2.drawChessboardCorners(img, (self.checker_shape[0],self.checker_shape[1]), corners,True)






class ExtrinsicCalibrator():
	def __init__(self):
		pass