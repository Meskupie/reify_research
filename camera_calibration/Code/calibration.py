import cv2
import numpy as np
import glob
import os
import yaml

class IntrinsicCalibrator():
    def __init__(self,checkerShape=(9,6)):
        self.loaded = False
        self.resolution = None
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.checkerShape = checkerShape
        self.checkerPoints = np.zeros((self.checkerShape[0]*self.checkerShape[1],3), np.float32)
        self.checkerPoints[:,:2] = np.mgrid[0:self.checkerShape[0],0:self.checkerShape[1]].T.reshape(-1,2)

    def get_resolution(self,imgs):
        for img in imgs:
            if self.resolution is None:
                self.resolution = img.shape
            elif self.resolution == img.shape:
                continue
            else:
                return False
        #print(f'resolution {self.resolution}')
        return True


    def find_corners(self,img,refine=True):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.checkerShape[0],self.checkerShape[1]),None)
        if refine and ret:
            terminationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),terminationCriteria)
        return ret, corners

    def checker_points_from_image_list(self,imgs):
        #TODO: Check if all the images are the same size, if they are not, calibration will fail
        #TODO: check the number of images, need at least 3
        objectPoints = [] # 3d point in real world space
        imagePoints = [] # 2d points in image plane.
        for img in imgs:
            ret, corners = self.find_corners(img)
            if ret:
                objectPoints.append(self.checkerPoints)
                imagePoints.append(corners)

        #TODO: Check if enough images were correctly processed for calibration to suceed
        #print('Processed {}of{} checkerboard images'.format(len(objectPoints),len(imgs)))
        return objectPoints,imagePoints

    def calculate_error (self,imageKeypoints,objectKeypoints,mtx,dist,rvecs,tvecs):
        total_error = 0
        for i in range(len(objectKeypoints)):
            image_keypoints2, _ = cv2.projectPoints(objectKeypoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imageKeypoints[i],image_keypoints2, cv2.NORM_L2)/len(image_keypoints2)
            total_error += error
        return total_error/len(objectKeypoints)

    def save_camera_params(self,paramPath,cameraMatrix,newCameraMatrix,distortionParams,fov,calibrationError):
        #TODO: Add resolution to file and add code to verify that the same resolution is being used as intended
        paramObject = {'cameraMatrix':cameraMatrix.tolist(),'newCameraMatrix':newCameraMatrix.tolist(),
            'distortionParams':distortionParams.tolist(),'fov':fov,'calibrationError':calibrationError}
        with open(paramPath, 'w') as file:
            documents = yaml.dump(paramObject, file)

    def load_camera_params(self,paramPath):
        #TODO: Do actual checking on this file to make sure that we loaded it correctly
        try:
            with open(paramPath) as file:
                paramObject = yaml.full_load(file)

            for key, value in paramObject.items():
                if key == 'cameraMatrix':
                    cameraMatrix = np.array(value)
                elif key == 'newCameraMatrix':
                    newCameraMatrix = np.array(value)
                elif key == 'distortionParams':
                    distortionParams = np.array(value)
                elif key == 'fov':
                    self.fov = value
                elif key == 'calibrationError':
                    calibrationError = value
                else:
                    raise RuntimeError('Unknown key when reading camera YAML')

            self.mapx, self.mapy = cv2.initUndistortRectifyMap(cameraMatrix,distortionParams,None,newCameraMatrix,self.resolution[:2][::-1],5)
            self.loaded = True
            return True

        except:
            print('File reading is broken')
            return False

    def calibrate_from_corners_list(self,paramPath,imageKeypoints,resolution):
        self.resolution = resolution
        objectKeypoints = []
        for i in imageKeypoints:
            objectKeypoints.append(self.checkerPoints)
        _, mtx, dist, rotationVectors, translationVectors = cv2.calibrateCamera(
            objectKeypoints, imageKeypoints,self.resolution[:2][::-1],None,None)
        # 4th arg determines how cropping occurs. 0 means scale in, 1 means scale out
        newCameraMatrix, fov = cv2.getOptimalNewCameraMatrix(mtx,dist,self.resolution[:2][::-1],1,self.resolution[:2][::-1])
        calibrationError = self.calculate_error(
            imageKeypoints,objectKeypoints,mtx,dist,rotationVectors,translationVectors)
        self.save_camera_params(paramPath,mtx,newCameraMatrix,dist,fov,calibrationError)
        self.load_camera_params(paramPath)


    def calibrate_from_image_list(self,paramPath,imgs):
        #TODO add a return true or false piped from various steps
        self.get_resolution(imgs)
        objectKeypoints, imageKeypoints = self.checker_points_from_image_list(imgs)
        _, mtx, dist, rotationVectors, translationVectors = cv2.calibrateCamera(
            objectKeypoints, imageKeypoints,self.resolution[:2][::-1],None,None)
        # 4th arg is the free scaleing param, 0 means scale in, 1 means scale out
        newcameramtx, fov = cv2.getOptimalNewCameraMatrix(mtx,dist,self.resolution[:2][::-1],0)
        calibrationError = self.calculate_error(imageKeypoints,objectKeypoints,mtx,dist,rotationVectors,translationVectors)
        self.save_camera_params(paramPath,mtx,newcameramtx,dist,fov,calibrationError)
        self.load_camera_params(paramPath)
    
    def calibrate_from_image_dir(self,paramPath,dirPath):
        dirPath = os.path.join(dirPath,'checker_*.jpg')
        imgPaths = glob.glob(dirPath)
        imgs = []
        for imgPath in imgPaths:
            imgs.append(cv2.imread(imgPath))

        self.calibrate_from_image_list(paramPath,imgs)

    def apply_intrinsic_transform(self,img):
        # TODO: need to look into the ordering of self.fov coming from 'cv2.getOptimalNewCameraMatrix'
        # undistort and crop the image
        print(img.shape)
        if self.loaded:
            out = cv2.remap(img,self.mapx,self.mapy,cv2.INTER_LINEAR)
            # print(out.shape)
            # x,y,w,h = self.fov
            # out = out[y:y+h, x:x+w]
            # print(out.shape)
            return out
        else:
            return None

    def draw_checkerboard(self,img):
        ret, corners = self.find_corners(img)
        cv2.drawChessboardCorners(img, (self.checkerShape[0],self.checkerShape[1]), corners,True)






class ExtrinsicCalibrator():
    def __init__(self):
        pass
