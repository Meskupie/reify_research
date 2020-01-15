import cv2
import numpy as np
import glob
import os
import yaml

class IntrinsicCalibrator():
    def __init__(self,checkerShape=(9,6)):
        self.loaded = False
        self.resolution = None
        self.cropBalance = 0
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.checkerShape = checkerShape
        self.checkerPoints = np.zeros((1,self.checkerShape[0]*self.checkerShape[1],3), np.float32)
        self.checkerPoints[0,:,:2] = np.mgrid[0:self.checkerShape[0],0:self.checkerShape[1]].T.reshape(-1,2)

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

    def save_camera_params(self,paramPath,cameraMatrix,newCameraMatrix,distortionCoefficients,):
        #TODO: Add resolution to file and add code to verify that the same resolution is being used as intended
        paramObject = {'cameraMatrix':cameraMatrix.tolist(),'newCameraMatrix':newCameraMatrix.tolist(),
            'distortionCoefficients':distortionCoefficients.tolist()}
        with open(paramPath, 'w') as file:
            yaml.dump(paramObject, file)

    def load_camera_params(self,paramPath):
        #TODO: Do actual checking on this file to make sure that we loaded it correctly
        cameraMatrix = np.zeros((3, 3))
        newCameraMatrix = np.zeros((3, 3))
        distortionCoefficients = np.zeros((4, 1))
        try:
            with open(paramPath) as file:
                paramObject = yaml.full_load(file)

            for key, value in paramObject.items():
                if key == 'cameraMatrix':
                    cameraMatrix = np.array(value)
                elif key == 'newCameraMatrix':
                    newCameraMatrix = np.array(value)
                elif key == 'distortionCoefficients':
                    distortionCoefficients = np.array(value)
                else:
                    raise RuntimeError('Unknown key when reading camera YAML')

            self.mapx, self.mapy = cv2.fisheye.initUndistortRectifyMap(
                cameraMatrix,
                distortionCoefficients,
                np.eye(3),
                newCameraMatrix,
                self.resolution[:2][::-1],
                cv2.CV_16SC2
            )
            self.loaded = True
            return True

        except:
            print('File reading is broken')
            return False

    def calibrate_from_corners_list(self,paramPath,imageKeypoints,resolution):
        self.resolution = resolution

        nSamples = len(imageKeypoints)
        cameraMatrix = np.zeros((3, 3))
        distortionCoefficients = np.zeros((4, 1))
        rotationVectors = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(nSamples)]
        translationVectors = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(nSamples)]
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW #+cv2.fisheye.CALIB_CHECK_COND
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            [self.checkerPoints]*nSamples,
            imageKeypoints,
            self.resolution[:2][::-1],
            cameraMatrix,
            distortionCoefficients,
            rotationVectors,
            translationVectors,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        # Note: see this link if calibration resolution and undistortion resolution are different
        # https://gist.github.com/mesutpiskin/0412c44bae399adf1f48007f22bdd22d
        newCameraMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            cameraMatrix,
            distortionCoefficients,
            self.resolution[:2][::-1],
            np.eye(3), balance=self.cropBalance
        )
        self.save_camera_params(paramPath,cameraMatrix,newCameraMatrix,distortionCoefficients)
        self.load_camera_params(paramPath)

    def calibrate_from_video(self,videoPath,paramPath):
        pass

    def apply_intrinsic_transform(self,img):
        # TODO: need to look into the ordering of self.fov coming from 'cv2.getOptimalNewCameraMatrix'
        # undistort and crop the image
        print(img.shape)
        if self.loaded:
            out = cv2.remap(img, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
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
