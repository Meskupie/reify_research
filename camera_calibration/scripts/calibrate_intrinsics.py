cameraNames = ['poc_1','poc_2','poc_3','poc_4']
calibrationDir = '../calibration_data'
imageExtension = '.png'
displayResolution = (1280,720)

import os
import cv2
import sys
import glob

sys.path.insert(1, '../code')
from calibration import IntrinsicCalibrator

basePath = os.path.abspath('')
calibrationPath = os.path.join(basePath,calibrationDir)
calibrator = IntrinsicCalibrator()

for cameraName in cameraNames:
    paramPath = os.path.join(calibrationPath,cameraName,cameraName+'.yaml')
    imgPaths = glob.glob(os.path.join(calibrationPath,cameraName,'*'+imageExtension))
    calibrator.calibrate_from_images(paramPath, imgPaths)
    calibrator.load_params(paramPath)

    img = cv2.imread(os.path.join(calibrationPath,cameraName,'test'+imageExtension))
    assert img is not None, "Need a test image named test.{extension}"
    out = calibrator.undistort_image(img)
    cv2.imshow('frame',out)
    while(True):
        key = cv2.waitKey(1)
        if key == 32: #spacebar
            break
