import os
import cv2
import sys
import glob
import argparse

sys.path.insert(1, '../code')
from calibration import IntrinsicCalibrator

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', required=True, help='Path to the camera folders')
ap.add_argument('-c', '--cameras', nargs='+', required=True, help='List of folder names')
ap.add_argument('-e', '--extension', help='Image extension', default='.png')
ap.add_argument('--height', help='Image height', default=720)
ap.add_argument('--width', help='Image width', default=1280)
args = vars(ap.parse_args())
cameraNames = args['cameras']
calibrationDir = args['dir']
imageExtension = args['extension']
displayResolution = (args['width'], args['height'])

basePath = os.path.abspath('')
calibrationPath = os.path.join(basePath, calibrationDir)
calibrator = IntrinsicCalibrator()

for cameraName in cameraNames:
    paramPath = os.path.join(calibrationPath, cameraName, cameraName + '.yaml')
    imgPaths = glob.glob(os.path.join(calibrationPath, cameraName, '*' + imageExtension))
    calibrator.calibrate_from_images(paramPath, imgPaths)
    calibrator.load_params(paramPath)

    img = cv2.imread(os.path.join(calibrationPath, cameraName, 'test' + imageExtension))
    assert img is not None, "Need a test image named test.{extension}"
    out = calibrator.undistort_image(img)
    cv2.putText(out, f'Test results for {cameraName} undistorsion. Press "space" to continue.', (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (179, 72, 19), 2, cv2.LINE_AA, False)
    cv2.imshow('frame', out)
    while (True):
        key = cv2.waitKey(1)
        if key == 32:  # spacebar
            break
