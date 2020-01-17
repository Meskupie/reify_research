videosDir = '../sample_data/extrinsic_1'
calibrationDir = '../calibration_data'
displayResolution = (1280,720)

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(1, '../code')
from calibration import IntrinsicCalibrator

def tryNewFrame(capture,frame):
    frameGrabbed, newFrame = capture.read()
    if frameGrabbed:
        out = newFrame.copy()
    else:
        out = frame
    return out

def forwards(capture,count):
    currentPlace = capture.get(cv2.CAP_PROP_POS_FRAMES)
    newPlace = min(capture.get(cv2.CAP_PROP_FRAME_COUNT),currentPlace+(count-1))
    capture.set(cv2.CAP_PROP_POS_FRAMES,newPlace)

def backwards(capture,count):
    currentPlace = capture.get(cv2.CAP_PROP_POS_FRAMES)
    newPlace = max(0,currentPlace-(count+1))
    capture.set(cv2.CAP_PROP_POS_FRAMES,newPlace)

basePath = os.path.abspath('')
videosPath = os.path.join(basePath,videosDir)
calibrationPath = os.path.join(basePath,calibrationDir)

videosName = videosPath.split('/')[1]
videoPaths = glob.glob(os.path.join(videosPath, '*.mp4'))
assert len(videoPaths) > 0
cameraNames = [path.split('/')[-1].split('.')[0] for path in videoPaths]
cv2.namedWindow("Labeler")
# open videos and locate keyframes
cropLength = 999999999
startFrames = []
for i in range(len(videoPaths)):
    cameraName = cameraNames[i]
    videoInPath = videoPaths[i]
    videoIn = cv2.VideoCapture(videoInPath)
    frameHeight = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
    ret, frame = videoIn.read()
    assert ret, "Cant find video"
    prevFrame = np.zeros(frame.shape)
    while(True):
        key = cv2.waitKey(1)
        if key == ord("a"): # Back one frame
            backwards(videoIn,1)
            frame = tryNewFrame(videoIn,frame)
            #cv2.imshow('Labeler',frame)
        elif key == ord("d"): # Forward one frame
            forwards(videoIn,1)
            frame = tryNewFrame(videoIn,frame)
            #cv2.imshow('Labeler',frame)
        elif key == ord("q"): # Back 10 frames
            backwards(videoIn,10)
            frame = tryNewFrame(videoIn,frame)
            #cv2.imshow('Labeler',frame)
        elif key == ord("e"): # Forwards 10 frames
            forwards(videoIn,10)
            frame = tryNewFrame(videoIn,frame)
             #cv2.imshow('Labeler',frame)
        elif key == 32: # space bar
            start = videoIn.get(cv2.CAP_PROP_POS_FRAMES)
            total = videoIn.get(cv2.CAP_PROP_FRAME_COUNT)
            cropLength = int(min(cropLength,total-start))
            startFrames.append(start)
            break
        elif key == -1:
            pass
        else:
            print(f'Getting key: {key}')
        # If frame has changed, display it
        if prevFrame.shape != frame.shape:
            frame = cv2.resize(frame,displayResolution,interpolation = cv2.INTER_AREA)
            prevFrame = frame.copy()
            cv2.imshow('Labeler',frame)
        elif not np.array_equal(prevFrame,frame):
            prevFrame = frame.copy()
            cv2.imshow('Labeler',frame)
    videoIn.release()

for i in range(len(videoPaths)):
    cameraName = cameraNames[i]
    videoInPath = videoPaths[i]
    videoIn = cv2.VideoCapture(videoInPath)
    videoIn.set(cv2.CAP_PROP_POS_FRAMES,startFrames[i]-1)
    fps = videoIn.get(cv2.CAP_PROP_FPS)
    frameWidth = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoOutPath = os.path.join(videosPath,videosName+'_'+cameraName+'.avi')
    videoOut = cv2.VideoWriter(videoOutPath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                        (frameWidth, frameHeight))
    paramPath = os.path.join(calibrationPath,cameraName,cameraName+'.yaml')
    calibrator = IntrinsicCalibrator()
    calibrator.load_params(paramPath)
    for j in tqdm(range(cropLength)):
        ret, frame = videoIn.read()
        if ret:
            out = calibrator.undistort_image(frame)
            videoOut.write(out)
        else:
            break
    videoIn.release()
    videoOut.release()
