import os
import sys
import glob
import yaml
import argparse
import cv2
import numpy as np

sys.path.insert(1, '../code')
from wand import ActiveBallMarker

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', required=True, help='Path to the processed videos folder')
ap.add_argument('-l', '--labels', required=True, help='Path to the labels dataset file (or desired location if none)')
ap.add_argument('-e', '--extension', help='Video extension', default='.avi')
ap.add_argument('--height', help='Image height', default=720)
ap.add_argument('--width', help='Image width', default=1280)
args = vars(ap.parse_args())
videosPath = args['dir']
labelsPath = args['labels']
videoExtension = args['extension']
videoResolution = np.array((args['width'], args['height']))

print('\n\n\nHey there, need instructions?\n\n' \
      'Press [a] and [d] to move backwards and forwards one frame\n' \
      'Press [q] and [e] to move backwards and forwards ten frames\n' \
      'Press [t] and [g] to switch between existing labels\n' \
      'Press [z] and [c] to switch between existing views\n' \
      'Press [esc] to back out and [k] to exit\n\n' \
      'Click on markers to zoom in, click again to mark a location (or just use auto)\n' \
      'Once marked, press "r" or "f" to label the mark as a "top" or "bottom" class\n' \
      'Click on an existing marker to remove it\n')


class Labeler():
    def __init__(self, labelsPath):
        self.labelDict = {}
        self.labelsPath = labelsPath
        if os.path.isfile(self.labelsPath):
            self.load_labels()

    def save_labels(self):
        # if os.path.isfile(self.labelsPath):
        #     os.remove(self.labelsPath)
        with open(self.labelsPath, 'w') as file:
            yaml.dump(self.labelDict, file)

    def load_labels(self):
        with open(self.labelsPath) as file:
            self.labelDict = yaml.full_load(file)

    def add_label(self, labelLocation, labelClass):
        global frameIndex, videoIndex
        if str(frameIndex) in self.labelDict:
            labels = self.labelDict[str(frameIndex)]
            labels.append({'location': np.array(labelLocation).tolist(), 'class': labelClass, 'view': videoIndex})
        else:
            self.labelDict[str(frameIndex)] = [
                {'location': np.array(labelLocation).tolist(), 'class': labelClass, 'view': videoIndex}]
        self.save_labels()

    def remove_label(self, label):
        global frameIndex, videoIndex
        if str(frameIndex) in self.labelDict:
            labels = self.labelDict[str(frameIndex)]
            labels.remove(label)
            if len(labels) != 0:
                self.labelDict[str(frameIndex)] = labels
            else:
                self.labelDict.pop(str(frameIndex), None)
        self.save_labels()

    def get_labels(self, lockFrame=True, lockView=True):
        global frameIndex, videoIndex
        if lockFrame:
            if str(frameIndex) in self.labelDict:
                labels = self.labelDict[str(frameIndex)]
                if lockView:
                    currentViewLabels = []
                    for label in labels:
                        if label['view'] == videoIndex:
                            currentViewLabels.append(label)
                    return currentViewLabels
                return labels
            return []
        else:
            raise RuntimeError()  # all labels

    def get_label_distance(self, direction=0):
        global frameIndex
        assert direction != 0, "Use direction 1 or -1"
        frameList = np.array([int(key) for key in list(self.labelDict.keys())])
        frameList.sort()
        distance = 0
        if len(frameList) != 0:
            if direction > 0:
                distance = frameList[np.argmax((frameList - frameIndex) > 0)] - frameIndex
            else:
                distance = frameList[::-1][np.argmax((frameIndex - frameList[::-1]) > 0)] - frameIndex
        return distance

    def get_previous_label_distance(self):
        global frameIndex
        frameList = np.array([int(key) for key in list(self.labelDict.keys())])
        frameList.sort()
        return frameList[::-1][np.argmax((frameIndex - frameList[::-1]) > 0)]

    def get_image_with_header(self, image):
        global frameIndex, videoIndex
        fontColor = (153, 51, 0)
        out = image.copy()
        out = cv2.putText(out, f'View {videoIndex + 1} of {len(videoPaths)} at frame {frameIndex}',
                          (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, fontColor, 2, cv2.LINE_AA, False)
        labels = self.get_labels(lockView=False)
        if len(labels) > 0:
            views = [label["view"] for label in labels]
            views = list(set(views))
            views.sort()
            out = cv2.putText(out, f'Labeled by view{"s" if len(views) > 0 else ""} ' + ', '.join(
                [str(view + 1) for view in views]),
                              (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, fontColor, 2, cv2.LINE_AA, False)
        return out

    def get_image_with_labels(self, image):
        global markerLocation, videoIndex
        labels = self.get_labels()
        out = image.copy()
        for label in labels:
            markLocation = np.array(label['location']).astype(int)
            if label['class'] == 'top':
                markColor = (178, 145, 242)
            elif label['class'] == 'bottom':
                markColor = (145, 242, 145)
            else:
                raise RuntimeError('wrong class name')
            cv2.circle(out, tuple(markLocation), 16, markColor, 3)
        return out


def capture_relative_frame(count):
    global videoIn, frameIndex, frame, frameHasChanged
    frameIndex = frameIndex + count
    frameIndex = max(1, min(int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT)), frameIndex))
    videoIn.set(cv2.CAP_PROP_POS_FRAMES, frameIndex - 1)
    frameGrabbed, newFrame = videoIn.read()
    if frameGrabbed:
        frame = newFrame
        frameHasChanged = True


def get_new_video():
    global prevVideoIndex, videoIndex, videoPaths, frameIndex, videoIn
    videoIndex = videoIndex % len(videoPaths)
    prevVideoIndex = videoIndex
    videoIn = cv2.VideoCapture(videoPaths[videoIndex])
    videoIn.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    capture_relative_frame(0)


def process_key(key, validKeys):
    global prevVideoIndex, videoIndex, frame, userInterfaceState, labeler
    if ord('d') in validKeys and key == ord('d'):  # Forward one frame
        capture_relative_frame(1)
    elif ord('a') in validKeys and key == ord('a'):  # Backward one frame
        capture_relative_frame(-1)
    elif ord('e') in validKeys and key == ord('e'):  # Forward 5 frames
        capture_relative_frame(5)
    elif ord('q') in validKeys and key == ord('q'):  # Backward 5 frames
        capture_relative_frame(-5)
    elif ord('t') in validKeys and key == ord('t'):  # Go to next label
        capture_relative_frame(labeler.get_label_distance(direction=1))
    elif ord('g') in validKeys and key == ord('g'):  # Go to previous label
        capture_relative_frame(labeler.get_label_distance(direction=-1))
    elif ord('c') in validKeys and key == ord('c'):  # Up one video stream
        videoIndex += 1
        get_new_video()
    elif ord('z') in validKeys and key == ord('z'):  # down one video stream
        videoIndex -= 1
        get_new_video()
    elif ord('r') in validKeys and key == ord('r'):  # Add label with 'top' class
        labeler.add_label(markerLocation, 'top')
        userInterfaceState = 'scrubbing'
    elif ord('f') in validKeys and key == ord('f'):  # Add label with 'green' class
        labeler.add_label(markerLocation, 'bottom')
        userInterfaceState = 'scrubbing'
    elif ord('m') in validKeys and key == ord('m'):  # Coverage map visualization
        pass
    elif key == 27:
        userInterfaceState = 'scrubbing'
    elif key == ord('k'):
        exit()


def click_event(event, x, y, flags, param):
    global clickFlag, clickLocation, initialized
    if event == cv2.EVENT_LBUTTONUP:
        if initialized == True:
            clickFlag = True
            clickLocation = np.array([x, y])
        else:
            initialized = True


zoomResolution = np.array([64, 36])
displayResolution = np.array([1280, 720])
ballCrop = 36

initialized = False
clickFlag = False
clickLocation = np.array([0, 0])
videoIndex = 0
frameIndex = 1
frame = np.zeros(0)
assert zoomResolution[0] / videoResolution[0] == zoomResolution[1] / videoResolution[1]
zoomRatio = zoomResolution[0] / videoResolution[0]
videosName = videosPath.split('/')[-1] if videosPath.split('/')[-1] is not '' else videosPath.split('/')[-2]
videoPaths = glob.glob(os.path.join(videosPath, '*' + videoExtension))
assert len(videoPaths) > 0
videoPaths.sort()

cv2.namedWindow("Labeler")
cv2.setMouseCallback("Labeler", click_event)
marker = ActiveBallMarker(180, 180, mediumBrightness=0.85, mediumSaturation=0.12)
labeler = Labeler(labelsPath)

labels = {}
userInterfaceState = 'scrubbing'
prevVideoIndex = -1
zoomedFrame = np.zeros(1)

get_new_video()
capture_relative_frame(labeler.get_label_distance(direction=1))

while (True):
    videoIndex = videoIndex % len(videoPaths)
    if videoIndex != prevVideoIndex:
        get_new_video()

    key = cv2.waitKey(1)

    if userInterfaceState == 'scrubbing':
        process_key(key, [ord('d'), ord('a'), ord('e'), ord('q'), ord('t'), ord('g'), ord('z'), ord('c')])
        capture_relative_frame(0)
        frame = labeler.get_image_with_labels(frame)

        if clickFlag:
            userInterfaceState = 'zoomed'
            clickFlag = False

            # check if deleting a label
            labels = labeler.get_labels()
            distances = np.array([np.sqrt(
                np.dot((np.array(label['location']) - clickLocation), (np.array(label['location']) - clickLocation)))
                for label in labels])
            if np.sum(distances < 16) == 1:
                index = np.argmax(distances < 16)
                labeler.remove_label(labels[index])
                userInterfaceState = 'scrubbing'
                capture_relative_frame(0)
                frame = labeler.get_image_with_header(frame)
                frame = labeler.get_image_with_labels(frame)
            else:
                # proceed with zoom
                focusLocation = (clickLocation - (zoomResolution / 2)).astype(int)
                kernelLocation = (clickLocation - (ballCrop / 2)).astype(int)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ballCrop, ballCrop))
                mask = np.zeros(frame.shape)
                mask[kernelLocation[1]:kernelLocation[1] + ballCrop,
                kernelLocation[0]:kernelLocation[0] + ballCrop] = np.dstack((kernel, kernel, kernel))
                noMask = mask == 0
                maskedImage = frame.copy()
                maskedImage[noMask] = mask[noMask]

                capture_relative_frame(0)
                frame = frame[focusLocation[1]:focusLocation[1] + int(zoomResolution[1]),
                        focusLocation[0]:focusLocation[0] + int(zoomResolution[0])]
                frame = cv2.resize(frame, tuple(videoResolution), interpolation=cv2.INTER_AREA)
                zoomedFrame = frame.copy()

                mark = marker.detect_pupil(maskedImage)
                if mark is not None:
                    markerLocation = np.array([mark['x'], mark['y']])
                    userInterfaceState = 'marked'
                    newMarkerLocation = (((markerLocation - focusLocation) / zoomResolution) * videoResolution).astype(
                        int)
                    # cv2.circle(frame, tuple(newMarkerLocation), int(mark['r'] / zoomRatio), (20, 20, 20), 3)
                    cv2.circle(frame, tuple(newMarkerLocation), 4, (250, 100, 0), -1)
            frameHasChanged = True

    elif userInterfaceState == 'zoomed':
        process_key(key, [27])
        if clickFlag:
            userInterfaceState = 'marked'
            clickFlag = False
            markerLocation = (clickLocation * zoomResolution / videoResolution + focusLocation)

            frame = zoomedFrame.copy()
            cv2.circle(frame, tuple(clickLocation), 4, (250, 100, 0), -1)
            frameHasChanged = True

    elif userInterfaceState == 'marked':
        process_key(key, [ord('r'), ord('f'), 27])
        if clickFlag:
            userInterfaceState = 'marked'
            clickFlag = False
            markerLocation = (clickLocation * zoomResolution / videoResolution + focusLocation)

            frame = zoomedFrame.copy()
            cv2.circle(frame, tuple(clickLocation), 4, (250, 100, 0), -1)
            frameHasChanged = True

    # If frame has changed, display it
    if frameHasChanged:
        frameHasChanged = False
        frame = labeler.get_image_with_header(frame)
        frame = cv2.resize(frame, tuple(displayResolution), interpolation=cv2.INTER_AREA)
        cv2.imshow('Labeler', frame)
