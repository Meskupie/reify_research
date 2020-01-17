import cv2
import numpy as np

videoPath = '../sample_data/intrinsic_test.mp4'
displayResolution = (1280,720)
zoomResolution = (64,36)


clickFlag = False
clickLocation = (0,0)


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

def clickEvent(event, x, y, flags, param):
    global clickFlag, clickLocation
    if event == cv2.EVENT_LBUTTONUP:
        clickFlag = True
        clickLocation = (x,y)

if __name__ == "__main__":
    cv2.namedWindow("Labeler")
    cv2.setMouseCallback("Labeler", clickEvent)
    videoIn = cv2.VideoCapture(videoPath)
    frameHeight = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame = np.ones((frameHeight,frameWidth,3))
    prevFrame = np.zeros((1))

    labels = {}
    state = 'initializing'
    zoomX = 0
    zoomY = 0
    zoomW = 0
    zoomH = 0
    pointX = 0
    pointY = 0
    while(True):
        key = cv2.waitKey(1)
        if state == 'initializing':
            if clickFlag:
                clickFlag = False
                state = 'scrubbing'
                frameGrabbed, frame = videoIn.read()
                assert frameGrabbed, "Incorrect file name"
        elif state == 'scrubbing':
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
            elif key == -1:
                pass
            else:
                print(f'Getting key: {key}')

            if clickFlag:
                state = 'zoomed'
                clickFlag = False
                zoomX = int(clickLocation[0]-(zoomResolution[0]/2))
                zoomY = int(clickLocation[1]-(zoomResolution[1]/2))
                zoomW = int(zoomResolution[0])
                zoomH = int(zoomResolution[1])
                frame = frame[zoomY:zoomY+zoomH,zoomX:zoomX+zoomW]
                frame = cv2.resize(frame, zoomResolution, interpolation = cv2.INTER_AREA)

        elif state == 'zoomed':
            if clickFlag:
                state == 'classification'
                clickFlag = False
                pointX = zoomX + zoomResolution[0]*clickLocation[0]/displayResolution[0]
                pointY = zoomX + zoomResolution[1]*clickLocation[1]/displayResolution[1]
                frame = tryNewFrame(videoIn,frame)

            if key == 27:
                state == 'scrubbing'

        elif state == 'classification':
            if key == ord("t"): # top class
                pass

            if key == 27:
                state == 'scrubbing'




        # If frame has changed, display it
        if prevFrame.shape != frame.shape:
            frame = cv2.resize(frame,displayResolution,interpolation = cv2.INTER_AREA)
            prevFrame = frame.copy()
            cv2.imshow('Labeler',frame)
        elif not np.array_equal(prevFrame,frame):
            prevFrame = frame.copy()
            cv2.imshow('Labeler',frame)
