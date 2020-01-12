import cv2
import numpy as np

def image_callback(frame):
	return frame

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == False:
    	break

    output = image_callback(frame)

    cv2.imshow('frame',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()