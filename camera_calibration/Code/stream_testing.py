import cv2
import numpy as np
from threading import Thread
from queue import Queue
import time

URL = 'rtsp://admin:123456@192.168.0.101:554/live/ch0'

def capture(stream_url,q):
	vcap = cv2.VideoCapture(stream_url)
	while(True):
		ret, frame = vcap.read()
		if ret:
			print(cv2.CAP_PROP_POS_MSEC)
			img_q.put(frame)


img_q = Queue()
t = Thread(target=capture, args=(URL,img_q,))
t.start()
while(True):
	frame = img_q.get()
	cv2.imshow('Reify-1', frame)
	cv2.waitKey(1)

