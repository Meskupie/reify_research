import cv2
import numpy as np
from threading import Thread
from queue import Queue
import time

from calibration import IntrinsicCalibrator


class Camera():
    def __init__(self, address):
        self.address = address

        self.calibrator = IntrinsicCalibrator()
        self.frame_queue = Queue()

    def capture_stream(self):
        cap = cv2.VideoCapture(self.address)
        while (True):
            ret, frame = cap.read()
            if ret:
                print(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.frame_queue.put(frame)

    def run_callback(self, callback):
        self.frame_queue = Queue()
        t = Thread(target=self.captureStream, args=())
        t.start()
        while (True):
            frame = self.frame_queue.get()
            callback(frame)
