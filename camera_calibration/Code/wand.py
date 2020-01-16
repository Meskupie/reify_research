import cv2
import numpy as np

from linalghelpers import lineEndPointsOnImage


# The pink aura
class ActiveBallMarker():
    def __init__(self, hue, hueRange, upperSaturation=0.2, mediumSaturation=0.04, lowerSaturation=0, upperBrightness=1,
                 mediumBrightness=0.96, lowerBrightness=0.85):
        self.imageBlur = 3
        self.maxBallRad = 20
        self.circularitySensitivity = 5  # 1-5ish range
        self.pupilBlur = 5  # must be odd

        self.hue = hue
        self.hueRange = hueRange
        self.upperSaturation = upperSaturation
        self.mediumSaturation = mediumSaturation
        self.lowerSaturation = lowerSaturation
        self.upperBright = upperBrightness
        self.mediumBrightness = mediumBrightness
        self.lowerBrightness = lowerBrightness

    def mask_color(self, imgHsv, hueCenter, hueRadius,
                   upperSaturation=1.0, lowerSaturation=0.3, upperValue=1.0, lowerValue=0.3):
        # TODO: hue_rad > 360 = bad, darkness bounds, whiteness bounds
        assert hueRadius >=0 and hueRadius <= 360, "hueRadius must be between 0 and 360"

        if hueCenter + hueRadius > 360:
            hueCenter = hueCenter - 360
        if hueCenter - hueRadius < 0:
            # above zero
            lower = (0, 255 * lowerSaturation, 255 * lowerValue)
            upper = (int((hueCenter + hueRadius) / 2), 255 * upperSaturation, 255 * upperValue)
            maskPart1 = cv2.inRange(imgHsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
            # below zero
            lower = (int((360 + hueCenter - hueRadius) / 2), 255 * lowerSaturation, 255 * lowerValue)
            upper = (179, 255 * upperSaturation, 255 * upperValue)
            maskPart2 = cv2.inRange(imgHsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
            mask = maskPart1 | maskPart2
        else:
            lower = (int((hueCenter - hueRadius) / 2), 255 * lowerSaturation, 255 * lowerValue)
            upper = (int((hueCenter + hueRadius) / 2), 255 * upperSaturation, 255 * upperValue)
            mask = cv2.inRange(imgHsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
        return mask

    def prepare_image(self, img):
        blurredImg = cv2.GaussianBlur(img, (self.imageBlur, self.imageBlur), 0)
        hsvImg = cv2.cvtColor(blurredImg, cv2.COLOR_BGR2HSV)
        return hsvImg

    def get_pupil_mask(self, hsvImg):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.pupilBlur, self.pupilBlur))
        mask = self.mask_color(hsvImg, self.hue, self.hueRange, self.mediumSaturation, self.lowerSaturation,
                               self.upperBright, self.mediumBrightness)
        mask = cv2.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        return mask

    def get_corona_mask(self, hsv_img):
        # Setting the kernel to be the max size of the ball ensures 
        # that the inner saturated ball gets filled with mask when dilating
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.maxBallRad * 2 + 1, self.maxBallRad * 2 + 1))
        mask = self.mask_color(hsv_img, self.hue, self.hueRange, self.upperSaturation, self.mediumSaturation,
                               self.mediumBrightness, self.lowerBrightness)
        mask = cv2.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        return mask

    def visualize_mask(self, img):
        hsvImg = self.prepare_image(img)
        pupilMask = self.get_pupil_mask(hsvImg)
        coronaMask = self.get_corona_mask(hsvImg)
        hasMask = np.logical_or(pupilMask > 0, coronaMask > 0)
        # Add mask layer to image
        img[hasMask] = np.dstack((coronaMask, np.zeros(pupilMask.shape), pupilMask))[hasMask]

    def detect(self, img):
        hsvImg = self.prepare_image(img)
        pupilMask = self.get_pupil_mask(hsvImg)
        coronaMask = self.get_corona_mask(hsvImg)
        markerMask = np.minimum(pupilMask, coronaMask)
        markerMask = cv2.GaussianBlur(markerMask, (self.pupilBlur, self.pupilBlur), 0)

        circles = cv2.HoughCircles(markerMask, cv2.HOUGH_GRADIENT, self.circularitySensitivity, 50,
                                   param1=100, param2=30, minRadius=0, maxRadius=self.maxBallRad)
        if circles is not None:
            if len(circles) == 1:
                circles = np.round(circles[0, :]).astype("int")
                return circles[0]
        return None


class BallWand():
    def __init__(self):
        # NOTE: where should these "hard codes" be located and how should they be referenced?
        self.markerDistance = 0.4
        self.topMarker = ActiveBallMarker(hue=345, hueRange=35)
        self.bottomMarker = ActiveBallMarker(hue=120, hueRange=40)

    def detect(self, img):
        top_keypoints = self.topMarker.detect(img)
        bottom_keypoints = self.bottomMarker.detect(img)
        if top_keypoints is None or bottom_keypoints is None:
            return None

        keypoints = {'top': {'x': top_keypoints[0], 'y': top_keypoints[1], 'r': top_keypoints[2]},
                     'bottom': {'x': bottom_keypoints[0], 'y': bottom_keypoints[1], 'r': bottom_keypoints[2]}}
        return keypoints

    def draw_results(self, img, key):
        cv2.circle(img, (key['top']['x'], key['top']['y']), key['top']['r'], (0, 0, 255), 5)
        cv2.circle(img, (key['bottom']['x'], key['bottom']['y']), key['bottom']['r'], (0, 255, 0), 5)
        cv2.line(img, (key['top']['x'], key['top']['y']), (key['bottom']['x'], key['bottom']['y']), (255, 0, 0), 5)

    def visualize_mask(self, img):
        self.topMarker.visualize_mask(img)
        self.bottomMarker.visualize_mask(img)
