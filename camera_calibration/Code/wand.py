import cv2
import numpy as np

from linalghelpers import lineEndPointsOnImage


# The pink aura
class ActiveBallMarker():
    def __init__(self, hue, hueRange, upperSaturation=0.5, mediumSaturation=0.1, mediumBrightness=0.94, lowerBrightness=0.8):
        self.lowerSaturation = 0
        self.upperBrightness = 1

        self.imageBlur = 3
        self.maxBallDiameter = 0.02 # Percentage of screen size (0.02 = 30/(sqrt(1280^2+720^2)))
        self.circularitySensitivity = 2  # 1-5ish range
        self.pupilBlur = 9 # must be odd

        self.hue = hue
        self.hueRange = hueRange
        self.upperSaturation = upperSaturation
        self.mediumSaturation = mediumSaturation
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
        mask = self.mask_color(hsvImg, 180, 180, self.mediumSaturation, self.lowerSaturation,
                               self.upperBrightness, self.mediumBrightness)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def get_corona_mask(self, hsvImg):
        h, w = hsvImg.shape[:2]
        pixelBallDiameter = int(self.maxBallDiameter*np.sqrt(w*w+h*h))
        # Make sure it is odd
        if pixelBallDiameter%2 == 0:
            pixelBallDiameter += 1

        # Setting the kernel to be the max size of the ball ensures 
        # that the inner saturated ball gets filled with mask when dilating
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixelBallDiameter, pixelBallDiameter))
        mask = self.mask_color(hsvImg, self.hue, self.hueRange, self.upperSaturation, self.mediumSaturation,
                               self.mediumBrightness, self.lowerBrightness)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def visualize_mask(self, img, outImg):
        hsvImg = self.prepare_image(img)
        pupilMask = self.get_pupil_mask(hsvImg)
        coronaMask = self.get_corona_mask(hsvImg)
        hasMask = np.logical_or(pupilMask > 0, coronaMask > 0)
        # Add mask layer to image
        #img[:] = np.dstack(( np.zeros(pupilMask.shape), np.zeros(pupilMask.shape), coronaMask))
        out = outImg.copy()
        out[hasMask] = np.dstack((coronaMask, np.zeros(pupilMask.shape), pupilMask))[hasMask]
        return out

    def detect(self, img):
        hsvImg = self.prepare_image(img)
        pupilMask = self.get_pupil_mask(hsvImg)
        coronaMask = self.get_corona_mask(hsvImg)
        markerMask = np.minimum(pupilMask, coronaMask)
        markerMask = cv2.GaussianBlur(markerMask, (self.pupilBlur, self.pupilBlur), 0)

        circles = cv2.HoughCircles(markerMask, cv2.HOUGH_GRADIENT, self.circularitySensitivity, 50,
                                   param1=100, param2=30, minRadius=0, maxRadius=int(self.maxBallDiameter/2))
        if circles is not None:
            if len(circles) == 1:
                circles = np.round(circles[0, :]).astype("int")
                return circles[0]
            # elif len(circles) > 1:
            #     print('hmmm')
        return None


class BallWand():
    def __init__(self):
        self.markerDistance = 0.4
        self.topMarker = ActiveBallMarker(hue=345, hueRange=40)
        self.bottomMarker = ActiveBallMarker(hue=120, hueRange=40)

    def detect(self, img):
        topKeypoints = self.topMarker.detect(img)
        bottomKeypoints = self.bottomMarker.detect(img)
        if topKeypoints is None or bottomKeypoints is None:
            return None

        keypoints = {'top': {'x': topKeypoints[0], 'y': topKeypoints[1], 'r': topKeypoints[2]},
                     'bottom': {'x': bottomKeypoints[0], 'y': bottomKeypoints[1], 'r': bottomKeypoints[2]}}
        return keypoints

    def draw(self, img, key):
        out = img.copy()
        cv2.circle(out, (key['top']['x'], key['top']['y']), key['top']['r'], (0, 0, 255), 5)
        cv2.circle(out, (key['bottom']['x'], key['bottom']['y']), key['bottom']['r'], (0, 255, 0), 5)
        cv2.line(out, (key['top']['x'], key['top']['y']), (key['bottom']['x'], key['bottom']['y']), (255, 0, 0), 5)
        return out

    def visualize_mask(self, img):
        raise RuntimeError("this method is broken, call on topMarker or bottomMarker instead")
        out = self.topMarker.visualize_mask(img,img)
        out = self.bottomMarker.visualize_mask(img,out)
        return out
