import numpy as np
import cv2
from sklearn.cluster import DBSCAN

debug = 0

class CellDetector(object):

    def __init__(self):
        pass

    def Detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug = 0
        # debug = 1
        if (debug == 1):
            cv2.imshow('gray', gray)
            cv2.waitKey()
            
        ret, th4 = cv2.threshold(gray, 0x82, 255, cv2.THRESH_BINARY)

        if (debug == 1):
            cv2.imshow('th4', th4)
            cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ret, contours_image = cv2.threshold(gray, 0xff, 255, cv2.THRESH_BINARY)

        # cv2.imshow('contours_image2', contours_image)
        # cv2.waitKey()

        cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)

        # if (debug == 1):
        #     cv2.imshow('contours_image2', contours_image)
        #     cv2.waitKey()
 
        # if (debug == 1):
        #     cv2.imshow('thresh', thresh)
        #     self.out.write(thresh)

        centers = []  # vector of object centroids in a frame
        points = []
        clusters = []
        fake_cell = 0
        for cnt in contours:
            try:

                a, b, w, h = cv2.boundingRect(cnt)
                rect_cell = gray[b:b + h, a:a + w]
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(rect_cell)
                ratio = float(w)/float(h)
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))

                if (radius < 5 and radius > 0.5):
                    # cv2.circle(frame, centeroid, 5, (255, 255, 255), 1)
                    b = np.array([[x], [y]])
                    centers.append(b)
                else:
                    if debug == 1:
                        print('fake object number in the frame:')
                    fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass

        return centers
