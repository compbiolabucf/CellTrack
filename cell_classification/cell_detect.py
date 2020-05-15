import numpy as np
import cv2
import time

line_thick = 8
scale = 8

class CellDetector(object):

    def __init__(self):
        pass

    def detect(self, frame, frame_index):

        if(len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        debug = 0
        # debug = 1

        if (debug == 1):
            cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray', 900,900)
            cv2.imshow('gray', gray)
            # cv2.waitKey()

        if (debug == 2):
            np.save("gray", gray)

    
        ret, black = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        black_white = black

        if (debug == 1):
            cv2.namedWindow('black',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black', 900,900)
            cv2.imshow('black', black)
            pass

        if (debug == 2):
            np.save("black", black)


        t1 = time.time()
        contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        t2 = time.time()

        if (debug == 1):

            contours_image = np.zeros_like(black_white)

            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), line_thick)

            cv2.namedWindow('contours_image2',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('contours_image2', 900, 900)
            cv2.imshow('contours_image2', contours_image)


        centers = []  # vector of object centroids in a frame

        fake_cell = 0
        
        count = 0
        t3 = time.time()
        maximum = 0
        for i in range(len(contours)):
            try:
                if(hierarchy[0][i][3] > -1 and hierarchy[0][hierarchy[0][i][3]][3] < 0 and len(contours[i]) < 200): # and len(contours[i]) < (10 * scale) * (10 * scale)
                    (x, y), radius = cv2.minEnclosingCircle(contours[i])
                    centeroid = (int(x), int(y))
                    radius = int(radius)
                    if (radius < 90 and radius > 7 and gray[int(y)][int(x)] > 100):
    
                        retval = cv2.minAreaRect(contours[i])

                        ratio = min(retval[1][0], retval[1][1])/max(retval[1][0], retval[1][1])

                        box = cv2.boxPoints(retval)
                        box = np.int0(box)
                        cv2.drawContours(black_white,[box],0,0,1)
                        cv2.putText(black_white, str(float("{0:.2f}".format(ratio))), (int(retval[0][0] - 20), int(retval[0][1] + 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                        cell_radius = 80

                        b = np.array([[x/scale], [y/scale], [ratio]])
                        centers.append(b)
                        
                        cv2.circle(frame, centeroid, 10*scale, (0, 255, 255), 2)

                    else:
                        fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 900,900)
            cv2.imshow('frame', frame)
            # cv2.waitKey()


            cv2.namedWindow('black_white',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black_white', 900,900)
            cv2.imshow('black_white', black_white)
            cv2.waitKey()
            pass

        if (debug == 2):
            np.save("black_white", black_white)
        
        # input()

        debug = 0

        return centers
