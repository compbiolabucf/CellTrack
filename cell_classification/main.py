#!/usr/bin/env python3
import cv2
import copy
from cell_detect import CellDetector
from cell_classify import CellClassifier
import os
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing 

import time

debug = 0
# crop_width = 512
# crop_height = 512
crop_width = 1328
crop_height = 1048
# scale = 1
scale = 8
line_thick = 1

debug = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def main():
    path = "./"
    process_one_video_main(path)

def process_one_video_main(path):

    detector = CellDetector()
    classifier = CellClassifier(30, 30, 5, 0)

    frame_count = 0
    image_dir = path + "input_images/"

    while True:

        print("frame_count: ", frame_count)
        
        #read from raw data
        image_path = image_dir + str(frame_count) + ".npy"
        ret = os.path.exists(image_path)
        if ret != True:
            break

        frame = np.load(image_path)
        centers = detector.detect(frame, frame_count)

        if len(centers) > 0:

            classifier.match_track(centers, frame, frame_count)

            # for point in centers:
            #     cv2.circle(frame, (point[0] * scale, point[1] * scale), 10 * scale, (255, 255, 0), line_thick)

            # out.write(frame)
            # 
            # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Tracking', 900,900)
            # cv2.imshow('Tracking', frame)
            # cv2.waitKey()
        else:
            print("not detected")

        frame_count = frame_count + 1

    print("Done!")
    # cap.release()
    # out.release()
    cv2.destroyAllWindows()

    classifier.Classify(path)
    mark_cells(classifier, path)


def mark_cells(classifier, path):

    out2 = None
    frame_count = 0

    image_dir = path + "input_images/"

    print("make cells, ", image_dir)

    while True:

        #read from raw data
        image_path = image_dir + str(frame_count) + ".npy"
        ret = os.path.exists(image_path)
        if ret != True:
            break

        frame = np.load(image_path)

        orig_frame = frame.copy()
        print("make cells frame_count:" + str(frame_count))

        cv2.putText(frame, str(frame_count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))

        if(frame.shape[1] != crop_width):
            frame = cv2.resize(frame, (crop_width, crop_height), interpolation = cv2.INTER_CUBIC)

        if(len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame = classifier.mark_cells(frame, frame_count, None, None, image_dir)
        
        if(not os.path.exists(path)):
            os.makedirs(path)

        if(out2 is None):
            out2 = cv2.VideoWriter(path + "cell_detected.mp4",fourcc, 5.0, (frame.shape[1], frame.shape[0]), isColor=True)
            # print("VideoWriter", frame.shape[1], frame.shape[0])

        # print(frame.shape[1], frame.shape[0])
        out2.write(frame)

        # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Tracking', 900,900)
        # cv2.imshow('Tracking', frame)
        # cv2.waitKey()

        frame_count = frame_count + 1

    print("Done!")
    # cap2.release()
    if(out2):
        out2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # execute main
    main()
