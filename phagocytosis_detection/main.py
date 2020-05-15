#!/usr/bin/env python3
import cv2
import copy
from cell_detect import CellDetector
from phagocytosis_detect import PhagocytosisDetector
import os

debug = 0
crop_width = 1328
crop_height = 1048
# crop_width = 512
# crop_height = 512

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
path = "./sample_0.mp4"

def main():

    cap = cv2.VideoCapture(path)
    out_path = './'

    c_detector = CellDetector()

    ph_detector = PhagocytosisDetector(10, 30, 5, 0)

    frame_count = 0

    while True:

        ret, whole_frame = cap.read()
        if ret != True:
            break

        print("frame:" + str(frame_count))

        start_vertical = 0
        start_horizontal = 0

        frame = whole_frame[start_vertical:start_vertical + crop_height, start_horizontal:start_horizontal + crop_width]

        orig_frame = copy.copy(frame)
        centers = c_detector.Detect(frame)

        if len(centers) > 0:

            ph_detector.match_track(centers, frame, frame_count)

            if debug == 1:
                print('dectections')
                for point in centers:
                    print(point[0], point[1])

            # out.write(frame)
            # cv2.imshow('Tracking1', frame)
            # cv2.waitKey()
        else:
            print("not detected")

        frame_count = frame_count + 1

    print("Done!")
    ph_detector.Save(out_path)
    cap.release()
    mark_cells(ph_detector)
    cv2.destroyAllWindows()

def mark_cells(ph_detector):
    cap2 = cv2.VideoCapture(path)

    out = cv2.VideoWriter('./phagocytosis.mp4',fourcc, 3.0, (crop_width, crop_height))

    frame_count = 0

    while True:

        ret, whole_frame = cap2.read()
        if ret != True:
            break

        frame = whole_frame[0:crop_height, 0:crop_width]
        orig_frame = copy.copy(frame)
        ph_detector.mark_cells(frame, frame_count)

        # cv2.imshow('Tracking2', frame)
        # cv2.waitKey()
        out.write(frame)
        frame_count = frame_count + 1

    print("Done!")
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
