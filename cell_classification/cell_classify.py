import cv2
import numpy as np
import pandas as pd
import time
import hungarian
import string
import os

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from inspect import currentframe, getframeinfo

debug = 0
array_size = 300
image_count = 193
scale = 8

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

class CellTrack(object):

    def __init__(self, prediction, trackIdCount):
        self.track_id = trackIdCount  # identification of each track object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path
        self.live_score = 0.0
        self.death_score = 0.0

class CellClassifier(object):

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):

        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.del_tracks = []
        self.trackIdCount = trackIdCount
        self.alive_mat = []
        self.coordinate_matrix = []


    def match_track(self, detections, frame, frame_index):

        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = CellTrack(detections[i], self.trackIdCount)
                self.tracks.append(track)

                one_row = np.zeros(array_size, dtype = float)
                one_row[:] = np.nan
                self.alive_mat.append(one_row)

                one_row = np.zeros(array_size*2, dtype = int)
                self.coordinate_matrix.append(one_row)

                self.trackIdCount += 1


        N = len(self.tracks)
        M = len(detections)
        # print('tracks, dections:', N, M)
        D = max(N, M)
        cost = np.full([D, D], 2000)  # Cost matrix

        predit = np.zeros([N, 2])
        for i in range(N):
            predit[i][0] = self.tracks[i].prediction[0][0]
            predit[i][1] = self.tracks[i].prediction[1][0]

        det = np.zeros([M, 2])
        for i in range(M):
            det[i][0] = detections[i][0][0]
            det[i][1] = detections[i][1][0]

        cost_new = distance_matrix(predit, det)
        cost_new = np.where(cost_new < 20, cost_new, 2000)

        cost[0:N, 0:M] = cost_new
        assignment = []
        for _ in range(N):
            assignment.append(-1)

        t3 = time.time()
        answers = hungarian.lap(cost)
        t4 = time.time()
        if(N > M):
            for i in range(M):
                assignment[answers[1][i]] = i
            sum0 = sum(cost[answers[1], range(len(answers[1]))])
            # print("sum0: ", sum0)
        else:
            for i in range(N):
                assignment[i] = answers[0][i]
            sum1 = sum(cost[range(len(answers[0])), answers[0]])
            # print("sum1: ", sum1)

        for i in range(len(assignment)):
            if (assignment[i] == -1 or cost[i][assignment[i]] > self.dist_thresh):
                assignment[i] = -1
                # un_assigned_tracks.append(i)
                self.tracks[i].skipped_frames += 1
            else:
                self.tracks[i].skipped_frames = 0

        i = 0
        while(i < len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                # print("track " + str(self.tracks[i].track_id) + " skipped " + str(self.tracks[i].skipped_frames) + "frames")
                self.del_tracks.append(self.tracks[i])
                del self.tracks[i]
                del assignment[i]
            else:
                i = i + 1
                
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)
                assignment.append(i)

        if (len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = CellTrack(detections[un_assigned_detects[i]], self.trackIdCount)
                self.tracks.append(track)

                one_row = np.zeros(array_size, dtype = float)
                one_row[:] = np.nan
                self.alive_mat.append(one_row)

                one_row = np.zeros(array_size*2, dtype = int)
                self.coordinate_matrix.append(one_row)

                self.trackIdCount += 1

        live_count = 0
        dead_count = 0

        for i in range(len(assignment)):

            if (assignment[i] != -1):
                if debug == 1:
                    print(assignment[i])
                self.tracks[i].trace.append(detections[assignment[i]])
                self.tracks[i].prediction = detections[assignment[i]]

                if (len(self.tracks[i].trace) > 0):

                    x3 = self.tracks[i].trace[len(self.tracks[i].trace) - 1][0][0]
                    y3 = self.tracks[i].trace[len(self.tracks[i].trace) - 1][1][0]
                    ratio = self.tracks[i].trace[len(self.tracks[i].trace) - 1][2][0]

                    self.coordinate_matrix[self.tracks[i].track_id][frame_index*2] = x3
                    self.coordinate_matrix[self.tracks[i].track_id][frame_index*2 + 1] = y3
                    self.alive_mat[self.tracks[i].track_id][frame_index] = ratio

                    cv2.putText(frame, str(self.tracks[i].track_id), (int(x3 + 9) * scale, int(y3 + 4) * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 127, 255), 2)

            if (len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]


    def Classify(self, outpath):
        print("tracker save.")
        window_radius = 4

        i = 0
        while(i < len(self.coordinate_matrix)):
            if(np.count_nonzero(self.coordinate_matrix[i]) < 60):
                del self.coordinate_matrix[i]
                del self.alive_mat[i]
                # input()
            else:
                i += 1

        file = open(outpath + "tracks_ratio before trace back.txt", "w")
        for i in range(len(self.alive_mat)):
            for j in range(array_size):
                file.write("%s, " % self.alive_mat[i][j])
            file.write("\n")        
        file.close()

        file = open(outpath + "file0.txt", "w")
        for i in range(0, len(self.alive_mat), 1):
            file.write("track: %d;" % i)
            one_track = np.zeros_like(self.alive_mat[i])
            one_track[:] = np.nan
            
            for j in range(window_radius, image_count - window_radius):
                temp_array = self.alive_mat[i][j - window_radius : j + window_radius + 1]
                temp_array = temp_array[~np.isnan(temp_array)]

                if(len(temp_array) > 1):
                    local_max = np.max(temp_array)
                    local_min = np.min(temp_array)
                    one_track[j] = local_max - local_min
                else:
                    pass

            for s in range(array_size):
                file.write("%s, " % one_track[s])
            file.write(";")

            for k in range(len(one_track)):
                if(one_track[k] > 0.3):
                    one_track[k] = 1
                else:
                    one_track[k] = 0

            for s in range(array_size):
                file.write("%s, " % one_track[s])
            file.write(";")

            one_track_str = str(one_track)
            one_track_str = one_track_str.replace("\n", "")
            one_track_str = one_track_str.replace("[", "")
            one_track_str = one_track_str.replace("]", "")


            sub_arrays = one_track_str.split('1.')
            max_dead_count = 0

            for m in range(1, len(sub_arrays) - 1, 1):
                dead_count = sub_arrays[m].count('0.')
                if(max_dead_count < dead_count):
                    max_dead_count = dead_count

            if(max_dead_count == 0):
                one_track[:] = -1
                pass
            else:# pad start
                n = array_size - 1
                while(not (one_track[n] == 1) and n > -1):
                    one_track[n] = -1
                    n = n - 1

                while(n > -1):
                    one_track[n] = 1
                    n = n - 1
                #pad end
                last_dead_status = np.where(one_track == -1)[0][0]
                # print("last_dead_status: ", last_dead_status)
                if(image_count - last_dead_status < max_dead_count):
                    file.write("%s, " % "image_count: " + str(image_count) + ", last_dead_status: " + str(last_dead_status) + ", max_dead_count: " + str(max_dead_count) + ";")

                    for p in range(last_dead_status, len(one_track), 1):
                        one_track[p] = 1
                else:
                    pass
                # print(np.where(one_track == -1))

            for s in range(array_size):
                file.write("%s, " % one_track[s])
            file.write(";")

            self.alive_mat[i] = one_track

            file.write("\n")
        file.close()

        cell_info = np.load("./0/cells_info.npy")
        live_dead_table = np.zeros((image_count, 4), dtype=int)


        for i in range(0, len(self.alive_mat), 1):
            for j in range(0, image_count):
                cell_x = self.coordinate_matrix[i][2 * j + 0]
                cell_y = self.coordinate_matrix[i][2 * j + 1]


                if(cell_x > 0 and cell_y > 0):

                    if(self.alive_mat[i][j] == 1):
                        live_dead_table[j][0] = live_dead_table[j][0] + 1
                    elif(self.alive_mat[i][j] == -1):
                        live_dead_table[j][1] = live_dead_table[j][1] + 1
                    else:
                        pass

                    k = 0
                    for k in range(len(cell_info)):
                        if(np.abs(cell_x - cell_info[k][j * 3 + 0]/8) < 10 and np.abs(cell_y - cell_info[k][j * 3 + 1]/8) < 10):
                            # print("true_0")
                            if(self.alive_mat[i][j] == cell_info[k][j * 3 + 2]):

                                if(self.alive_mat[i][j] == 1):
                                    live_dead_table[j][2] = live_dead_table[j][2] + 1
                                elif(self.alive_mat[i][j] == -1):
                                    live_dead_table[j][3] = live_dead_table[j][3] + 1
                                else:
                                    pass
                                break

        if(not os.path.exists(outpath)):
            os.makedirs(outpath)

        np.savetxt(outpath + 'live_dead_table.txt', live_dead_table, fmt='%d')

    def mark_cells(self, frame, frame_index, sheet, beacon_count, image_dir):

        live_count = 0
        dead_count = 0
        zero_cell = 0

        for i in range(len(self.alive_mat)):
            x3 = self.coordinate_matrix[i][frame_index*2]
            y3 = self.coordinate_matrix[i][frame_index*2 + 1]
            live_status = self.alive_mat[i][frame_index]

            if(x3 > 0 or y3 > 0):
                if(live_status > 0):
                    cv2.circle(frame, (int(x3), int(y3)), 10, (255, 255, 0), 1)
                    # cv2.putText(frame, str(i), (int(x3) + 3, int(y3) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                    live_count += 1
                elif(live_status < 0):
                    cv2.circle(frame, (int(x3), int(y3)), 10, (0, 255, 255), 1)
                    # cv2.putText(frame, str(i), (int(x3) + 3, int(y3) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255))
                    dead_count += 1
                else:
                    pass
        
        if(sheet):
            sheet.cell(row=frame_index + 2, column = beacon_count * 3 + 0 + 1).value = live_count
            sheet.cell(row=frame_index + 2, column = beacon_count * 3 + 1 + 1).value = dead_count
            sheet.cell(row=frame_index + 2, column = beacon_count * 3 + 2 + 1).value = live_count + dead_count

        cv2.putText(frame, str(frame_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 221, 48), 1)
        cv2.putText(frame, "alive: " + str(live_count) + ", ", (60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))                
        cv2.putText(frame, "dead: " + str(dead_count), (175, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
        
        return frame
