import cv2
import numpy as np
import statistics
import hungarian
import time

#from scipy.optimize import linear_sum_assignment
from inspect import currentframe, getframeinfo

from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix

debug = 0

frame_number = 300

class CellTrack(object):

    def __init__(self, prediction, trackIdCount):
        self.track_id = trackIdCount  # identification of each track object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path
        self.live_score = 0.0
        self.death_score = 0.0
        self.cluster_index = np.nan

class PhagocytosisDetector(object):

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.del_tracks = []
        self.trackIdCount = trackIdCount
        self.cluster_img_mat = []
        self.cell_num_in_cluster = []
        self.cell_num_in_cluster_x = []
        self.clusterContour_img_mat = []
        self.clusters = []
        self.clusters_eating_or_not = []

    def match_track(self, detections, frame, frame_index):

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = CellTrack(detections[i], self.trackIdCount)
                self.tracks.append(track)
                self.trackIdCount += 1

        N = len(self.tracks)
        M = len(detections)

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
        else:
            for i in range(N):
                assignment[i] = answers[0][i]
            sum1 = sum(cost[range(len(answers[0])), answers[0]])

        for i in range(len(assignment)):
            if (assignment[i] == -1 or cost[i][assignment[i]] > self.dist_thresh):
                assignment[i] = -1
                self.tracks[i].skipped_frames += 1
            else:
                self.tracks[i].skipped_frames = 0
        
        i = 0
        while(i < len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                self.del_tracks.append(self.tracks[i])
                del self.tracks[i]
                del assignment[i]
            else:
                i = i + 1
                
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)

        points = []
        cluster_points = []

        if(frame_index > 0):
            nCluster = len(self.clusters)
            for j in range(nCluster):
                cluster_points.append([])

        if (len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = CellTrack(detections[un_assigned_detects[i]], self.trackIdCount)
                self.tracks.append(track)

                frame1 = np.zeros([1328, 1048])
                x = int(detections[un_assigned_detects[i]][0][0])
                y = int(detections[un_assigned_detects[i]][1][0])

                cv2.circle(frame1, (x, y), 10, (255, 255, 255), 1)
                
                for i in range(len(self.clusters)):
                    frame2 = np.zeros([1328, 1048])
                    if(len(self.clusterContour_img_mat[i][frame_index - 1]) > 0):
                        cv2.fillPoly(frame2, pts=self.clusterContour_img_mat[i][frame_index - 1][0], color=(255, 255, 255))
                        frame3 = np.logical_and(frame1, frame2)
                        if True in frame3:
                            # if(i == 31):
                            #     print(i, x, y)
                            four_corner = 3
                            cluster_points[i].append([x - four_corner, y - four_corner])
                            cluster_points[i].append([x - four_corner, y + four_corner])
                            cluster_points[i].append([x + four_corner, y - four_corner])
                            cluster_points[i].append([x + four_corner, y + four_corner])
                            track.cluster_index = i
                            self.clusters[i].append(track.track_id)

                            break
                    else:
                        pass

                self.trackIdCount += 1


        for i in range(len(assignment)):

            if debug == 1:
                print('assign')
            if (assignment[i] != -1):
                if debug == 1:
                    print(assignment[i])
                self.tracks[i].trace.append(detections[assignment[i]])
                self.tracks[i].prediction = detections[assignment[i]]

                if (len(self.tracks[i].trace) > 0):
                    x3 = self.tracks[i].trace[len(self.tracks[i].trace) - 1][0][0]
                    y3 = self.tracks[i].trace[len(self.tracks[i].trace) - 1][1][0]

                    if(frame_index == 0):
                        points.append([x3, y3])
                    else:
                        if(self.tracks[i].cluster_index > -1):
                            four_corner = 3
                            cluster_points[self.tracks[i].cluster_index].append([x3 - four_corner, y3 - four_corner])
                            cluster_points[self.tracks[i].cluster_index].append([x3 + four_corner, y3 - four_corner])
                            cluster_points[self.tracks[i].cluster_index].append([x3 - four_corner, y3 + four_corner])
                            cluster_points[self.tracks[i].cluster_index].append([x3 + four_corner, y3 + four_corner])


            if (len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

        if(frame_index == 0):
            clustering = DBSCAN(eps=15, min_samples=4).fit(points)
            nCluster = max(clustering.labels_) + 1
            
            for i in range(nCluster):
                cluster_points.append([])
                self.clusters.append([])

                one_row = np.zeros(frame_number * 3, dtype = float)
                one_row[:] = np.nan
                self.cluster_img_mat.append(one_row)

                one_row = np.zeros(frame_number, dtype = float)
                one_row[:] = np.nan
                self.cell_num_in_cluster.append(one_row)

                x_axis = np.zeros(frame_number, dtype = float)
                x_axis[:] = np.nan
                self.cell_num_in_cluster_x.append(x_axis)

                one_row = []
                for i in range(frame_number):
                    one_row.append([])
                self.clusterContour_img_mat.append(one_row)

            self.clusters_eating_or_not = np.zeros(nCluster)

            for i in range(len(clustering.labels_)):
                if(clustering.labels_[i] > -1):
                    cluster_points[clustering.labels_[i]].append([points[i][0] - 5, points[i][1] - 5])
                    cluster_points[clustering.labels_[i]].append([points[i][0] + 5, points[i][1] - 5])
                    cluster_points[clustering.labels_[i]].append([points[i][0] - 5, points[i][1] + 5])
                    cluster_points[clustering.labels_[i]].append([points[i][0] + 5, points[i][1] + 5])
                    
                    self.clusters[clustering.labels_[i]].append(i)

                    self.tracks[i].cluster_index = clustering.labels_[i]

        for i in range(len(cluster_points)):
            if(len(cluster_points[i]) > 0):
                hull = cv2.convexHull(np.array(cluster_points[i], np.int32))

                (x, y), radius = cv2.minEnclosingCircle(np.array(cluster_points[i], np.int32))
                cv2.drawContours(frame, [hull], -1, (0, 0, 255), 1)
                cv2.putText(frame, str(i), (int(x + radius), int(y) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                
                self.clusterContour_img_mat[i][frame_index].append([hull])

                self.cluster_img_mat[i][frame_index * 3 + 0] = x
                self.cluster_img_mat[i][frame_index * 3 + 1] = y
                self.cluster_img_mat[i][frame_index * 3 + 2] = radius    

                self.cell_num_in_cluster[i][frame_index] = len(cluster_points[i]) >> 2
                self.cell_num_in_cluster_x[i][frame_index] = frame_index

    def Save(self, out_path):
        for i in range(len(self.clusters)):
            arr10_x = self.cell_num_in_cluster_x[i]
            arr10_y = self.cell_num_in_cluster[i]
            arr10_x = arr10_x[~np.isnan(arr10_x)]
            arr10_y = arr10_y[~np.isnan(arr10_y)]

            cluster_max = np.nanmax(self.cell_num_in_cluster[i])
            cluster_min = np.nanmin(self.cell_num_in_cluster[i])
            cluster_mean = np.nanmean(self.cell_num_in_cluster[i])

            if(cluster_max > 10 and (cluster_max - cluster_min)/cluster_max > 0.5 and len(arr10_x) > 30 and cluster_mean > 7):

                arr10_x = arr10_x.reshape((-1, 1))
                model = LinearRegression().fit(arr10_x, arr10_y)
                print("%d, %.6s, %d" % (i, model.coef_[0], self.cell_num_in_cluster[i][0]))
                if(model.coef_[0] < -0.04):
                    print("%d, %.6s" % (i, model.coef_[0]))
                    print(self.cell_num_in_cluster[i])
                    self.clusters_eating_or_not[i] = 1

            if(cluster_mean <= 7):
                self.clusters_eating_or_not[i] = -1

        file = open(out_path + "tracks_live.txt", "w")

        for cluster_inf in self.cell_num_in_cluster:
            # print(cluster_inf)
            for i in range(frame_number):
                file.write('%s, ' % cluster_inf[i])
            file.write('\n')

        file.close()

    def mark_cells(self, frame, frame_index):

        print("frame_index: ", frame_index)

        for i in range(len(self.cluster_img_mat)):

            if(len(self.clusterContour_img_mat[i][frame_index]) > 0):

                x3 = self.cluster_img_mat[i][frame_index * 3 + 0]
                y3 = self.cluster_img_mat[i][frame_index * 3 + 1]
                radius = self.cluster_img_mat[i][frame_index * 3 + 2]               
                num = self.cell_num_in_cluster[i][frame_index]

                if(num > 4):
                    if(self.clusters_eating_or_not[i] == 1):
                        cv2.drawContours(frame, self.clusterContour_img_mat[i][frame_index][0], -1, (0, 0, 255), 1)
                        cv2.putText(frame, str(int(i)), (int(x3 + radius), int(y3) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))
                    elif(self.clusters_eating_or_not[i] == 0):
                        cv2.drawContours(frame, self.clusterContour_img_mat[i][frame_index][0], -1, (0, 255, 255), 1)
                        cv2.putText(frame, str(int(i)), (int(x3 + radius), int(y3) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))
                    else:
                        pass

        cv2.putText(frame, str(frame_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))

