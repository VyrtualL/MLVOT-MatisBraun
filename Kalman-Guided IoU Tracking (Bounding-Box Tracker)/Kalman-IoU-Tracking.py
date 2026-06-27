import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

dt = pd.read_csv("../ADL-Rundle-6/det/Yolov5l/det.txt", sep="\s+")
dt = dt.T.reset_index().T.reset_index(drop=True)
frame = 1

import numpy as np
def bbox_to_centroid(bbox):
    x, y, w, h = bbox
    cx = x + w/2
    cy = y + h/2
    return np.array([cx, cy])

def centroid_to_bbox(cx, cy, w, h):
    x = cx - w/2
    y = cy - h/2
    return [x, y, w, h]

class KalmanFilter():
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_sdt_meas = x_sdt_meas
        self.y_sdt_meas = y_sdt_meas
        self.u = np.asarray([[self.u_x], [self.u_y]])
        self.x_k = np.asarray([[0], [0], [0], [0]])
        self.A = np.eye(4)
        self.A[0][2] = dt
        self.A[1][3] = dt
        self.B = np.asarray([[((1/2) * (dt**2)), 0], [0, ((1/2) * (dt**2))], [dt, 0], [0, dt]])
        self.H = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.zeros((4, 4))
        self.Q[0][0] = (dt**4) / 4
        self.Q[0][2] = (dt**3) / 2
        self.Q[1][1] = (dt**4) / 4
        self.Q[1][3] = (dt**3) / 2
        self.Q[2][0] = (dt**3) / 2
        self.Q[2][2] = (dt**2)
        self.Q[3][1] = (dt**3) / 2
        self.Q[3][3] = (dt**2)
        self.Q = self.Q * (std_acc ** 2)
        self.R = np.asarray([[x_sdt_meas, 0], [0, y_sdt_meas]])
        self.P = np.eye(self.A.shape[1])
    def predict(self):
        self.x_k = (self.A @ self.x_k) + (self.B @ self.u)
        self.P = ((self.A @ self.P) @ self.A.T) + self.Q
        return self.x_k.flatten()
    def update(self, z_k):
        s_k = ((self.H @ self.P) @ self.H.T) + self.R
        k_k = ((self.P @ self.H.T) @ (np.linalg.inv(s_k)))
        self.x_k = self.x_k + k_k @ (z_k - (self.H @ self.x_k))
        tmp = k_k @ self.H
        self.P = (np.eye(self.H.shape[1]) - tmp) @ self.P
        return self.x_k.flatten()
    
import numpy as np
import cv2
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    img_edges = cv2.Canny(gray,  50, 190, 3)
    #cv2.imshow('img_edges', img_edges)
    ret, img_thresh = cv2.threshold(img_edges, 254, 255,cv2.THRESH_BINARY)
    #cv2.imshow('img_thresh', img_thresh)
    contours, _= cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_radius_thresh= 3
    max_radius_thresh= 30
    centers=[]
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
    cv2.imshow('contours', img_thresh)
    return centers



def get_array_bounding(frame):
    rows_dt = dt[dt[0] == frame]
    rows_gt = dt[dt[0] == frame - 1]
    dt_x, dt_y, dt_w, dt_h = rows_dt[2].values, rows_dt[3].values, rows_dt[4].values, rows_dt[5].values
    gt_x, gt_y, gt_w, gt_h = rows_gt[2].values, rows_gt[3].values, rows_gt[4].values, rows_gt[5].values
    coord_dt = [[dt_x[i], dt_y[i], dt_w[i], dt_h[i]] for i in range(len(dt_x))]
    coord_gt = [[gt_x[i], gt_y[i], gt_w[i], gt_h[i]] for i in range(len(gt_x))]
    return coord_dt, coord_gt
def predict_tracks(tracks, kalman_filters):
    pred = []
    for id, track in tracks.items():
        if id not in kalman_filters:
            kalman_filters[id] = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1)
            kalman_filters[id].update(np.array([[track["bbox"][0]], [track["bbox"][1]]]))
        pred = kalman_filters[id].predict()
        pred.append([pred[0], pred[1], track["bbox"][2], track["bbox"][3]])
    return pred
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = ((boxA[0] + boxA[2]) - boxA[0] + 1) * ((boxA[1] + boxA[3]) - boxA[1] + 1)
    boxBArea = ((boxB[0] + boxB[2]) - boxB[0] + 1) * ((boxB[1] + boxB[3]) - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
def similarity_matrix(coord_dt, coord_gt):
    similarity = np.zeros((len(coord_gt), len(coord_dt)))
    for i, gt in enumerate(coord_gt):
        for j, dtt in enumerate(coord_dt):
            iou = bb_intersection_over_union(gt, dtt)
            similarity[i, j] = iou
    return similarity
def cal_similarity(similarity):
    n_similarity = 1 - similarity
    row_ind, col_ind = linear_sum_assignment(n_similarity)
    return row_ind, col_ind
tracks = {}
kalman_filters = {}
next_id = 1
max_missed_frames = 2
kdt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas = 0.1, 1, 1, 1, 0.1, 0.1
def track_management(similarity, row_ind, col_ind, coord_dt, pred_boxes):
    global next_id, tracks, max_missed_frames
    matched_tracks = set()
    unmatched_detections = set(range(len(coord_dt)))
    unmatched_tracks = set(tracks.keys())

    for i, j in zip(row_ind, col_ind):
        if i >= len(pred_boxes) or j >= len(coord_dt):
            continue
        if similarity[i][j] > 0.3:
            track_id = list(tracks.keys())[i]
            det_bbox = coord_dt[j]
            det_cx, det_cy = bbox_to_centroid(det_bbox)
            z_k = np.array([[det_cx],[det_cy]])
            updated_state = tracks[track_id]["kalman"].update(z_k)
            w, h = tracks[track_id]["bbox"][2], tracks[track_id]["bbox"][3]
            tracks[track_id]["bbox"] = centroid_to_bbox(updated_state[0], updated_state[1], w, h)
            tracks[track_id]["missed_frames"] = 0
            matched_tracks.add(track_id)
            unmatched_detections.discard(j)
            unmatched_tracks.discard(track_id)
    for id in unmatched_tracks:
        tracks[id]["missed_frames"] += 1
        if tracks[id]["missed_frames"] > max_missed_frames:
            del tracks[id]
    for id2 in unmatched_detections:
        det_bbox = coord_dt[id2]
        cx, cy = bbox_to_centroid(det_bbox)
        new_kf = KalmanFilter(kdt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)
        new_kf.x_k = np.array([[cx],[cy],[0],[0]])
        tracks[next_id] = {"bbox": det_bbox, "missed_frames": 0, "kalman": new_kf}
        next_id += 1
    return tracks
def save_tracking_results(tracks, output_file, frame):
    with open(output_file, 'a') as f:
        for track_id, track in tracks.items():
            bbox = track["bbox"]
            x, y, w, h = bbox
            conf = 1
            x_world, y_world, z_world = -1, -1, -1
            line = f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf},{x_world},{y_world},{z_world}\n"
            f.write(line)
import os

def process_video(input_folder, detections_path, output_video_path, result_path):
    global tracks, next_id
    frame_idx = dt[0].unique()
    first_frame = cv2.imread(f"{input_folder}/000001.jpg")
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter("../" + output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    for frame in frame_idx:
        frame = int(frame)
        if frame == frame_idx[-1]:
            break
        img_path = f"{input_folder}/{str(frame).zfill(6)}.jpg"
        frame_img = cv2.imread(img_path)
        if frame_img is None:
            print(f"Frame {frame}: File not found {img_path}")
            continue
        coord_dt, coord_gt = get_array_bounding(frame)
        pred_boxes = []
        for t_id, trk in tracks.items():
            pred = trk["kalman"].predict()
            cx_pred, cy_pred = pred[0], pred[1]
            w, h = trk["bbox"][2], trk["bbox"][3]
            predicted_bbox = centroid_to_bbox(cx_pred, cy_pred, w, h)
            pred_boxes.append(predicted_bbox)

        if len(tracks) == 0:
            for idx, d_bbox in enumerate(coord_dt):
                cx, cy = bbox_to_centroid(d_bbox)
                new_kf = KalmanFilter(kdt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)
                new_kf.x_k = np.array([[cx],[cy],[0],[0]])
                tracks[next_id] = {"bbox": d_bbox, "missed_frames": 0, "kalman": new_kf}
                next_id += 1
            for track_id, track in tracks.items():
                bbox = track["bbox"]
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame_img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame_img, f"ID {track_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            save_tracking_results(tracks, result_path, frame)
            out.write(frame_img)
            continue
        similarity = similarity_matrix(coord_dt, pred_boxes)
        row_ind, col_ind = cal_similarity(similarity)
        tracks = track_management(similarity, row_ind, col_ind, coord_dt, pred_boxes)
        for track_id, track in tracks.items():
            bbox = track["bbox"]
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_img, f"ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        save_tracking_results(tracks, result_path, frame)
        cv2.imshow("Tracking", frame_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame_img)

    out.release()
    cv2.destroyAllWindows()
    print(f"Tracking video saved at: {output_video_path}")
input_folder = "../ADL-Rundle-6/img1"
detections_path = "../ADL-Rundle-6/det/Yolovl/det.txt"
output_video_path = "tracking_output.avi"
results_file = "../gt.txt"
process_video(input_folder, detections_path, output_video_path, results_file)

