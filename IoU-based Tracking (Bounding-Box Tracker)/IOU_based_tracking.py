import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import os

dt = pd.read_csv("../ADL-Rundle-6/det/Yolov5l/det.txt", sep="\s+")
#gt = pd.read_csv("../ADL-Rundle-6/gt/gt.txt")
dt = dt.T.reset_index().T.reset_index(drop=True)
print(dt)

tracks = {}
next_id = 1
max_missed_frames = 10
id_mapping = {}

def get_array_bounding(frame):
    rows_dt = dt[dt[0] == frame]
    rows_gt = dt[dt[0] == frame - 1]
    dt_x, dt_y, dt_w, dt_h = rows_dt[2].values, rows_dt[3].values, rows_dt[4].values, rows_dt[5].values
    gt_x, gt_y, gt_w, gt_h = rows_gt[2].values, rows_gt[3].values, rows_gt[4].values, rows_gt[5].values
    coord_dt = [[dt_x[i], dt_y[i], dt_w[i], dt_h[i]] for i in range(len(dt_x))]
    coord_gt = [[gt_x[i], gt_y[i], gt_w[i], gt_h[i]] for i in range(len(gt_x))]
    conf_dt = rows_dt[6].values
    return coord_dt, coord_gt, conf_dt

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

def predict_bbox(track):
    if "prev_bbox" not in track:
        return track["bbox"]
    prev_bbox = track["prev_bbox"]
    curr_bbox = track["bbox"]
    dx = curr_bbox[0] - prev_bbox[0]
    dy = curr_bbox[1] - prev_bbox[1]
    return [curr_bbox[0] + dx, curr_bbox[1] + dy, curr_bbox[2], curr_bbox[3]]

def similarity_matrix_with_prediction(tracks, coord_dt):
    similarity = np.zeros((len(tracks), len(coord_dt)))
    for i, track_id in enumerate(tracks.keys()):
        predicted_bbox = predict_bbox(tracks[track_id])
        for j, det_bbox in enumerate(coord_dt):
            similarity[i, j] = bb_intersection_over_union(predicted_bbox, det_bbox)
    return similarity

def weighted_similarity_matrix(similarity, conf_scores):
    for j in range(similarity.shape[1]):
        similarity[:, j] *= conf_scores[j]
    return similarity

def apply_motion_penalty(similarity, tracks, coord_dt):
    for i, track_id in enumerate(tracks.keys()):
        track_bbox = tracks[track_id]["bbox"]
        for j, det_bbox in enumerate(coord_dt):
            center_track = (track_bbox[0] + track_bbox[2] / 2, track_bbox[1] + track_bbox[3] / 2)
            center_det = (det_bbox[0] + det_bbox[2] / 2, det_bbox[1] + det_bbox[3] / 2)
            distance = np.linalg.norm(np.array(center_track) - np.array(center_det))
            similarity[i, j] -= 0.001 * distance
    return similarity

def cal_similarity(similarity):
    cost_matrix = 1 - similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def track_management(similarity, row_ind, col_ind, coord_dt, conf_dt):
    global next_id, tracks, max_missed_frames
    matched_tracks = set()
    unmatched_detections = set(range(len(coord_dt)))
    unmatched_tracks = set(tracks.keys())

    for i, j in zip(row_ind, col_ind):
        if i >= len(tracks) or j >= len(coord_dt):
            continue
        if similarity[i, j] > 0.3:
            track_ids = list(tracks.keys())
            track_id = track_ids[i]
            tracks[track_id]["prev_bbox"] = tracks[track_id]["bbox"]
            tracks[track_id]["bbox"] = coord_dt[j]
            tracks[track_id]["missed_frames"] = 0
            matched_tracks.add(track_id)
            unmatched_detections.discard(j)
            unmatched_tracks.discard(track_id)
    for track_id in unmatched_tracks:
        tracks[track_id]["missed_frames"] += 1
        if tracks[track_id]["missed_frames"] > max_missed_frames:
            del tracks[track_id]
    for idx in unmatched_detections:
        tracks[next_id] = {"bbox": coord_dt[idx], "missed_frames": 0}
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

def process_video(input_folder, detections_path, output_video_path, result_path):
    global tracks, next_id
    frame_idx = dt[0].unique()
    first_frame = cv2.imread(f"{input_folder}/000001.jpg")
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter("../" + output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    for frame in frame_idx:
        frame = int(frame)
        img_path = f"{input_folder}/{str(frame).zfill(6)}.jpg"
        frame_img = cv2.imread(img_path)
        if not os.path.exists(img_path):
            continue

        coord_dt, coord_gt, conf_dt = get_array_bounding(frame)
        similarity = similarity_matrix_with_prediction(tracks, coord_dt)
        similarity = weighted_similarity_matrix(similarity, conf_dt)
        similarity = apply_motion_penalty(similarity, tracks, coord_dt)
        row_ind, col_ind = cal_similarity(similarity)
        tracks = track_management(similarity, row_ind, col_ind, coord_dt, conf_dt)
        save_tracking_results(tracks, result_path, frame)
        for track_id, track in tracks.items():
            bbox = track["bbox"]
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_img, f"ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(frame_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()

input_folder = "../ADL-Rundle-6/img1"
detections_path = "../ADL-Rundle-6/det/Yolovl/det.txt"
output_video_path = "tracking_output.avi"
results_file = "../gt.txt"

process_video(input_folder, detections_path, output_video_path, results_file)


"""
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import os

dt = pd.read_csv("../ADL-Rundle-6/det/Yolov5l/det.txt", sep="\s+")
#gt = pd.read_csv("../ADL-Rundle-6/gt/gt.txt")
dt = dt.T.reset_index().T.reset_index(drop=True)
print(dt)

frame = 1

def get_array_bounding(frame):
    rows_dt = dt[dt[0] == frame]
    rows_gt = dt[dt[0] == frame - 1]
    #print(rows_gt)
    dt_x, dt_y, dt_w, dt_h = rows_dt[2].values, rows_dt[3].values, rows_dt[4].values, rows_dt[5].values
    gt_x, gt_y, gt_w, gt_h = rows_gt[2].values, rows_gt[3].values, rows_gt[4].values, rows_gt[5].values
    #print(gt_x)
    #print(dt_x)
    #coord_dt = dt_x[[2, 3, 4, 5]].values.tolist()
    #print(coord_dt)
    coord_dt = [[dt_x[i], dt_y[i], dt_w[i], dt_h[i]] for i in range(len(dt_x))]
    coord_gt = [[gt_x[i], gt_y[i], gt_w[i], gt_h[i]] for i in range(len(gt_x))]
    #print(coord_gt)
    #print(coord_dt)
    return coord_dt, coord_gt

coord_dt, coord_gt = get_array_bounding(frame)
#print(coord_dt)
print(coord_gt)

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

def predict_bbox(track):
    if "prev_bbox" not in track:
        return track["bbox"]  # No motion data yet
    prev_bbox = track["prev_bbox"]
    curr_bbox = track["bbox"]

    dx = curr_bbox[0] - prev_bbox[0]
    dy = curr_bbox[1] - prev_bbox[1]
    predicted_bbox = [
        curr_bbox[0] + dx,
        curr_bbox[1] + dy,
        curr_bbox[2],
        curr_bbox[3],
    ]
    return predicted_bbox


def similarity_matrix(coord_dt, coord_gt):
    similarity = np.zeros((len(coord_gt), len(coord_dt)))
    for i, gt in enumerate(coord_gt):
        for j, dtt in enumerate(coord_dt):
            iou = bb_intersection_over_union(gt, dtt)
            similarity[i, j] = iou
    return similarity

similarity = similarity_matrix(coord_dt, coord_gt)
print(similarity)

def cal_similarity(similarity):
    n_similarity = 1 - similarity
    row_ind, col_ind = linear_sum_assignment(n_similarity)
    return row_ind, col_ind

row_ind, col_ind = cal_similarity(similarity)
print(row_ind, col_ind)

tracks = {}
next_id = 1
max_missed_frames = 2

def track_management(similarity, row_ind, col_ind, coord_dt, coord_gt):
    global next_id, tracks, max_missed_frames
    matched_tracks = set()
    unmatched_detections = set(range(len(coord_dt)))
    unmatched_tracks = set(tracks.keys())
    for i, j in zip(row_ind, col_ind):
        if i >= len(coord_gt) or j >= len(coord_dt):
            continue
        if similarity[i][j] > 0.3:
            track_ids = list(tracks.keys())
            track_id = track_ids[i]
            for t in tracks:
                if tracks[t]["bbox"] == coord_gt[i]:
                    track_id = t
                    break
            print(track_id)
            print(f"Matching track index {i} with detection index {j}")
            print(f"track to frame-1 compare : {tracks[track_id]} and {coord_gt[i]}")
            print(f"Track bbox: {coord_gt[i]}, Detection bbox: {coord_dt[j]}")
            tracks[track_id]["bbox"] = coord_dt[j]
            tracks[track_id]["missed_frames"] = 0
            matched_tracks.add(track_id)
            unmatched_detections.discard(j)
            unmatched_tracks.discard(track_id)

    for track_id in unmatched_tracks:
        tracks[track_id]["missed_frames"] += 1
        if tracks[track_id]["missed_frames"] > max_missed_frames:
            del tracks[track_id]
    for idx in unmatched_detections:
        tracks[next_id] = {"bbox": coord_dt[idx], "missed_frames": 0}
        next_id += 1
    return tracks

track = track_management(similarity, row_ind, col_ind, coord_dt, coord_gt)
print(tracks)

def save_tracking_results(tracks, output_file, frame):
    with open(output_file, 'a') as f:
        for track_id, track in tracks.items():
            bbox = track["bbox"]
            x, y, w, h = bbox
            conf = 1
            x_world, y_world, z_world = -1, -1, -1
            line = f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf},{x_world},{y_world},{z_world}\n"
            f.write(line)


def process_video(input_folder, detections_path, output_video_path, result_path):
    global tracks, next_id
    frame_idx = dt[0].unique()
    # print(frame_idx)
    # frame_idx = sorted(frame_idx)
    first_frame = cv2.imread(f"{input_folder}/000001.jpg")
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter("../" + output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    for frame in frame_idx:
        frame = int(frame)
        if frame == frame_idx[-1]:
            break
        img_path = f"{input_folder}/{str(frame).zfill(6)}.jpg"
        frame_img = cv2.imread(img_path)
        if not os.path.exists(img_path):
            print(f"Frame {frame}: File not found {img_path}")
            continue
        coord_dt, coord_gt = get_array_bounding(frame)
        # coord_gt = [tracks[t]["bbox"] for t in tracks]
        similarity = similarity_matrix(coord_gt, coord_dt)
        row_ind, col_ind = cal_similarity(similarity)
        tracks = track_management(similarity, row_ind, col_ind, coord_dt, coord_gt)
        save_tracking_results(tracks, results_file, frame)
        for track_id, track in tracks.items():
            bbox = track["bbox"]
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_img, f"ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
"""