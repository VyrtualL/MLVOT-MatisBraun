import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import onnxruntime
import time

###############################################################################
#   Global Config
################################################################################

IMG_FOLDER = "../ADL-Rundle-6/img1"
OUTPUT_VIDEO = "tracking_output.avi"
RESULTS_TXT = "../gt.txt"
REID_MODEL = "../reid_osnet_x025_market1501.onnx"

YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5
YOLO_CLASSES = [0]

ROI_HEIGHT = 128
ROI_WIDTH = 64
ROI_MEANS = [0.485, 0.456, 0.406]
ROI_STDS = [0.229, 0.224, 0.225]


MAX_MISSED_FRAMES = 5
ALPHA = 0.3 # Weight for IoU
BETA = 0.7 # Weight for appearance

# Kalman Filter default parameters
kdt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas = 0.1, 1, 1, 1, 0.1, 0.1

################################################################################
#   YOLO Detector Initialization
################################################################################

model = YOLO(YOLO_MODEL_PATH)
model.conf = YOLO_CONFIDENCE
model.classes = YOLO_CLASSES


def detect_pedestrians(frame_img):
    results = model.predict(source=frame_img, verbose=False)
    coord_dt = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()[:4]
        conf = box.conf[0].item()
        cls = box.cls[0].item()
        if conf < YOLO_CONFIDENCE:
            continue
        if cls != 0:
            continue

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        coord_dt.append([x1, y1, w, h])

    return coord_dt


################################################################################
#   Kalman Filter
################################################################################

class KalmanFilter():
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_sdt_meas = x_sdt_meas
        self.y_sdt_meas = y_sdt_meas
        self.u = np.asarray([[self.u_x], [self.u_y]])
        self.x_k = np.zeros((4, 1), dtype=np.float32)

        self.A = np.eye(4)
        self.A[0, 2] = dt
        self.A[1, 3] = dt
        self.B = np.asarray([[0.5 * (dt ** 2), 0], [0, 0.5 * (dt ** 2)], [dt, 0], [0, dt]], dtype=np.float32)
        self.H = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        self.Q = np.zeros((4, 4), dtype=np.float32)
        self.Q[0, 0] = (dt ** 4) / 4
        self.Q[0, 2] = (dt ** 3) / 2
        self.Q[1, 1] = (dt ** 4) / 4
        self.Q[1, 3] = (dt ** 3) / 2
        self.Q[2, 0] = (dt ** 3) / 2
        self.Q[2, 2] = (dt ** 2)
        self.Q[3, 1] = (dt ** 3) / 2
        self.Q[3, 3] = (dt ** 2)
        self.Q *= (std_acc ** 2)
        self.R = np.asarray([[x_sdt_meas, 0], [0, y_sdt_meas]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)

    def predict(self):
        self.x_k = self.A @ self.x_k + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_k.flatten()

    def update(self, z_k):
        z_k = z_k.reshape((2, 1))
        S_k = self.H @ self.P @ self.H.T + self.R
        K_k = self.P @ self.H.T @ np.linalg.inv(S_k)
        y_k = z_k - (self.H @ self.x_k)
        self.x_k = self.x_k + K_k @ y_k
        I = np.eye(self.H.shape[1], dtype=np.float32)
        self.P = (I - K_k @ self.H) @ self.P
        return self.x_k.flatten()


################################################################################
#   ReID Model & Appearance Functions
###############################################################################

reid_session = onnxruntime.InferenceSession(REID_MODEL, providers=['CPUExecutionProvider'])



def extract_reid_feature(frame_img, bbox):
    x, y, w, h = [int(v) for v in bbox]
    patch = frame_img[y:y + h, x:x + w]

    if patch.shape[0] <= 0 or patch.shape[1] <= 0:
        return np.zeros(512, dtype=np.float32)

    patch = cv2.resize(patch, (ROI_WIDTH, ROI_HEIGHT))
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = patch.astype(np.float32)
    patch[..., 0] = (patch[..., 0] / 255.0 - ROI_MEANS[0]) / ROI_STDS[0]
    patch[..., 1] = (patch[..., 1] / 255.0 - ROI_MEANS[1]) / ROI_STDS[1]
    patch[..., 2] = (patch[..., 2] / 255.0 - ROI_MEANS[2]) / ROI_STDS[2]
    patch = np.moveaxis(patch, -1, 0)
    patch = np.expand_dims(patch, axis=0).astype(np.float32)

    input_name = reid_session.get_inputs()[0].name
    outputs = reid_session.run(None, {input_name: patch})
    feat = outputs[0]
    feat = feat[0]
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat /= norm
    return feat




def euclidean_distance(feat_a, feat_b):
    return np.linalg.norm(feat_a - feat_b)


################################################################################
#   IoU + Appearance Combination
################################################################################

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def combined_cost_matrix(tracks, track_ids, coord_dt, frame_img, alpha=ALPHA, beta=BETA):
    num_tracks = len(track_ids)
    num_dets = len(coord_dt)
    cost_mat = np.zeros((num_tracks, num_dets), dtype=np.float32)
    det_features = []
    for det_bbox in coord_dt:
        det_feat = extract_reid_feature(frame_img, det_bbox)
        det_features.append(det_feat)
    for i, t_id in enumerate(track_ids):
        track_bbox = tracks[t_id]["bbox"]
        track_feat = tracks[t_id]["reid_feat"]
        for j, det_bbox in enumerate(coord_dt):
            iou_val = bb_intersection_over_union(track_bbox, det_bbox)
            if track_feat is None:
                combined_similarity = iou_val
            else:
                det_feat = det_features[j]
                dist = euclidean_distance(track_feat, det_feat)
                appearance_sim = 1.0 / (1.0 + dist)
                combined_similarity = alpha * iou_val + beta * appearance_sim
            cost_mat[i, j] = 1.0 - combined_similarity
    return cost_mat

def bbox_to_centroid(bbox):
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return np.array([cx, cy])

def centroid_to_bbox(cx, cy, w, h):
    x = cx - w / 2
    y = cy - h / 2
    return [x, y, w, h]


###############################################################################
#   Save Results in MOT Format
#############################################################################

def save_tracking_results(tracks, output_file, frame):
    """
    Format (MOT Challenge):
      frame, ID, x, y, w, h, conf=1, x_world=-1, y_world=-1, z_world=-1
    """
    with open(output_file, 'a') as f:
        for t_id, t_data in tracks.items():
            bbox = t_data["bbox"]
            x, y, w, h = bbox
            conf = 1
            line = f"{frame},{t_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf},-1,-1,-1\n"
            f.write(line)


################################################################################
#   Main Tracking Loop
################################################################################

def process_images(img_folder, output_video_path, result_path):
    if os.path.exists(result_path):
        os.remove(result_path)


    img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])
    if not img_files:
        return
    first_img_path = os.path.join(img_folder, img_files[0])
    first_frame = cv2.imread(first_img_path)
    if first_frame is None:
        return
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    tracks = {}
    next_id = 1
    frame_count = 0
    start_time = time.time()
    for img_name in img_files:
        frame_count += 1
        frame_path = os.path.join(img_folder, img_name)
        frame_img = cv2.imread(frame_path)
        if frame_img is None:
            continue
        coord_dt = detect_pedestrians(frame_img)
        track_ids = list(tracks.keys())
        for t_id in track_ids:
            pred = tracks[t_id]["kalman"].predict()
            cx_pred, cy_pred = pred[0], pred[1]
            w, h = tracks[t_id]["bbox"][2], tracks[t_id]["bbox"][3]
            predicted_bbox = centroid_to_bbox(cx_pred, cy_pred, w, h)
            tracks[t_id]["bbox"] = predicted_bbox
        if len(tracks) == 0:
            for d_bbox in coord_dt:
                cx, cy = bbox_to_centroid(d_bbox)
                new_kf = KalmanFilter(kdt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)
                new_kf.x_k = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                reid_feat = extract_reid_feature(frame_img, d_bbox)
                tracks[next_id] = {"bbox": d_bbox, "missed_frames": 0, "kalman": new_kf, "reid_feat": reid_feat}
                next_id += 1
            for t_id, t_data in tracks.items():
                x, y, w, h = [int(v) for v in t_data["bbox"]]
                cv2.rectangle(frame_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_img, f"ID {t_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            save_tracking_results(tracks, result_path, frame_count)
            out.write(frame_img)
            continue

        track_ids = list(tracks.keys())
        cost_mat = combined_cost_matrix(tracks, track_ids, coord_dt, frame_img, ALPHA, BETA)
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        matched_tracks = set()
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(coord_dt)))
        for i, j in zip(row_ind, col_ind):
            if i >= len(track_ids) or j >= len(coord_dt):
                continue
            t_id = track_ids[i]
            if cost_mat[i, j] < 0.8:
                det_bbox = coord_dt[j]
                det_cx, det_cy = bbox_to_centroid(det_bbox)
                z_k = np.array([det_cx, det_cy], dtype=np.float32)
                updated_state = tracks[t_id]["kalman"].update(z_k)
                old_w, old_h = tracks[t_id]["bbox"][2], tracks[t_id]["bbox"][3]
                new_w, new_h = det_bbox[2], det_bbox[3]
                tracks[t_id]["bbox"] = centroid_to_bbox(updated_state[0], updated_state[1], new_w, new_h)
                tracks[t_id]["missed_frames"] = 0
                new_feat = extract_reid_feature(frame_img, det_bbox)
                tracks[t_id]["reid_feat"] = new_feat
                matched_tracks.add(t_id)
                unmatched_tracks.discard(t_id)
                unmatched_dets.discard(j)

        for t_id in unmatched_tracks:
            tracks[t_id]["missed_frames"] += 1
            if tracks[t_id]["missed_frames"] > MAX_MISSED_FRAMES:
                del tracks[t_id]
        for d_id in unmatched_dets:
            d_bbox = coord_dt[d_id]
            cx, cy = bbox_to_centroid(d_bbox)
            new_kf = KalmanFilter(kdt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)
            new_kf.x_k = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
            reid_feat = extract_reid_feature(frame_img, d_bbox)
            tracks[next_id] = {"bbox": d_bbox, "missed_frames": 0, "kalman": new_kf, "reid_feat": reid_feat}
            next_id += 1
        for t_id, t_data in tracks.items():
            x, y, w, h = [int(v) for v in t_data["bbox"]]
            cv2.rectangle(frame_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_img, f"ID {t_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        save_tracking_results(tracks, result_path, frame_count)
        out.write(frame_img)
        cv2.imshow("Tracking", frame_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps = frame_count / elapsed


if __name__ == "__main__":
    process_images(IMG_FOLDER, OUTPUT_VIDEO, RESULTS_TXT)
