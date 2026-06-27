# Multi-Object Tracking with Kalman Filters

A progressive series of 5 exercises : starting from a single-object Kalman tracker, then adding IoU-based multi-object tracking, Kalman-guided prediction, appearance re-identification (ReID), and finally a real-time deep-learning detector.

## TP1 — Single Object Tracking (Kalman Filter)

A centroid tracker for one object, using a constant-velocity Kalman filter on the state `x = [cx, cy, vx, vy]ᵀ` (centroid position + velocity).

**Prediction:**
```text
x_k = A·x_(k-1) + B·u
P_k = A·P_(k-1)·Aᵀ + Q
```

**Update** (with measured centroid `z_k`):
```text
S_k = H·P_k·Hᵀ + R
K_k = P_k·Hᵀ·S_k⁻¹
x_k = x_k + K_k·(z_k - H·x_k)
P_k = (I - K_k·H)·P_k
```

Where `A` is the constant-velocity transition matrix, `H` extracts position from the state, `Q` and `R` are the process and measurement noise covariances. Detections come from a simple blob detector.

## TP2 — IoU-Based Multi-Object Tracking

Tracks pedestrians on bounding-box overlap, using detections pre-computed by Yolov5s/Yolov5l (loaded from a `det.txt` file).

```text
IoU(A, B) = area(A ∩ B) / area(A ∪ B)
```

A similarity matrix is built between existing tracks and new detections, then matched optimally with the **Hungarian algorithm** on the cost `1 - IoU`. 

Track management has three cases :
- matched (similarity above threshold) -> update the track's bounding box, reset its missed-frame counter
- unmatched track -> increment its missed-frame counter, delete it once a max is exceeded
- unmatched detection -> create a new track with a new ID

This version also weighs the similarity by detection confidence and applies a small penalty proportional to the distance between box centers.

## TP3 — Kalman-Guided IoU Tracking

Combines TP1 and TP2 : each track now carries its own Kalman filter (bounding boxes are converted to centroids to use it, then back to boxes). 

The flow per frame is :

1. Predict each track's centroid with its Kalman filter -> convert to a predicted bounding box.
2. Compute IoU between **predicted** boxes and new detections -> build the similarity matrix.
3. Solve the assignment with the Hungarian algorithm.
4. Update each matched track's Kalman filter with the new detection.
5. Run track management (same 3 cases as TP2).

## TP4 — Appearance-Aware (ReID) Tracking

Adds visual appearance to the matching criterion, since geometry (IoU) alone struggles when boxes overlap or briefly disappear. 

Each detection's image patch is passed through an **OSNet** re-identification model (`reid_osnet_x025_market1501.onnx`) to get a feature embedding, and combined with IoU into a single score :

```text
score = α · IoU(track, detection) + β · appearance_similarity(track, detection)
```

`appearance_similarity` is either the cosine similarity between embeddings, or `1 / (1 + L2_distance)`, depending on the version. The Hungarian algorithm then matches on this combined score instead of IoU alone.

## TP5 — Real-Time Detector Extension

Replaces the pre-computed `det.txt` detections with a live, lightweight **YOLOv8n** detector (`ultralytics`), keeping the rest of the TP4 pipeline (Kalman + IoU + ReID) unchanged. This removes the dependency on offline detection files and runs detection + tracking end-to-end on raw frames.

## File → Exercise Mapping

| TP | Role |
|---|---|
| TP1 | Kalman filter class, blob detector, single-object tracking loop |
| TP2 | IoU + Hungarian multi-object tracker |
| TP3 | Kalman-guided IoU tracker |
| TP4 | + OSNet ReID, cosine similarity |
| TP5 | + YOLOv8n real-time detector |

## Implementation Notes

- **TP1**: the Kalman state must be a column vector (`shape (4,1)`), not a flat array — a common source of broadcasting bugs.
- **TP2**: track management is sensitive to the similarity threshold and `max_missed_frames`; values had to be relaxed (threshold `0.7 → 0.3`, missed frames `2 → 10`) to avoid constantly spawning new IDs.
- **TP3**: ID switches still occur when a box's size changes quickly, due to imperfect Kalman prediction; box dimensions are therefore updated conservatively to keep IDs stable.
- **TP4**: the `α`/`β` weighting and the ReID confidence threshold required manual tuning; the ONNX preprocessing pipeline (resize, normalization, channel order) needed to match the OSNet training setup exactly.
- **TP5**: the bonus MOT-metric evaluation (`seqmap`-based) could not be completed due to file-path/format issues with the evaluation toolkit.

## Results

Each exercise's output video is available on the project's Git repository on the folder `Video_Result`
