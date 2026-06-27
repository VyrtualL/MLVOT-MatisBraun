import cv2
import numpy as np

from Detector import detect
from KalmanFilter import KalmanFilter

dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas = 0.1, 1, 1, 1, 0.1, 0.1
filter = KalmanFilter(dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)
cap = cv2.VideoCapture("randomball.avi")
output_video_path = "tracking_output_tp1.avi"
trajectory = []
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    centroid = detect(frame)
    if len(centroid) > 0:
        pred = filter.predict()
        #print(pred.shape)
        trajectory.append((int(centroid[0][0][0]), int(centroid[0][1][0])))
        x, y = int(pred[0]), int(pred[1])
        cv2.circle(frame, (int(centroid[0][0]), int(centroid[0][1])), 5, (0, 128, 0), 1)
        up = filter.update(centroid[0])
        x2, y2 = int(up[0]), int(up[1])
        cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 1)
        cv2.rectangle(frame, (x2 - 15, y2 - 15), (x2 + 15, y2 + 15), (255, 0, 0), 1)

    if len(trajectory) > 1:
        cv2.polylines(frame, [np.array(trajectory, dtype=np.int32)], isClosed=False, color=(25, 25, 112), thickness=1)

    out.write(frame)
    cv2.imshow("Object tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()