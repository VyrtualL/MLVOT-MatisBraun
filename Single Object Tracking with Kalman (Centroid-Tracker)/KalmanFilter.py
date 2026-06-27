import numpy as np


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
