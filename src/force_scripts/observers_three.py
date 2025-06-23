# ==============================================================================
# SENSE: A Force-Sensor-Free, Model-Based Framework for Estimating External 
#        Interaction Forces on Humanoid Robots
#
# Paper: IEEE RO-MAN 2025 (Regular Paper #507)
# Authors: Chouaib Fedsi, et al.
# Contact: chouaib.fedsi@univ-evry.fr
# Repository: https://github.com/chofdc/sense-humanoids
#
# This file is part of the SENSE framework and distributed for academic use.
# License: MIT (see LICENSE file)
# ==============================================================================

import numpy as np
import random

class Observer:
    def __init__(self, name, axis, dt, n, m):
        self.name = name
        self.axis = axis
        self.dt = dt
        self.n = n
        self.m = m
        self.xAct = np.zeros(n)
        self.initialized = False

    def init(self, x0=None, P0=None):
        self.initialized = True
        if x0 is not None:
            self.xAct = x0
        else:
            self.xAct = np.zeros(self.n)
        if P0 is not None:
            self.P = P0
        else:
            self.P = np.eye(self.n) * 100

    def update(self, U, Y):
        raise NotImplementedError

    def state(self):
        return {self.name: self.xAct}

    def uncertainty(self):
        return {self.name: np.zeros((self.n, self.n))}

class CompositeObserver(Observer):
    def __init__(self, name):
        super().__init__(name, axis=None, dt=None, n=None, m=None)
        self.children = []

    def add(self, obs):
        self.children.append(obs)

    def rem(self, obs):
        self.children.remove(obs)

    def update(self, U, Y):
        for child in self.children:
            child.update(U, Y)

    def state(self):
        state_map = {}
        for child in self.children:
            state_map.update(child.state())
        return state_map

    def uncertainty(self):
        uncertainty_map = {}
        for child in self.children:
            uncertainty_map.update(child.uncertainty())
        return uncertainty_map

class KalmanFilter(Observer):
#    def __init__(self, dt, A, B, C, R, sigma_jerk, sigma_ddfext, name, axis):
    def __init__(self, dt, A, B, C, R, Sigma, name, axis):
        super().__init__(name, axis, dt, A.shape[0], C.shape[0])
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        cov_input = np.diag(Sigma)
        self.Q = B @ cov_input @ B.T
        #self.P = np.eye(A.shape[0]) * 100  # Initial uncertainty

    def update(self, U, Y):
        y = np.array([Y[self.axis, 0], Y[self.axis, 2], Y[self.axis, 1]])

        # Prediction step
        self.xAct = self.A @ self.xAct
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Correction step
        K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        self.xAct = self.xAct + K @ (y - self.C @ self.xAct)
        self.P = self.P - K @ self.C @ self.P
        #self.P = (np.eye(self.n) - K @ self.C) @ self.P @ (np.eye(self.n) - K @ self.C).T + K @ self.R @ K.T

    def update_with_C(self, C, U, Y):
        self.C = C
        self.update(U, Y)

class KalmanFilterN(Observer):
#    def __init__(self, dt, A, B, C, R, sigma_jerk, sigma_ddfext, name, axis):
    def __init__(self, dt, A, B, C, R, Sigma, name, axis):
        super().__init__(name, axis, dt, A.shape[0], C.shape[0])
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        cov_input = np.diag(Sigma)
        self.Q = B @ cov_input @ B.T
        #self.P = np.eye(A.shape[0]) * 100  # Initial uncertainty

    def update(self, U, Y):
        y = np.array([Y[self.axis, 0], Y[self.axis, 2], Y[self.axis, 1]])
        if self.axis != 2:
            y = np.array([Y[self.axis, 0], Y[self.axis, 2], Y[self.axis, 1], Y[self.axis, 3]])
        # Prediction step
        self.xAct = self.A @ self.xAct
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Correction step
        K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        self.xAct = self.xAct + K @ (y - self.C @ self.xAct)
        self.P = self.P - K @ self.C @ self.P
        #self.P = (np.eye(self.n) - K @ self.C) @ self.P @ (np.eye(self.n) - K @ self.C).T + K @ self.R @ K.T

    def update_with_C(self, C, U, Y):
        self.C = C
        self.update(U, Y)

class KalmanComposite(CompositeObserver):
    def __init__(self, name, g, Mc):
        super().__init__(name)
        self.g = g
        self.Mc = Mc
        self.kalmanX = None
        self.kalmanY = None
        self.kalmanZ = None

    def addKalmanX(self, obs):
        self.kalmanX = obs
        self.add(obs)

    def addKalmanY(self, obs):
        self.kalmanY = obs
        self.add(obs)

    def addKalmanZ(self, obs):
        self.kalmanZ = obs
        self.add(obs)

    def update(self, U, Y):
        self.kalmanZ.update(U, Y)


        zcHat = self.kalmanZ.state()["kalman_z"][0]   # estimate of z CoM
        ddzcHat = self.kalmanZ.state()["kalman_z"][2] # estimate of acceleration z CoM
        fzHat = self.kalmanZ.state()["kalman_z"][3]   # estimate of external force z axis
        grfHat = self.Mc * ddzcHat + self.Mc * self.g - fzHat

        Cx = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, -self.Mc * zcHat / grfHat, zcHat / grfHat, 0]
        ])


        self.kalmanX.update_with_C(Cx, U, Y)
        self.kalmanY.update_with_C(Cx, U, Y)

class KalmanComposite_N(CompositeObserver):
    def __init__(self, name, g, Mc):
        super().__init__(name)
        self.g = g
        self.Mc = Mc
        self.kalmanX = None
        self.kalmanY = None
        self.kalmanZ = None

    def addKalmanX(self, obs):
        self.kalmanX = obs
        self.add(obs)

    def addKalmanY(self, obs):
        self.kalmanY = obs
        self.add(obs)

    def addKalmanZ(self, obs):
        self.kalmanZ = obs
        self.add(obs)

    def update(self, U, Y):
        self.kalmanZ.update(U, Y)

        zcHat = self.kalmanZ.state()["kalmgan_z"][0]   # estimate of z CoM
        ddzcHat = self.kalmanZ.state()["kalmgan_z"][2] # estimate of acceleration z CoM
        fzHat = self.kalmanZ.state()["kalmgan_z"][3]   # estimate of external force z axis
        grfHat = self.Mc * ddzcHat + self.Mc * self.g - fzHat

        Cx = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, -self.Mc*zcHat/grfHat, zcHat/grfHat, 1/grfHat, -1/grfHat],
            [0, 0, 0, 0, 0, 1]
        ])

        Cy = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, -self.Mc*zcHat/grfHat, zcHat/grfHat, -1/grfHat, 1/grfHat],
            [0, 0, 0, 0, 0, 1]
        ])

        self.kalmanX.update_with_C(Cx, U, Y)
        self.kalmanY.update_with_C(Cy, U, Y)

class StephensFilter(Observer):
    def __init__(self, dt, A, B, C, Q, R, name, axis):
        super().__init__(name, axis, dt, A.shape[0], C.shape[0])
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        #self.P = np.eye(A.shape[0]) * 100  # Initial uncertainty

    def update(self, U, Y):
        u = U[self.axis, :]
        if self.axis == 2:
            y = np.array([Y[self.axis, 0], 0.0])
        else:
            y = np.array([Y[self.axis, 0], Y[self.axis, 1]])

        # Prediction step
        self.xAct = self.A @ self.xAct + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Correction step
        G = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        self.xAct = self.xAct + G @ (y - self.C @ self.xAct)
        self.P = self.P - G @ self.C @ self.P