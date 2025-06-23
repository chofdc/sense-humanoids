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
from force_scripts.observers_three import KalmanFilter, KalmanFilterN, KalmanComposite, StephensFilter, KalmanComposite_N
from config_qibullet import *

# Constants
comTargetHeight = 0.33
ni = np.sqrt(gravity / comTargetHeight)

# Kalman Observer initialization
def initialize_kalman(comInitialPosition, comInitialAcceleration):
    A = np.array([
        [1, dT, dT**2/2, 0, 0],
        [0, 1, dT, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, dT],
        [0, 0, 0, 0, 1]
    ])
    B = np.array([
        [dT**3/6, 0],
        [dT**2/2, 0],
        [dT, 0],
        [0, dT**2/2],
        [0, dT]
    ])
    Cz = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -Mc, 1, 0]
    ])
    sigma_jerk = 1e3
    sigma_ddfext = 1e3
    Rz = np.array([
        [0.01, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [0.01, 0, 0],
        [0, 1, 0],
        [0, 0, 0.01]
    ])
    Ry = Rx
    initX = np.array([comInitialPosition[0], 0., comInitialAcceleration[0], 0., 0.])
    initY = np.array([comInitialPosition[1], 0., comInitialAcceleration[1], 0., 0.])
    initZ = np.array([comInitialPosition[2], 0., comInitialAcceleration[2], 0., 0.])
    f = 1e2
    P0 = f * np.eye(5)  # Identity matrix scaled by f

    kalman_z = KalmanFilter(dT, A, B, Cz, Rz, [sigma_jerk, sigma_ddfext], "kalman_z", 2)
    #kalman_z.init()
    kalman_z.init(initZ)

    kalman_x = KalmanFilter(dT, A, B, Cz, Rx, [sigma_jerk, sigma_ddfext], "kalman_x", 0)
    #kalman_x.init()
    kalman_x.init(initX)

    kalman_y = KalmanFilter(dT, A, B, Cz, Ry, [sigma_jerk, sigma_ddfext], "kalman_y", 1)
    #kalman_y.init()
    kalman_y.init(initY)
    
    composite = KalmanComposite("kalman_composite", gravity, Mc)
    composite.addKalmanZ(kalman_z)
    composite.addKalmanX(kalman_x)
    composite.addKalmanY(kalman_y)
    
    return composite

# Kalman Generalized Observer initialization
def initialize_kalman_N(comInitialPosition, comInitialAcceleration, MomentumInitialPosition):
    A = np.array([
        [1, dT, dT**2/2, 0],
        [0, 1, dT, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    B = np.array([
        [dT**3/6, 0],
        [dT**2/2, 0],
        [dT, 0],
        [0, dT]
    ])
    Ax = np.array([
        [1, dT, dT**2/2, 0, 0, 0],
        [0, 1, dT, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    Bx = np.array([
        [dT**3/6, 0, 0, 0],
        [dT**2/2, 0, 0, 0],
        [dT, 0, 0, 0],
        [0, dT, 0, 0],
        [0, 0, dT, 0],
        [0, 0, 0, dT]
    ])
    Cz = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, -Mc, 1]
    ])
    sigma_jerk = 1e5
    sigma_ddfext = 1e5
    sigma_ddL = 1e2
    sigma_drext = 1e3
    sigma_dMext = 1e2

    Rz = np.array([
        [0.01, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [0.1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Ry = Rx
    initX = np.array([comInitialPosition[0], 0., comInitialAcceleration[0], 0., 0., MomentumInitialPosition[1]])
    initY = np.array([comInitialPosition[1], 0., comInitialAcceleration[1], 0., 0., MomentumInitialPosition[0]])
    initZ = np.array([comInitialPosition[2], 0., comInitialAcceleration[2], 0.])
    f = 1e2
    P0 = f * np.eye(6)  # Identity matrix scaled by f

    kalman_z = KalmanFilterN(dT, A, B, Cz, Rz, [sigma_jerk, sigma_ddfext], "kalmgan_z", 2)
    kalman_z.init(initZ)
    kalman_x = KalmanFilterN(dT, Ax, Bx, Cz, Rx, [sigma_jerk, sigma_ddfext, sigma_dMext, sigma_ddL], "kalmgan_x", 0)
    kalman_x.init(initX)
    kalman_y = KalmanFilterN(dT, Ax, Bx, Cz, Ry, [sigma_jerk, sigma_ddfext, sigma_dMext, sigma_ddL], "kalmgan_y", 1)
    kalman_y.init(initY)

    composite = KalmanComposite_N("kalmgan_composite", gravity, Mc)
    composite.addKalmanZ(kalman_z)
    composite.addKalmanX(kalman_x)
    composite.addKalmanY(kalman_y)
    
    return composite

# Stephens Observer initialization
def initialize_stephens(comInitialPosition, zmpInitialPosition):
    A = np.array([
        [1, dT, 0, 0],
        [ni**2 * dT, 1, -ni**2 * dT, dT],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    B = np.array([[0], [0], [dT], [0]])
    C = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    posProNoise = np.exp(-8); velProNoise = np.exp(-4); forceProNoise = np.exp(-1); posOutputNoise = np.exp(-8); zmpOutputNoise = np.exp(-4)

    Q = np.diag([posProNoise, velProNoise, velProNoise, forceProNoise])
    R = np.diag([posOutputNoise, zmpOutputNoise])
    f = 1e2
    P0 = f * np.eye(4)  # Identity matrix scaled by f
    initX = np.array([comInitialPosition[0], 0., zmpInitialPosition[0], 0.])
    initY = np.array([comInitialPosition[1], 0., zmpInitialPosition[1], 0.])
    initZ = np.array([comInitialPosition[2], 0., 0., 0.])

    observers = []
    observers.append(StephensFilter(dT, A, B, C, Q, R, "stephens_x", 0))
    observers[-1].init(initX, P0)
    observers.append(StephensFilter(dT, A, B, C, Q, R, "stephens_y", 1))
    observers[-1].init(initY, P0)
    observers.append(StephensFilter(dT, A, B, C, Q, R, "stephens_z", 2))
    observers[-1].init(initZ, P0)
    
    return observers