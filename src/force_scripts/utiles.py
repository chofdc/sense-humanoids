import numpy as np
from qibullet import NaoFsr
import pybullet as p
import pandas as pd

# Positions of FSRs on the left foot (relative to the left foot frame)
fsr_positions_left = np.array([
    [0.07025, 0.0299, 0.0],
    [0.07025, -0.0231, 0.0],
    [-0.03025, 0.0299, 0.0],
    [-0.02965, -0.0191, 0.0]
])

# Positions of FSRs on the right foot (relative to the right foot frame)
fsr_positions_right = np.array([
    [0.07025, 0.0231, 0.0],
    [0.07025, -0.0299, 0.0],
    [-0.03025, 0.0191, 0.0],
    [-0.02965, -0.0299, 0.0]
])

# Function to calculate ZMP
def zmp_classique(nao, forces_left, forces_right):
    vLFSRfl, vLFSRfr, vLFSRrl, vLFSRrr = forces_left
    vRFSRfl, vRFSRfr, vRFSRrl, vRFSRrr = forces_right

    LFSRfl = nao.getLinkPosition('LFsrFL_frame')[0]
    LFSRfr = nao.getLinkPosition('LFsrFR_frame')[0]
    LFSRrl = nao.getLinkPosition('LFsrRL_frame')[0]
    LFSRrr = nao.getLinkPosition('LFsrRR_frame')[0]

    RFSRfl = nao.getLinkPosition('RFsrFL_frame')[0]
    RFSRfr = nao.getLinkPosition('RFsrFR_frame')[0]
    RFSRrl = nao.getLinkPosition('RFsrRL_frame')[0]
    RFSRrr = nao.getLinkPosition('RFsrRR_frame')[0]

    LFsr = vLFSRfl + vLFSRfr + vLFSRrl + vLFSRrr
    RFsr = vRFSRfl + vRFSRfr + vRFSRrl + vRFSRrr
        
    total_force = LFsr + RFsr

    if total_force == 0:
        # If no force is measured, return NaN or 0 to indicate that the ZMP is undefined
        #return [float('nan'), float('nan'), 0]
        return [float(0), float(0), 0]
    
    zmp_x = (LFSRfl[0]*vLFSRfl + LFSRfr[0]*vLFSRfr + LFSRrl[0]*vLFSRrl + LFSRrr[0]*vLFSRrr +
             RFSRfl[0]*vRFSRfl + RFSRfr[0]*vRFSRfr + RFSRrl[0]*vRFSRrl + RFSRrr[0]*vRFSRrr) / total_force
    zmp_y = (LFSRfl[1]*vLFSRfl + LFSRfr[1]*vLFSRfr + LFSRrl[1]*vLFSRrl + LFSRrr[1]*vLFSRrr +
             RFSRfl[1]*vRFSRfl + RFSRfr[1]*vRFSRfr + RFSRrl[1]*vRFSRrl + RFSRrr[1]*vRFSRrr) / total_force

    return [zmp_x, zmp_y, 0]

# Function to calculate COM
def compute_center_of_mass(robot, segments_data):
    """ Computes the robot's center of mass using segment CoM positions. """
    total_mass = sum(data["mass"] for data in segments_data.values())
    CoM_robot = np.zeros(3)
    
    for segment, data in segments_data.items():
        mass = data["mass"]
        CoM_local = data["CoM_offset"]
        
        link_state = robot.getLinkPosition(segment)
        link_pos = np.array(link_state[0])
        link_orn = np.array(link_state[1])

        R_world_link = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
        CoM_world = link_pos + R_world_link @ CoM_local
        CoM_robot += mass * CoM_world
    
    return CoM_robot / total_mass

def skew_symmetric(vector):
    """ Compute the skew-symmetric matrix of a 3D vector """
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

def compute_whole_body_inertia(robot, segments_data, CoM_nao):
    """ Computes the whole-body inertia matrix in the world frame. """
    I_total = np.zeros((3, 3))
    
    for segment, data in segments_data.items():
        mass = data["mass"]
        I_g = data["I_g"]
        
        link_state = robot.getLinkPosition(segment)
        link_pos = np.array(link_state[0])
        link_orn = np.array(link_state[1])
        
        R_world_link = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
        
        CoM_local = data["CoM_offset"]
        CoM_world = link_pos + R_world_link @ CoM_local
        r_i = CoM_world - CoM_nao
        
        r_skew = skew_symmetric(r_i)
        I_shifted = mass * (r_skew @ r_skew.T)
        
        I_world = R_world_link @ I_g @ R_world_link.T
        
        I_total += I_world + I_shifted
    
    return I_total