import numpy as np
import pybullet as p
import pybullet_data
import pandas as pd
import time
import json
from qibullet import SimulationManager, NaoVirtual, NaoFsr
import force_scripts.utiles as utiles  

from force_scripts.initializations import initialize_kalman, initialize_stephens, initialize_kalman_N
from force_scripts.observers_three import CompositeObserver

from config_qibullet import *

def transform_com_to_torso(com_world, robot_id, link_index):
    """
    Transforms the Center of Mass (CoM) position from world frame to the torso frame.
    
    Parameters:
        com_world (list or np.array): CoM position in world coordinates [x, y, z].
        robot_id (int): PyBullet ID of the robot.
    
    Returns:
        np.array: CoM position in the torso frame.
    """
    # Get the torso state (assuming torso is link index 0, check your URDF)
    torso_state = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    torso_pos, torso_ori = torso_state[0], torso_state[1]  # Extract position and orientation

    # Compute inverse transform (world -> torso frame)
    torso_inv_pos, torso_inv_ori = p.invertTransform(torso_pos, torso_ori)

    # Transform CoM position into torso frame
    com_torso, _ = p.multiplyTransforms(torso_inv_pos, torso_inv_ori, com_world, [0, 0, 0, 1])

    return np.array(com_torso)  # Return as a NumPy array

def getMotorJointStates(robot):
    num_joints = p.getNumJoints(robot)
    joint_data = [(p.getJointInfo(robot, i), p.getJointState(robot, i)) for i in range(num_joints)]
    motor_joints = [(info, state) for info, state in joint_data if info[3] > -1]

    joint_indices = [info[0] for info, _ in motor_joints]
    joint_names = [info[1].decode("utf-8") for info, _ in motor_joints]
    joint_positions = [state[0] for _, state in motor_joints]
    joint_velocities = [state[1] for _, state in motor_joints]

    return joint_positions, joint_velocities, joint_indices, joint_names

def plotForceVector(line_id, text_id, robotId, link_id, force_to_apply):
    # Get the position of the link
    link_state = p.getLinkState(robotId, link_id)
    position = link_state[0]
    
    # Calculate the end position of the force vector for visualization
    end_position = [position[0] + force_to_apply[0] * 0.1, position[1] + force_to_apply[1] * 0.1, position[2] + force_to_apply[2] * 0.1]
    
    # Update the force vector
    p.addUserDebugLine(position, end_position, [1, 0.5, 0], 2, replaceItemUniqueId=line_id)
    
    # Update the force value text
    force_magnitude = np.linalg.norm(force_to_apply)
    text_position = [(position[0] + end_position[0]) / 2, (position[1] + end_position[1]) / 2, (position[2] + end_position[2]) / 2 + 0.05]
    p.addUserDebugText(f"{force_magnitude:.2f} N", text_position, [1, 0.5, 0], textSize=1.5, replaceItemUniqueId=text_id)

def load_segment_data(csv_file):
    """ Load predefined segment parameters from CSV file. """
    df = pd.read_csv(csv_file, index_col=0, converters={
        "CoM in Link Frame": eval,
        "Inertia at CoM": eval,
        "Inertia at Origin": eval
    })
    
    segments_data = {}
    for segment in df.index:
        segments_data[segment] = {
            "mass": df.loc[segment, "Mass"],
            "CoM_offset": np.array(df.loc[segment, "CoM in Link Frame"]),
            "I_g": np.array(df.loc[segment, "Inertia at CoM"]),
            "I_o": np.array(df.loc[segment, "Inertia at Origin"])
        }
    return segments_data

def compute_angular_momentum_dot(robot, I_nao):
    """ Computes the derivative of angular momentum \dot{L} using the centroidal model. """
    global omega_prev, dT
    
    omega = np.array(robot.getImuGyroscopeValues())
    dot_omega = (omega - omega_prev) / dT
    omega_prev = omega  
    #dot_L = np.dot(I_nao, dot_omega) + np.cross(omega, np.dot(I_nao, omega))
    dot_L = I_nao @ dot_omega + np.cross(omega, I_nao @ omega)

    return dot_L

def apply_force(robotId, link_id, position, iteration):
    if force_type == "constant":
        force = [A_x, A_y, A_z]
    elif force_type == "periodic":
        F_x = A_x * np.sin(2 * np.pi * frequency_x * dT * iteration + phi_x)
        F_y = A_y * np.sin(2 * np.pi * frequency_y * dT * iteration + phi_y)
        F_z = A_z * np.sin(2 * np.pi * frequency_z * dT * iteration + phi_z)
        force = [F_x, F_y, F_z]
    
    p.applyExternalForce(objectUniqueId=robotId, linkIndex=link_id, forceObj=force, posObj=position, flags=p.WORLD_FRAME)
    return force

def main():
    global omega_prev, dT  # Declare omega_prev as global

    csv_file = "predefined_segments.csv"
    segments_data = load_segment_data(csv_file)
    
    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=True)
    #p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #planeId = p.loadURDF('plane.urdf', [0, 0, 0])
        
    nao_robot = simulation_manager.spawnNao(client, spawn_ground_plane=True)
    robotId = nao_robot.getRobotModel()
    
    # Set the robot to StandInit posture
    nao_robot.goToPosture("StandInit", 1.0)
    robotPos, _ = p.getBasePositionAndOrientation(robotId)
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=135, cameraPitch=-10,
                                    cameraTargetPosition=robotPos)
    # Get Nao's body id and apply force
    link_id = nao_robot.getLink(link_name).getIndex()
    # Initialize the robot posture
    time.sleep(3)  
    nao_robot.subscribeImu(frequency=1/dT)

    p.setRealTimeSimulation(0)  # Disable real-time simulation

    # Create debug items (line and text)
    line_id = p.addUserDebugLine([0, 0, 0], [0, 0, 0], [1, 0, 0], 2, lifeTime=0)
    text_id = p.addUserDebugText("", [0, 0, 0], [1, 0, 0], textSize=1.5, lifeTime=0)

    # Precompute the range check
    force_application_start = int(5 / dT)
    force_application_end = int(25 / dT)    

    # Initialize `omega_prev` before using it in `compute_angular_momentum_dot`
    #omega_prev = np.array(nao_robot.getImuGyroscopeValues())  # Set initial angular velocity

    # Initial CoM position
    comInitialPosition = utiles.compute_center_of_mass(nao_robot, segments_data)
    omega_prev, comInitialAcceleration = nao_robot.getImuValues()
    forces_left = nao_robot.getFsrValues(NaoFsr.LFOOT)
    forces_right = nao_robot.getFsrValues(NaoFsr.RFOOT)
    zmpInitialPosition = utiles.zmp_classique(nao_robot, forces_left, forces_right)
    I_nao = utiles.compute_whole_body_inertia(nao_robot, segments_data, comInitialPosition)
    MomentumInitialPosition = compute_angular_momentum_dot(nao_robot, I_nao)  # Now omega_prev is properly initialized

    # Initialize Composite Observer
    composite_observer = CompositeObserver("composite")

    # Add Kalman Observer Composite
    composite_observer.add(initialize_kalman(comInitialPosition, comInitialAcceleration))

    # Add Kalman Generalized Observer Composite
    composite_observer.add(initialize_kalman_N(comInitialPosition, comInitialAcceleration, MomentumInitialPosition))

    # Add Stephens Observers
    for observer in initialize_stephens(comInitialPosition, zmpInitialPosition):
        composite_observer.add(observer)
        
    # Initialize lists to store estimates
    kalman_estimates = []
    kalmgan_estimates = []
    stephens_estimates = []
    com_pos = comInitialPosition
    data = []
    # Timing for loop
    simulation_time = time.perf_counter()
    start_time = simulation_time
    
    for iter in range(num_samples):
        #local_com_torso = transform_com_to_torso(com_pos, robotId, 0)  # Convert to torso frame "link_id"

        #position = list(nao_robot.getLinkPosition(link_name)[0])
        position = list(com_pos)
        if force_application_start < iter < force_application_end:
            force_to_apply = apply_force(robotId, link_id, position, iter)
        else:
            force_to_apply = [0, 0, 0]
        
        plotForceVector(line_id, text_id, robotId, link_id, force_to_apply)

        #angular_velocity, com_acceleration = nao_robot.getImuValues()  #No need, bcz omega_prev is updated in the L.dot function (global)

        # Compute other necessary measurements
        forces_left = nao_robot.getFsrValues(NaoFsr.LFOOT)
        forces_right = nao_robot.getFsrValues(NaoFsr.RFOOT)
        f_o = -(np.sum(forces_left) + np.sum(forces_right)) * gravity

        com_pos = utiles.compute_center_of_mass(nao_robot, segments_data)
        zmp_pos = utiles.zmp_classique(nao_robot, forces_left, forces_right)
        com_pos = utiles.compute_center_of_mass(nao_robot, segments_data)
        com_accel = np.array(nao_robot.getImuAccelerometerValues())
        I_nao = utiles.compute_whole_body_inertia(nao_robot, segments_data, com_pos)
        Ldot_pos = compute_angular_momentum_dot(nao_robot, I_nao)  # Now omega_prev is globally updated

        # Retrieve FSR values
        forces_left = nao_robot.getFsrValues(NaoFsr.LFOOT)
        forces_right = nao_robot.getFsrValues(NaoFsr.RFOOT)
        f_o = -(np.sum(forces_left) + np.sum(forces_right)) * gravity

        observer_start_time = time.perf_counter()
        # Create U and Y matrices
        U = np.zeros((4, 1))
        Y = np.zeros((3, 4))

        # Fill U and Y with appropriate values from data
        #U[:, 0] = force_to_apply
        Y[0, :] = [com_pos[0], zmp_pos[0], com_accel[0], Ldot_pos[1]]#, position[0], true_force[0]]
        Y[1, :] = [com_pos[1], zmp_pos[1], com_accel[1], Ldot_pos[0]]#, position[1], true_force[1]]
        #Y[2, :] = [com_pos[2], -(grf_left[2] + grf_right[2]), com_accel[2]]
        Y[2, :] = [com_pos[2], -f_o + Mc*gravity, com_accel[2], Ldot_pos[2]]#, position[2], true_force[2]]
        
        # Update composite observer
        composite_observer.update(U, Y)

        observer_end_time = time.perf_counter()
        observer_dt = observer_end_time - observer_start_time
        #print(f"    Observer update time: {observer_dt:.6f} s")
    
        data.append({
            "com_pos": com_pos.tolist(),
            "angular_momentum_dot": Ldot_pos.tolist(),
            "applied_force": force_to_apply
        })

        # Save observer states
        for obs in composite_observer.children:
            if "kalman" in obs.name:
                kalman_estimates.append({
                    "state": {k: v.tolist() for k, v in obs.state().items()},
                    "uncertainty": {k: v.tolist() for k, v in obs.uncertainty().items()}
                })
            elif "kalmgan" in obs.name:
                kalmgan_estimates.append({
                    "state": {k: v.tolist() for k, v in obs.state().items()},
                    "uncertainty": {k: v.tolist() for k, v in obs.uncertainty().items()}
                })            
            elif "stephens" in obs.name:
                stephens_estimates.append({
                    "state": {k: v.tolist() for k, v in obs.state().items()},
                    "uncertainty": {k: v.tolist() for k, v in obs.uncertainty().items()}
                })

        # **Sleep uniquement si `dT > loop_time`**
        elapsed_time = time.perf_counter() - start_time
        time_to_sleep = max(0, dT - elapsed_time)  # Avoid negative sleep times
        time.sleep(time_to_sleep)

        #print(f"Temps d'échantillonnage : {time.perf_counter() - start_time:.6f} s")
        start_time = time.perf_counter()
        #time.sleep(dT)
        p.stepSimulation()

    print(f"Temps de simulation : {time.perf_counter() - simulation_time:.6f} s")
    # Add these lines to close the simulation window
    simulation_manager.stopSimulation(client)

    # Save data to file
    with open(data_folder / f"online_data_{data_subname}_{force_type}_{link_name}.json", "w") as f:
        json.dump(data, f)

    # Save estimates to file
    with open(data_folder / f"online_estimates_{data_subname}_{force_type}_{link_name}.json", "w") as f:
        json.dump({
            "kalman": kalman_estimates,
            "kalman_g": kalmgan_estimates,
            "stephens": stephens_estimates
        }, f)

if __name__ == "__main__":
    main()
