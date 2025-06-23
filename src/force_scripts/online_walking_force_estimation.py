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

import os
import time, multiprocessing
from multiprocessing.queues import Empty 
import threading
import queue

import numpy as np, pandas as pd, pybullet as p
from qibullet import SimulationManager, NaoFsr

from config_walking_qibullet import *
import force_scripts.utiles as utiles
from force_scripts.initializations import initialize_kalman, initialize_kalman_N, initialize_stephens
from force_scripts.observers_three import CompositeObserver

import json

cwd = os.getcwd()
motions_folder = Path(cwd)/"motions"     #/motions" 

def read_csv(filename):
    df = pd.read_csv(filename)
    num_poses = df.count()[0]
    cols = df.columns
    num_joints = len(cols) - 2
    joint_names = cols[2:]
    return num_poses, num_joints, joint_names, df

def animate(force_queue, num_poses, num_joints, joint_names, df, robot, delay=0.05):
    current_force = None  # Initialization
    for pose in range(num_poses):
        start_time = time.perf_counter()
        robotId = robot.getRobotModel()
        
        joint_names_list = list(joint_names)  # list of all joint names
        joint_values_list = [float(df[name][pose]) for name in joint_names_list]

        robot.setAngles(joint_names_list, joint_values_list, 1.0)
        
        for j in range(num_joints):
            '''
            name = joint_names[j]
            angle = float(df[name][pose])
            robot.setAngles(name, angle, 1.0)
            '''
            p.stepSimulation()

            # Retrieves the last available force in the queue (non-blocking)
            try:
                current_force = force_queue.get_nowait()
            except queue.Empty:
                pass  # Keep the old force if nothing new

            # Reapply force if available
            if current_force is not None:
                p.applyExternalForce(
                    current_force["robotId"],
                    current_force["link_id"],
                    current_force["force"],
                    current_force["position"],
                    p.WORLD_FRAME
                )

        robotPos, _ = p.getBasePositionAndOrientation(robotId)
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=135, cameraPitch=-10,
                                     cameraTargetPosition=robotPos)
            
        elapsed_time = time.perf_counter() - start_time
        time_to_sleep = max(0, delay - elapsed_time)  # Avoid negative sleep times
        time.sleep(time_to_sleep)

def walking_thread_function(robot, force_queue, event):
    #Walking forward
    filename = motions_folder/'Forwards50.csv'
    num_poses1, num_joints1, joint_names1, df1 = read_csv(filename)
    #Side Step Left
    filename = motions_folder/'SideStepLeft.csv'
    num_poses2, num_joints2, joint_names2, df2 = read_csv(filename)
    #Side Step Right
    filename = motions_folder/'SideStepRight.csv'
    num_poses3, num_joints3, joint_names3, df3 = read_csv(filename)
    while not event.is_set():
        animate(force_queue, num_poses1, num_joints1, joint_names1, df1, robot, delay=0.04)
        #time.sleep(3.0)
        animate(force_queue, num_poses2, num_joints2, joint_names2, df2, robot, delay=0.04)
        #time.sleep(3.0)
        animate(force_queue, num_poses3, num_joints3, joint_names3, df3, robot, delay=0.04)
        #time.sleep(3.0)

def simulation_loop(data_queue, save_queue, force_queue, event):
    global omega_prev, dT

    simulation_manager = SimulationManager()
    client_id = simulation_manager.launchSimulation(gui=True)
    nao_robot = simulation_manager.spawnNao(client_id, spawn_ground_plane=True)
    robotId = nao_robot.getRobotModel()
    
    nao_robot.goToPosture("StandInit", 1.0)
    robotPos, _ = p.getBasePositionAndOrientation(robotId)
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=135, cameraPitch=-10,
                                    cameraTargetPosition=robotPos)
    time.sleep(2)

    p.setRealTimeSimulation(0)
    #nao_robot.subscribeImu(frequency=1/dT)
    link_id = nao_robot.getLink(link_name).getIndex()

    # Create debug items (line and text)
    line_id = p.addUserDebugLine([0, 0, 0], [0, 0, 0], [1, 0.5, 0], 2, lifeTime=0)
    text_id = p.addUserDebugText("", [0, 0, 0], [1, 0.5, 0], textSize=1.5, lifeTime=0)

    segments_data = load_segment_data("predefined_segments.csv")
    
    comInitialPosition = utiles.compute_center_of_mass(nao_robot, segments_data)
    omega_prev, comInitialAcceleration = nao_robot.getImuValues()
    forces_left = nao_robot.getFsrValues(NaoFsr.LFOOT)
    forces_right = nao_robot.getFsrValues(NaoFsr.RFOOT)
    zmpInitialPosition = utiles.zmp_classique(nao_robot, forces_left, forces_right)
    I_nao = utiles.compute_whole_body_inertia(nao_robot, segments_data, comInitialPosition)
    MomentumInitialPosition = compute_angular_momentum_dot(nao_robot, I_nao)

    init_data = {
        "comInitialPosition": comInitialPosition,
        "comInitialAcceleration": comInitialAcceleration,
        "zmpInitialPosition": zmpInitialPosition,
        "MomentumInitialPosition": MomentumInitialPosition
    }
    data_queue.put(init_data)
    
    # Launch of the walking thread
    '''
    sim_thread = threading.Thread(target=step_thread, args=(event,force_queue), daemon=True)
    sim_thread.start()
    '''
    walk_thread = threading.Thread(target=walking_thread_function, args=(nao_robot, force_queue, event), daemon=True)
    #walk_thread = threading.Thread(target=walking_thread_function, args=(nao_robot, event), daemon=True)
    walk_thread.start()
    
    com_pos = comInitialPosition
    simulation_time = time.perf_counter()
    start_time = simulation_time

    for iter in range(num_samples):
        position = list(com_pos)
        if int(5 / dT) < iter < int(25 / dT):
            force_to_apply = apply_force(robotId, link_id, position, iter)
        else:
            force_to_apply = [0, 0, 0]

        # Update force in queue
        force_queue.put({"robotId": robotId, "link_id": link_id, "force": force_to_apply, "position": position})

        plotForceVector(line_id, text_id, robotId, link_id, force_to_apply)

        forces_left = nao_robot.getFsrValues(NaoFsr.LFOOT)
        forces_right = nao_robot.getFsrValues(NaoFsr.RFOOT)
        f_o = -(np.sum(forces_left) + np.sum(forces_right)) * gravity

        com_pos = utiles.compute_center_of_mass(nao_robot, segments_data)
        zmp_pos = utiles.zmp_classique(nao_robot, forces_left, forces_right)
        com_accel = np.array(nao_robot.getImuAccelerometerValues())
        I_nao = utiles.compute_whole_body_inertia(nao_robot, segments_data, com_pos)
        Ldot_pos = compute_angular_momentum_dot(nao_robot, I_nao)

        data_to_send = {
            "com_pos": com_pos,
            "zmp_pos": zmp_pos,
            "com_accel": com_accel,
            "f_o": f_o,
            "LFSR":forces_left,
            "RFSR":forces_right,
            "force_to_apply": force_to_apply,
            "Ldot_pos": Ldot_pos
        }

        if data_queue.qsize() > 5:
            data_queue.get_nowait()
        data_queue.put(data_to_send)
        save_queue.put({"type": "walking", "data": data_to_send})

        elapsed_time = time.perf_counter() - start_time
        time.sleep(max(0, dT - elapsed_time))        

        #print(f"Sampling time :  {time.perf_counter() - start_time:.6f} s")
        start_time = time.perf_counter()
        p.stepSimulation()

    print(f"[Simulation] Total time : {time.perf_counter() - simulation_time:.2f}s")
    event.set()
    walk_thread.join()
    '''
    sim_thread.join()
    '''

def observer_function(data_queue, save_queue, event):
    composite_observer = CompositeObserver("composite")
    # Wait for the first initialization data
    print("Observer: Waiting for initialization values...")

    init_data = data_queue.get()
    comInitialPosition = init_data["comInitialPosition"]
    comInitialAcceleration = init_data["comInitialAcceleration"]
    zmpInitialPosition = init_data["zmpInitialPosition"]
    MomentumInitialPosition = init_data["MomentumInitialPosition"]
    print("Observer: Initialization completed.")

    # Add observers
    composite_observer.add(initialize_kalman(comInitialPosition, comInitialAcceleration))
    composite_observer.add(initialize_kalman_N(comInitialPosition, comInitialAcceleration, MomentumInitialPosition))
    for observer in initialize_stephens(comInitialPosition, zmpInitialPosition):
        #print(f"[DEBUG] Stephens Observer Initialized: {observer.name}")
        composite_observer.add(observer)

    
    observer_step_count = 0

    while not event.is_set():
        try:
            data = data_queue.get()  # Blocks until data is available
            observer_step_count += 1  # Track number of processed steps

            com_pos = data["com_pos"]
            zmp_pos = data["zmp_pos"]
            com_accel = data["com_accel"]
            f_o = data["f_o"]
            Ldot_pos = data["Ldot_pos"]
        
            # Observer Processing Timing
            observer_start_time = time.perf_counter()
            # Observer Inputs
            U = np.zeros((4, 1))
            Y = np.zeros((3, 4))
            Y[0, :] = [com_pos[0], zmp_pos[0], com_accel[0], Ldot_pos[1]]
            Y[1, :] = [com_pos[1], zmp_pos[1], com_accel[1], Ldot_pos[0]]
            Y[2, :] = [com_pos[2], -f_o + Mc * gravity, com_accel[2], Ldot_pos[2]]

            # Update Observer
            composite_observer.update(U, Y)
            
            observer_end_time = time.perf_counter()
            observer_dt = observer_end_time - observer_start_time
            print(f"    Observer update time: {observer_dt:.6f} s")

            # Saving observers' estimates
            observer_data = {
                "kalman": [],
                "kalmgan": [], 
                "stephens": []  
            }
            
            for obs in composite_observer.children:
                obs_name = obs.name.lower()
                #if "stephens" in obs.name.lower():
                #    print(f"[DEBUG] Étape {observer_step_count} - State de {obs.name}: {json.dumps(convert_to_serializable(obs.state()), indent=2)}")
                obs_data = {
                    "state": {k: v.tolist() for k, v in obs.state().items()},
                    "uncertainty": {k: v.tolist() for k, v in obs.uncertainty().items()}
                }

                if "kalman" in obs_name:
                    observer_data["kalman"] = obs_data
                elif "kalmgan" in obs_name:
                    observer_data["kalmgan"] = obs_data
                elif "stephens" in obs_name:
                    observer_data["stephens"].append(obs_data)
                                
            save_queue.put({"type": "observer", "data": observer_data}) #see possible cause is the conversion to list
            
        except Empty:
            print(f"[WARNING] Observer missed a step at count {observer_step_count}")
    print("[DEBUG] Observer process exiting.")

def main():
    data_queue = multiprocessing.Queue()
    save_queue = multiprocessing.Queue()
    force_queue = multiprocessing.Queue()

    event = multiprocessing.Event()
    
    simulation_process = multiprocessing.Process(target=simulation_loop, args=(data_queue, save_queue, force_queue, event))
    #simulation_process = multiprocessing.Process(target=simulation_loop, args=(data_queue, save_queue, event))
    observer_process = multiprocessing.Process(target=observer_function, args=(data_queue, save_queue, event))
    save_process = multiprocessing.Process(target=save_data_function, args=(save_queue, event))

    simulation_process.start()
    observer_process.start()
    save_process.start()

    simulation_process.join()
    observer_process.join()  # Ensure observer completes
    event.set()  # Signal save process to finish
    save_process.join()  # Ensure saving is done

'''
ADD Functions
'''
def apply_force(robotId, link_id, position, iteration):
    if force_type == "constant":
        force = [A_x, A_y, A_z]
    elif force_type == "periodic":
        F_x = A_x * np.sin(2 * np.pi * frequency_x * dT * iteration + phi_x)
        F_y = A_y * np.sin(2 * np.pi * frequency_y * dT * iteration + phi_y)
        F_z = A_z * np.sin(2 * np.pi * frequency_z * dT * iteration + phi_z)
        force = [F_x, F_y, F_z]
    else:
        F_x = A_x * np.sin(2 * np.pi * frequency_x * dT * iteration + phi_x)
        F_y = A_y * np.sin(2 * np.pi * frequency_y * dT * iteration + phi_y)
        force = [F_x, F_y, A_z]
    p.applyExternalForce(objectUniqueId=robotId, linkIndex=link_id, forceObj=force, posObj=position, flags=p.WORLD_FRAME)
    return force

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
    global omega_prev
    omega = np.array(robot.getImuGyroscopeValues())
    #with imu_lock:
    #    omega = np.array(imu_data["gyro"]) if imu_data["gyro"] is not None else np.zeros(3)
    dot_omega = (omega - omega_prev) / dT
    omega_prev = omega
    # dot_L = np.dot(I_nao, dot_omega) + np.cross(omega, np.dot(I_nao, omega))
    dot_L = I_nao @ dot_omega + np.cross(omega, I_nao @ omega)
    return dot_L

data_folder.mkdir(parents=True, exist_ok=True)
data_file = data_folder / f"online_walking_data_{data_subname}_{force_type}_{link_name}.json"
estimates_file = data_folder / f"online_walking_estimates_{data_subname}_{force_type}_{link_name}.json"

def convert_to_serializable(obj):
    """Recursively convert NumPy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

def save_data_function(save_queue, event):
    """Accumulate data in memory and save it at the end."""
    
    walking_data = []  # Store all walking data
    observer_estimates = []  # Store all observer data

    while not event.is_set() or not save_queue.empty():
        try:
            #item = save_queue.get(timeout=0.5)
            item = save_queue.get_nowait()

            if item["type"] == "walking":
                walking_data.append(item["data"])
            elif item["type"] == "observer":
                observer_estimates.append(item["data"])
            else:
                print(f"[ERROR] Unexpected item type: {item}")

        except Empty:
            pass  # No data, continue waiting
    
    try:
        with open(data_file, "w") as f:
            json.dump(convert_to_serializable(walking_data), f, indent=4)

        with open(estimates_file, "w") as f:
            json.dump(observer_estimates, f, indent=4)
        
    finally:
        print("[DEBUG] Save process completed.")  # Ensures execution of cleanup
    
    print("[DEBUG] Exiting save_data_function.")  # 🔥 If we never see this, it hangs here


if __name__ == "__main__":
    main()