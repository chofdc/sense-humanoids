# Script that computes the total RMSE, the total MAE, the partial RMSE and the Time To Convergence (TTC) for all the observers given the log files.

import json
import numpy as np
import matplotlib.pyplot as plt
from config_qibullet import *

# --- Function that computes the RMSE value of the signal. ---
# @param est: np array of the estimates
# @param gt: np array of the truth values
# @return rmse: the corresponding value of the RMSE

def rmse(est, gt):
    e_sq = (np.linalg.norm(gt - est, axis=-1))**2
    return np.sqrt(e_sq.mean())


# --- Function that computes the MAE value of the signal. ---
# @param est: np array of the estimates
# @param gt: np array of the truth values
# @return rmse: the corresponding value of the MAE

def mae(est, gt):
    e_sq = np.linalg.norm(gt - est, axis=-1)
    return e_sq.mean()


def main():
    # Load estimates data
    with open(data_folder / f"estimates_{data_subname}_{force_type}.json", "r") as f:
        estimates = json.load(f)

    # Extract ground truth and observer estimates
    with open(data_folder / f"data_{data_subname}_{force_type}.json", "r") as f:
        data = json.load(f)

    ground_truth_force = np.array([sample["applied_force"] for sample in data])
    force_gt = np.stack([ground_truth_force[:,0], ground_truth_force[:,1]], axis=-1)
    
    # Extracting the state values with key checks
    kalman_x_states = [state["state"]["kalman_x"] if "kalman_x" in state["state"] else None for state in estimates["kalman"]]
    kalman_y_states = [state["state"]["kalman_y"] if "kalman_y" in state["state"] else None for state in estimates["kalman"]]
    kalman_z_states = [state["state"]["kalman_z"] if "kalman_z" in state["state"] else None for state in estimates["kalman"]]

    stephens_x_states = [state["state"]["stephens_x"] if "stephens_x" in state["state"] else None for state in estimates["stephens"]]
    stephens_y_states = [state["state"]["stephens_y"] if "stephens_y" in state["state"] else None for state in estimates["stephens"]]

    # Filter out None values from the lists
    kalman_x_states = [state for state in kalman_x_states if state is not None]
    kalman_y_states = [state for state in kalman_y_states if state is not None]
    kalman_z_states = [state for state in kalman_z_states if state is not None]

    force_x_kalm = [state[3] for state in kalman_x_states]
    force_y_kalm = [state[3] for state in kalman_y_states]

    force_kalm = np.stack([force_x_kalm, force_y_kalm], axis=-1)
    
    stephens_x_states = [state for state in stephens_x_states if state is not None]
    stephens_y_states = [state for state in stephens_y_states if state is not None]

    force_x_steph = [state[3]*Mc for state in stephens_x_states]
    force_y_steph = [state[3]*Mc for state in stephens_y_states]

    force_steph = np.stack([force_x_steph, force_y_steph], axis=-1)

    rmseK_tot = rmse(force_kalm, force_gt)
    rmseS_tot = rmse(force_steph, force_gt)

    maeK = mae(force_kalm, force_gt)
    maeS = mae(force_steph, force_gt)

    print('______________________________________________________________________')
    print()
    print('KALMAN data:')
    print('RMSE tot: {}'.format(rmseK_tot))
    print('MAE tot: {}'.format(maeK))
    print('----------------------------------------------------------------------')
    print('STEPHENS data:')
    print('RMSE tot: {}'.format(rmseS_tot))
    print('MAE tot: {}'.format(maeS))
    print()
    print('______________________________________________________________________')


if __name__ == "__main__":
    main()
