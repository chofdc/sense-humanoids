from pathlib import Path
import numpy as np

force_type = "constant"  # periodic, constant, manual
#A_x = 4.8;   A_y = 3.2; A_z = 5;  frequency_x = 0.05;    frequency_y = 0.1;  frequency_z = 0.15;  phi_x = 0;  phi_y = np.pi/2;  phi_z = np.pi/4
A_x = 4;   A_y = -5;   A_z = 3 

data_folder = Path("./DATA")

data_subname = "28mars" # "theday"

link_name = "torso" # "r_wrist" "r_gripper" "LForeArm" "RElbow"

# Initial setup
gravity = 9.80665
Mc = 5.305350006
num_samples = 1900 #1800
dT = 0.0167

#---- name="RLeg_effector_fixedjoint" "LLeg_effector_fixedjoint"