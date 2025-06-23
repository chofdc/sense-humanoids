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

from pathlib import Path
import numpy as np

force_type = "hybrid"  # periodic, constant, hybrid
A_x = 4.8;   A_y = 3.2; A_z = 5;  frequency_x = 0.05;    frequency_y = 0.1;  frequency_z = 0.15;  phi_x = 0;  phi_y = np.pi/2;  phi_z = np.pi/4
#A_x = 3;   A_y = -5;   A_z = 4 

data_folder = Path("./DATA")

data_subname = "31mars" # "theday"

link_name = "torso" # "r_wrist" "r_gripper" "LForeArm" "RElbow"

# Initial setup
gravity = 9.80665
Mc = 5.305350006
num_samples = 1800 #1800 (14*innerloop ~ 1800)
dT = 0.0167 #1/60