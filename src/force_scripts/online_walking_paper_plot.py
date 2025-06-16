'''
import matplotlib.font_manager as fm
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if "Times" in font:
        print(font)
        
'''
import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10, 
    'mathtext.fontset': 'stix', # uses a font similar to Times for math $$
    'axes.labelsize': 10,
    'axes.titlesize': 10,     
    'legend.fontsize': 9,      
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
}) # corresponds to the body of the IEEE text
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
#matplotlib.use('TkAgg')  # Or 'Qt5Agg', depending on what is available
import visualize_datapro as viz  

from scipy.signal import butter, filtfilt
from scipy.stats import zscore

from config_walking_qibullet import *


#matplotlibuse('TkAgg')  # Or 'Qt5Agg', depending on what is available

def load_json(filename):
    """ Load JSON file and handle errors """
    with open(filename, "r") as f:
        return json.load(f)
    
# Function to extract state values safely
def extract_states(estimates_list, observer_name, key_name):
    extracted_states = []
    
    for entry in estimates_list:
        if observer_name in entry:
            observer_data = entry[observer_name]  # Peut être une liste (stephens) ou un dict (kalman)

            if isinstance(observer_data, list):  # Cas où c'est une liste (stephens)
                for sub_entry in observer_data:
                    if isinstance(sub_entry, dict) and "state" in sub_entry and key_name in sub_entry["state"]:
                        extracted_states.append(sub_entry["state"][key_name])
            elif isinstance(observer_data, dict):  # Cas standard (kalman, stephens, etc.)
                if "state" in observer_data and key_name in observer_data["state"]:
                    extracted_states.append(observer_data["state"][key_name])

    return extracted_states

def main():
    # Load JSON Data
    data = load_json(data_folder / f"online_walking_data_{data_subname}_{force_type}_{link_name}.json")
    estimates_list = load_json(data_folder / f"online_walking_estimates_{data_subname}_{force_type}_{link_name}.json")

    timestamps = np.array(range(len(data))) * dT
    
    ground_truth_force = np.array([sample["force_to_apply"] for sample in data])
    angular_momentum = np.array([sample["Ldot_pos"] for sample in data])
    com_position = np.array([sample["com_pos"] for sample in data])
    zmp_pos = np.array([sample["zmp_pos"] for sample in data])
    com_accel = np.array([sample["com_accel"] for sample in data])
    ground_reaction_force = np.array([sample["f_o"] for sample in data])
    forces_left = - np.sum([sample["LFSR"] for sample in data], axis=1) * gravity
    forces_right = - np.sum([sample["RFSR"] for sample in data], axis=1) * gravity

    forces = np.stack((forces_left, forces_right), axis=1)

    # Extracting Kalman states
    kalman_x_states = extract_states(estimates_list, "kalman", "kalman_x")
    kalman_y_states = extract_states(estimates_list, "kalman", "kalman_y")
    kalman_z_states = extract_states(estimates_list, "kalman", "kalman_z")

    # Extracting Kalman-G states
    g_kalman_x_states = extract_states(estimates_list, "kalmgan", "kalmgan_x")
    g_kalman_y_states = extract_states(estimates_list, "kalmgan", "kalmgan_y")
    g_kalman_z_states = extract_states(estimates_list, "kalmgan", "kalmgan_z")

    # Suppose que z_values = [state[3] for state in g_kalman_z_states]
    x_values = np.array([state[3] for state in g_kalman_x_states])
    y_values = np.array([state[3] for state in g_kalman_y_states])
    z_values = np.array([state[3] for state in g_kalman_z_states])
    
    # 1. Supprimer les outliers (option: threshold=3 ou + strict selon ton besoin)
    cleaned_x = remove_outliers(x_values, threshold=3)
    cleaned_y = remove_outliers(y_values, threshold=3)
    cleaned_z = remove_outliers(z_values, threshold=3)
    # 2. Filtrage passe-bas (ex : 5 Hz de coupure pour un signal à 100 Hz)
    filtered_x = lowpass_filter(cleaned_x, cutoff=5.0, fs=100.0)
    filtered_y = lowpass_filter(cleaned_y, cutoff=5.0, fs=100.0)
    filtered_z = lowpass_filter(cleaned_z, cutoff=5.0, fs=100.0)
    
    # Extracting Stephens states
    stephens_x_states = extract_states(estimates_list, "stephens", "stephens_x")
    stephens_y_states = extract_states(estimates_list, "stephens", "stephens_y")
    stephens_z_states = extract_states(estimates_list, "stephens", "stephens_z")

    # Create plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.delaxes(axs[1, 1])  # This removes the empty subplot

    # Plot Kalman Observer for x-axis
    axs[0,0].plot(timestamps, ground_truth_force[:,0], label='Ground Truth x', color='black', linestyle='--', linewidth=3)
    axs[0,0].plot(timestamps, [state[3] for state in kalman_x_states], label='Hawley $\mathbf{F_x}$', color='blue', linewidth=1.5)
    axs[0,0].plot(timestamps, [state[3]*Mc for state in stephens_x_states], label='Stephens $\mathbf{F_x}$', color='red', linewidth=1.5)
    axs[0,0].plot(timestamps, [state[3] for state in g_kalman_x_states], label='SENSE $\mathbf{F_x}$', color='green', linewidth=2)
    axs[0,0].set_xlabel("Time (s)", fontsize=14)
    axs[0,0].set_ylabel("Force (N)", fontsize=14)
    axs[0,0].legend(prop={'size': 10, 'weight': 'bold'}) #, prop={'weight': 'bold'}
    axs[0,0].grid(True)

    # Plot Kalman Observer for y-axis
    axs[0,1].plot(timestamps, ground_truth_force[:,1], label='Ground Truth y', color='black', linestyle='--', linewidth=3)
    axs[0,1].plot(timestamps, [state[3] for state in kalman_y_states], label='Hawley $\mathbf{F_y}$', color='blue', linewidth=1.5)
    axs[0,1].plot(timestamps, [state[3]*Mc for state in stephens_y_states], label='Stephens $\mathbf{F_y}$', color='red', linewidth=1.5)
    axs[0,1].plot(timestamps, [state[3] for state in g_kalman_y_states], label='SENSE $\mathbf{F_y}$', color='green', linewidth=2)
    axs[0,1].set_xlabel("Time (s)", fontsize=14)
    axs[0,1].set_ylabel("Force (N)", fontsize=14)
    axs[0,1].legend(prop={'size': 10, 'weight': 'bold'})
    axs[0,1].grid(True)

    # Plot Kalman Observer for z-axis
    axs[1,0].plot(timestamps, ground_truth_force[:,2], label='Ground Truth z', color='black', linestyle='--', linewidth=3)
    axs[1,0].plot(timestamps, [state[3] for state in kalman_z_states], label='Hawley $\mathbf{F_z}$', color='blue', linewidth=1.5)
    #axs[1,0].plot(timestamps, [state[3]*Mc for state in stephens_z_states], label='Simplified Kalman $\mathbf{F_z}$', color='green', linewidth=3)
    axs[1,0].plot(timestamps, filtered_z, label='SENSE $\mathbf{F_z}$', color='green', linewidth=2)
    #axs[1,0].plot(timestamps, [state[3] for state in g_kalman_z_states], label='SENSE $\mathbf{F_z}$', color='green', linewidth=2)
    axs[1,0].set_xlabel("Time (s)", fontsize=14)
    axs[1,0].set_ylabel("Force (N)", fontsize=14)
    axs[1,0].legend(prop={'size': 10, 'weight': 'bold'})
    axs[1,0].grid(True)

    # Create inset (zoom) inside axs[1, 0]
    axins = inset_axes(axs[1, 0], width="20%", height="20%", loc='right') #'upper right' 'center'

    # Same plot inside inset
    axins.plot(timestamps, ground_truth_force[:, 2], color='black', linestyle='--', linewidth=3)
    axins.plot(timestamps, [state[3] for state in kalman_z_states], color='blue', linewidth=1.5)
    axins.plot(timestamps, filtered_z, color='green', linewidth=2)

    # Define zoomed region (adjust this based on your data)
    x1, x2 = 17, 20   # time range
    y1, y2 = 4.5, 5.3    # force range
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # Hide inset ticks
    axins.set_xticks([])
    axins.set_yticks([])

    # Draw a rectangle and lines to connect inset and original plot
    mark_inset(axs[1, 0], axins, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=1)

    # Adjust spacing of main plot
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    # Create main GridSpec
    gs = GridSpec(2, 2, figure=fig)

    # Replace bottom-right cell with subgridspec
    gs_errors = gs[1, 1].subgridspec(3, 1, hspace=0.3)

    # === Error plot for F_x ===
    ax_fx_err = fig.add_subplot(gs_errors[0])
    ax_fx_err.plot(timestamps, [state[3] for state in kalman_x_states] - ground_truth_force[:, 0],
                label='Hawley $\mathbf{e_x}$', color='blue', linewidth=1.5)
    ax_fx_err.plot(timestamps, [state[3]*Mc for state in stephens_x_states] - ground_truth_force[:, 0],
                label='Stephens $\mathbf{e_x}$', color='red', linewidth=1)
    ax_fx_err.plot(timestamps, [state[3] for state in g_kalman_x_states] - ground_truth_force[:, 0],
                label='SENSE $\mathbf{e_x}$', color='green', linewidth=2)
    ax_fx_err.axhline(0, linestyle='--', color='black', linewidth=2.5)
    ax_fx_err.set_ylabel("Error $\mathrm{e_x}$ (N)", fontsize=14)
    ax_fx_err.legend(prop={'size': 8, 'weight': 'bold'})
    ax_fx_err.grid(True)

    # === Error plot for F_y ===
    ax_fy_err = fig.add_subplot(gs_errors[1])
    ax_fy_err.plot(timestamps, [state[3] for state in kalman_y_states] - ground_truth_force[:, 1],
                label='Hawley $\mathbf{e_y}$', color='blue', linewidth=1.5)
    ax_fy_err.plot(timestamps, [state[3]*Mc for state in stephens_y_states] - ground_truth_force[:, 1],
                label='Stephens $\mathbf{e_y}$', color='red', linewidth=1)
    ax_fy_err.plot(timestamps, [state[3] for state in g_kalman_y_states] - ground_truth_force[:, 1],
                label='SENSE $\mathbf{e_y}$', color='green', linewidth=2)
    ax_fy_err.axhline(0, linestyle='--', color='black', linewidth=2.5)
    ax_fy_err.set_ylabel("Error $\mathrm{e_y}$ (N)", fontsize=14)
    ax_fy_err.legend(prop={'size': 8, 'weight': 'bold'})
    ax_fy_err.grid(True)

    # === Error plot for F_z ===
    ax_fz_err = fig.add_subplot(gs_errors[2])
    ax_fz_err.plot(timestamps, [state[3] for state in kalman_z_states] - ground_truth_force[:, 2],
                label='Hawley $\mathbf{e_z}$', color='blue', linewidth=1.5)
    #ax_fz_err.plot(timestamps, [state[3]*Mc for state in stephens_z_states] - ground_truth_force[:, 2],
    #               label='Stephens $\mathbf{F_z}$', color='red', linewidth=1.5)
    ax_fz_err.plot(timestamps, filtered_z - ground_truth_force[:, 2],
                label='SENSE $\mathbf{e_z}$', color='green', linewidth=2)
    ax_fz_err.axhline(0, linestyle='--', color='black', linewidth=2.5)
    ax_fz_err.set_ylabel("Error $\mathrm{e_z}$ (N)", fontsize=14)
    ax_fz_err.set_xlabel("Time (s)", fontsize=14)
    ax_fz_err.legend(prop={'size': 8, 'weight': 'bold'})
    ax_fz_err.grid(True)

    # Final layout
    plt.tight_layout()
    #plt.savefig(f"walking_force_estimation.png", dpi=300, bbox_inches='tight')   # Save as PNG
    #plt.savefig(f"walking_force_estimation.eps", format='eps', bbox_inches='tight')  # Save as EPS

    # === Plot estimated external moments M_x, M_y, M_z ===
    fig_moment, axs_m = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs_m[0].plot(timestamps, [state[4] for state in g_kalman_x_states], label='SENSE $M_x$', color='blue', linewidth=1.5)
    axs_m[0].set_ylabel("Moment $M_x$ (N·m)", fontsize=14)
    axs_m[0].grid(True)
    axs_m[0].legend(loc='upper right', fontsize=10)

    axs_m[1].plot(timestamps, [state[4] for state in g_kalman_y_states], label='SENSE $M_y$', color='green', linewidth=1.5)
    axs_m[1].set_ylabel("Moment $M_y$ (N·m)", fontsize=14)
    axs_m[1].grid(True)
    axs_m[1].legend(loc='upper right', fontsize=10)

    fig_moment.tight_layout()
    # Optionally save this figure
    fig_moment.savefig(f"walking_estimated_moments.png", dpi=300, bbox_inches='tight')
    fig_moment.savefig(f"walking_estimated_moments.eps", format='eps', bbox_inches='tight') 
    plt.show()

    plt.show()


    #viz.normal_plot(com_accel, com_position, zmp_pos, forces, dT)
    #viz.momentum_plot(omega, angular_momentum, dT)

def lowpass_filter(data, cutoff, fs, order=4):
    data = np.nan_to_num(data)  # Remplacer les NaNs par 0 temporairement
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def remove_outliers(data, threshold=2.5):
    data = np.array(data)
    median = np.median(data)
    diff = np.abs(data - median)
    mad = np.median(diff)  # Median Absolute Deviation
    outlier_mask = diff < threshold * mad
    return np.where(outlier_mask, data, np.nan)  # Remplace outliers par NaN

if __name__ == "__main__":
    main()