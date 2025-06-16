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
#matplotlibuse('TkAgg')  # Or 'Qt5Agg', depending on what is available
from config_qibullet import *

def load_json(filename):
    """ Load JSON file and handle errors """
    with open(filename, "r") as f:
        return json.load(f)
    
# Function to extract state values safely
def extract_states(estimates_dict, observer_name, key_name):
    # Check if the observer exists and is a list
    if observer_name in estimates_dict and isinstance(estimates_dict[observer_name], list):
        return [
            entry["state"][key_name] if "state" in entry and key_name in entry["state"] else None
            for entry in estimates_dict[observer_name]
        ]
    return []

def main():
    # Load estimates data
    data = load_json(data_folder / f"online_data_{data_subname}_{force_type}_{link_name}.json")
    estimates = load_json(data_folder / f"online_estimates_{data_subname}_{force_type}_{link_name}.json")
    
    
    timestamps = np.array(range(len(data))) * dT

    ground_truth_force = np.array([sample["applied_force"] for sample in data])
    angular_momentum = np.array([sample["angular_momentum_dot"] for sample in data])
    com_position = np.array([sample["com_pos"] for sample in data])

    # Extracting the state values with key checks
    # Extracting Kalman states
    kalman_x_states = extract_states(estimates, "kalman", "kalman_x")
    kalman_y_states = extract_states(estimates, "kalman", "kalman_y")
    kalman_z_states = extract_states(estimates, "kalman", "kalman_z")

    # Extracting Kalman-G states
    g_kalman_x_states = extract_states(estimates, "kalman_g", "kalmgan_x")
    g_kalman_y_states = extract_states(estimates, "kalman_g", "kalmgan_y")
    g_kalman_z_states = extract_states(estimates, "kalman_g", "kalmgan_z")

    # Extracting Stephens states
    stephens_x_states = extract_states(estimates, "stephens", "stephens_x")
    stephens_y_states = extract_states(estimates, "stephens", "stephens_y")
    stephens_z_states = extract_states(estimates, "stephens", "stephens_z")

    # Filter out None values from the lists
    stephens_x_states = [state for state in stephens_x_states if state is not None]
    stephens_y_states = [state for state in stephens_y_states if state is not None]
    stephens_z_states = [state for state in stephens_z_states if state is not None]

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
    #axs[1,0].plot(timestamps, [state[3]*Mc for state in stephens_z_states], label='Simplified Kalman $F_z$', color='green', linewidth=3)
    axs[1,0].plot(timestamps, [state[3] for state in g_kalman_z_states], label='SENSE $\mathbf{F_z}$', color='green', linewidth=2)
    axs[1,0].set_xlabel("Time (s)", fontsize=14)
    axs[1,0].set_ylabel("Force (N)", fontsize=14)
    axs[1,0].legend(prop={'size': 10, 'weight': 'bold'})
    axs[1,0].grid(True)

    # Create inset (zoom) inside axs[1, 0]
    axins = inset_axes(axs[1, 0], width="20%", height="20%", loc='center') #'lower right' 'center'

    # Same plot inside inset
    axins.plot(timestamps, ground_truth_force[:, 2], color='black', linestyle='--', linewidth=3)
    axins.plot(timestamps, [state[3] for state in kalman_z_states], color='blue', linewidth=1.5)
    axins.plot(timestamps, [state[3] for state in g_kalman_z_states], color='green', linewidth=2)

    # Define zoomed region (adjust this based on your data)
    x1, x2 = 24.8, 28   # time range
    y1, y2 = -0.5, 0.3    # force range
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
    #               label='Stephens $F_z$', color='red', linewidth=1.5)
    ax_fz_err.plot(timestamps, [state[3] for state in g_kalman_z_states] - ground_truth_force[:, 2],
                label='SENSE $\mathbf{e_z}$', color='green', linewidth=2)
    ax_fz_err.axhline(0, linestyle='--', color='black', linewidth=2.5)
    ax_fz_err.set_ylabel("Error $\mathrm{e_z}$ (N)", fontsize=14)
    ax_fz_err.set_xlabel("Time (s)", fontsize=14)
    ax_fz_err.legend(prop={'size': 8, 'weight': 'bold'})
    ax_fz_err.grid(True)

    # Final layout
    plt.tight_layout()
    #plt.savefig(f"force_estimation_{force_type}.png", dpi=300, bbox_inches='tight')   # Save as PNG
    #plt.savefig(f"force_estimation_{force_type}.eps", format='eps', bbox_inches='tight')  # Save as EPS

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
    fig_moment.savefig(f"estimated_moments_{force_type}.png", dpi=300, bbox_inches='tight')
    fig_moment.savefig(f"estimated_moments_{force_type}.eps", format='eps', bbox_inches='tight') 
    plt.show()

if __name__ == "__main__":
    main()