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

#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def setup_plots():
    # Initialize the figure and axis for plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Set titles and labels for each subplot
    ax1.set_title('Real-time CoM Acceleration Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Acceleration [m/s^2]')
    
    ax2.set_title('Real-time CoM Position Data')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position [m]')
    
    ax3.set_title('Real-time ZMP Position Data')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Position [m]')
    
    ax4.set_title('Real-time FSR Data')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Force [kg]')
    
    # Plot initial legends
    ax1.plot([], [], color='r', label='X-axis')
    ax1.plot([], [], color='g', label='Y-axis')
    ax1.plot([], [], color='b', label='Z-axis')
    ax1.legend()

    ax2.plot([], [], color='r', label='X-axis')
    ax2.plot([], [], color='g', label='Y-axis')
    ax2.plot([], [], color='b', label='Z-axis')
    ax2.legend()

    ax3.plot([], [], color='r', label='X-axis')
    ax3.plot([], [], color='g', label='Y-axis')
    ax3.legend()
    
    ax4.plot([], [], color='r', label='L-foot')
    ax4.plot([], [], color='g', label='R-foot')
    ax4.legend()
    
    return fig, ((ax1, ax2), (ax3, ax4))

def update_plots(axes, time_data, acceleration_data, position_data, zmp_data, fsr_data):
    global k
    k += 1
    # Unpack axes
    ((ax1, ax2), (ax3, ax4)) = axes
    
    # Update CoM acceleration data
    ax1.lines[0].set_data(time_data, acceleration_data[:k, 0])  # X-axis
    ax1.lines[1].set_data(time_data, acceleration_data[:k, 1])  # Y-axis
    ax1.lines[2].set_data(time_data, acceleration_data[:k, 2])  # Z-axis

    # Update CoM position data
    ax2.lines[0].set_data(time_data, position_data[:k, 0])  # X-axis
    ax2.lines[1].set_data(time_data, position_data[:k, 1])  # Y-axis
    ax2.lines[2].set_data(time_data, position_data[:k, 2])  # Z-axis

    # Plot ZMP position data
    ax3.lines[0].set_data(time_data, zmp_data[:k, 0])  # X-axis
    ax3.lines[1].set_data(time_data, zmp_data[:k, 1])  # Y-axis

    # Plot FSR force data
    ax4.lines[0].set_data(time_data, fsr_data[:k, 0])  # L-foot
    ax4.lines[1].set_data(time_data, fsr_data[:k, 1])  # R-foot

    # Adjust plot limits
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    #print(k)
    
    return [ax1.lines[0], ax1.lines[1], ax1.lines[2], ax2.lines[0], ax2.lines[1], ax2.lines[2],
            ax3.lines[0], ax3.lines[1], ax4.lines[0], ax4.lines[1]]

def update_plots_real(axes, time_data, acceleration_data, position_data, zmp_data, fsr_data):
    # Unpack axes
    ((ax1, ax2), (ax3, ax4)) = axes
    
    # Update CoM acceleration data
    ax1.lines[0].set_data(time_data, acceleration_data[:, 0])  # X-axis
    ax1.lines[1].set_data(time_data, acceleration_data[:, 1])  # Y-axis
    ax1.lines[2].set_data(time_data, acceleration_data[:, 2])  # Z-axis

    # Update CoM position data
    ax2.lines[0].set_data(time_data, position_data[:, 0])  # X-axis
    ax2.lines[1].set_data(time_data, position_data[:, 1])  # Y-axis
    ax2.lines[2].set_data(time_data, position_data[:, 2])  # Z-axis

    # Plot ZMP position data
    ax3.lines[0].set_data(time_data, zmp_data[:, 0])  # X-axis
    ax3.lines[1].set_data(time_data, zmp_data[:, 1])  # Y-axis

    # Plot FSR force data
    ax4.lines[0].set_data(time_data, fsr_data[:, 0])  # L-foot
    ax4.lines[1].set_data(time_data, fsr_data[:, 1])  # R-foot

    # Adjust plot limits
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    #print(k)
    
    return [ax1.lines[0], ax1.lines[1], ax1.lines[2], ax2.lines[0], ax2.lines[1], ax2.lines[2],
            ax3.lines[0], ax3.lines[1], ax4.lines[0], ax4.lines[1]]

def normal_plot(acceleration_data, position_data, zmp_data, fsr_data, sampling_step):
    # Initialize the figure and axis for plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    num_samples = len(acceleration_data)
    time_data = np.linspace(0, (num_samples - 1) * sampling_step, num_samples)

    # Set titles and labels for each subplot
    ax1.set_title('Real-time CoM Acceleration Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Acceleration [m/s^2]')
    
    ax2.set_title('Real-time CoM Position Data')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position [m]')
    
    ax3.set_title('Real-time ZMP Position Data')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Position [m]')
    
    ax4.set_title('Real-time FSR Data')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Force [kg]')
    
    # Plot initial legends
    ax1.plot(time_data, acceleration_data[:, 0], color='r', label='X-axis')
    ax1.plot(time_data, acceleration_data[:, 1], color='g', label='Y-axis')
    ax1.plot(time_data, acceleration_data[:, 2], color='b', label='Z-axis')
    ax1.legend()

    ax2.plot(time_data, position_data[:, 0], color='r', label='X-axis')
    ax2.plot(time_data, position_data[:, 1], color='g', label='Y-axis')
    ax2.plot(time_data, position_data[:, 2], color='b', label='Z-axis')
    ax2.plot(time_data, np.zeros(len(time_data)), color='y', linestyle='--')
    ax2.legend()

    ax3.plot(time_data, zmp_data[:, 0], color='r', label='X-axis')
    ax3.plot(time_data, zmp_data[:, 1], color='g', label='Y-axis')
    ax3.plot(time_data, np.zeros(len(time_data)), color='y', linestyle='--')
    ax3.legend()
    
    ax4.plot(time_data, fsr_data[:, 0], color='r', label='L-foot')
    ax4.plot(time_data, fsr_data[:, 1], color='g', label='R-foot')
    ax4.legend()

    plt.show()
    
def online_plot(acceleration_data, position_data, zmp_data, fsr_data, sampling_step, T_anim):
    num_samples = len(acceleration_data)
    fig, axes = setup_plots()
    time_data = []
    global k
    k = 0 

    def update_plot(frame):
        time_data.append(k * sampling_step)
        artists = update_plots(axes, time_data, acceleration_data, position_data, zmp_data, fsr_data)
        return artists
    
    def stop_animation():    
        if k == num_samples - 1:  # Check if we have reached the desired number of samples
            ani.event_source.stop()  # Stop the FuncAnimation
        

    ani = FuncAnimation(fig, update_plot, frames=range(num_samples), interval=T_anim, blit=True)
    ani.event_source.add_callback(stop_animation)  # Add callback to stop animation
    plt.show()

def momentum_plot(omega_data, momentum_data, sampling_step):
    # Initialize the figure and axis for plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    num_samples = len(omega_data)
    time_data = np.linspace(0, (num_samples - 1) * sampling_step, num_samples)

    # Set titles and labels for each subplot
    ax1.set_title('Real-time CoM Angular velocity Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Angular velocity [rad/s^2]')
    
    ax2.set_title('Real-time CoM Momentum Data')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Momentum ')
    
    # Plot initial legends
    ax1.plot(time_data, omega_data[:, 0], color='r', label='X-axis')
    ax1.plot(time_data, omega_data[:, 1], color='g', label='Y-axis')
    ax1.plot(time_data, omega_data[:, 2], color='b', label='Z-axis')
    ax2.plot(time_data, np.zeros(len(time_data)), color='y', linestyle='--')
    ax1.legend()

    ax2.plot(time_data, momentum_data[:, 0], color='r', label='X-axis')
    ax2.plot(time_data, momentum_data[:, 1], color='g', label='Y-axis')
    ax2.plot(time_data, momentum_data[:, 2], color='b', label='Z-axis')
    ax2.plot(time_data, np.zeros(len(time_data)), color='y', linestyle='--')
    ax2.legend()

    plt.show()