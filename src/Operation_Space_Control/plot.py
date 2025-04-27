#! /usr/bin/python3
import os, sys
from math import *
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from common_utils import Robot_KM
from pose_transform import quat2mat, make_quat_continuity


class PlotData:
    def __init__(self):
        # load demo data
        self.load_demo(name='demo')

        # Check if the file of performed task already exists, if it exists load the data.
        curr_dir=os.getcwd()
        if os.path.exists(curr_dir + '/data/task_performed' + '.npz'):
            print("task_performed already exists.")
            self.load_task_performed(name="task_performed")

        # Modified-DH Parameters 
        self.DOF = 6
        alpha = np.array([0, -np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2])   
        a = np.array([0, 0, 0.409, 0, 0, 0])  # data from parameter data-sheet (in meters)
        d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])  # data from parameter data-sheet (in meters)
        d_nn = np.array([[0.0], [0.0], [0.0]])  # TCP coord in end-effector frame
        DH_params="modified"

        self.KM = Robot_KM(self.DOF, alpha, a, d, d_nn, DH_params)

        # Draw unit sphere for reference (vector part of unit quaternion lies within this)
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        self.sphere_x = np.outer(np.cos(u), np.sin(v))
        self.sphere_y = np.outer(np.sin(u), np.sin(v))
        self.sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

    def load_demo(self, name='demo'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.q_demo = data['q']    # shape: (N, 6), in rad
        self.position_demo = data['traj']   # shape: (N, 3), in m
        self.orientation_demo = data['ori']   # shape: (N, 4)
        self.linear_velocity_demo = data['vel']   # shape: (N, 3), in m/s
        self.angular_velocity_demo = data['omega']   # shape: (N, 3), in rad/s

        # Important to make quaternions continuous 
        self.orientation_demo = make_quat_continuity(self.orientation_demo)

        self.N_demo = self.position_demo.shape[0]   # no of sample points

    def load_task_performed(self, name='task_performed'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.position = data['traj']   # shape: (N, 3), in m
        self.orientation = data['ori']   # shape: (N, 4)
        self.motor_torque = data['motor_torque']   # shape: (N, 3), in Nm
        self.joint_torque = data['joint_torque']   # shape: (N, 3), in Nm

        # Important to make quaternions continuous 
        self.orientation = -make_quat_continuity(self.orientation)

        self.N = self.position.shape[0]   # no of sample points

    """#####################################################################################################################"""
    
    def setup_performed_task_plot(self):
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 3)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0,0], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,1], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,2]))
        return self.create_animation_2(self.update_plot_2)
    
    """#####################################################################################################################"""
    
    def create_animation_2(self, update_func):
        # Adjust interval based on data size to maintain smooth animation
        if self.N > 400:  # Very high frequency data (e.g., 50Hz)
            frames = np.linspace(0, self.N-1, 250, dtype=int)    # use a subset of frames for smoother playback
        else:
            frames = self.N-1  # use all the frames

        anim = FuncAnimation(self.fig, update_func, frames=frames, interval=10, blit=False, repeat=False)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        return anim
    
    """#####################################################################################################################"""
    
    def update_plot_2(self, frame):
        # Robot visualization
        self._plot_4(self.axs[0], frame)  # position data
        self._plot_5(self.axs[1], frame)  # orientation data
        self._plot_6(self.axs[2], frame)  # orientation data
        self._set_3d_plot_properties(self.axs[0], x_lim=[-0.25, 1.0], y_lim=[-0.5, 0.5], z_lim=[-0.5, 1.0])   # set graph properties 
        self._set_3d_plot_properties(self.axs[1],  x_lim=[-1.0, 1.0], y_lim=[-1.0, 1.0], z_lim=[-1.0, 1.0])   # set graph properties 
        return self.axs
    
    """#####################################################################################################################"""

    def _plot_4(self, ax, k):
        ax.clear()
        ax.plot(self.position_demo[:,0], self.position_demo[:,1], self.position_demo[:,2], "k--", linewidth=1.5)
        ax.plot(self.position[:k+1,0], self.position[:k+1,1], self.position[:k+1,2], "r", linewidth=1.25)

    def _plot_5(self, ax, k):
        ax.clear()

        # Plot unit sphere
        ax.plot_surface(self.sphere_x, self.sphere_y, self.sphere_z, color="lightblue", alpha=0.3)

        # Normalize quaternions to ensure they're on the unit hypersphere
        q_demo_norms = np.sqrt(self.orientation_demo[:,0]**2 + self.orientation_demo[:,1]**2 + self.orientation_demo[:,2]**2 + self.orientation_demo[:,3]**2)
       
        qx_demo_norm = self.orientation_demo[:,0] / q_demo_norms
        qy_demo_norm = self.orientation_demo[:,1] / q_demo_norms
        qz_demo_norm = self.orientation_demo[:,2] / q_demo_norms
            
        # Plot the trajectory line
        ax.plot(qx_demo_norm, qy_demo_norm, qz_demo_norm, 'k-', linewidth=2.0)

        # Draw reference unit vectors
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.5, length=0.5, normalize=True)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, length=0.5, normalize=True)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.5, length=0.5, normalize=True)
    
        # Normalize quaternions to ensure they're on the unit hypersphere
        q_norms = np.sqrt(self.orientation[:k+1,0]**2 + self.orientation[:k+1,1]**2 + self.orientation[:k+1,2]**2 + self.orientation[:k+1,3]**2)
        
        qx_norm = self.orientation[:k+1,0] / q_norms
        qy_norm = self.orientation[:k+1,1] / q_norms
        qz_norm = self.orientation[:k+1,2] / q_norms
            
        # Plot the trajectory line
        ax.plot(qx_norm, qy_norm, qz_norm, 'r-', linewidth=1.5)

    def _plot_6(self, ax, k):
        ax.clear()
        pass

    """#####################################################################################################################"""

    def _set_3d_plot_properties(self, ax, x_lim, y_lim, z_lim):
        # ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_zlim(z_lim[0], z_lim[1])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

    def show(self):
        plt.show()

if __name__ == "__main__":
    plotter = PlotData()

    anim = plotter.setup_performed_task_plot()
    plotter.show()