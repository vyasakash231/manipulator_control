#! /usr/bin/python3
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from impedance_control import CartesianImpedanceControl
from learn_dmp import PositionDMP, QuaternionDMP

# # move to initial position first
# # p1= posj(0,25,110,0,45,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
# p1= posj(0,0,90,0,90,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
# movej(p1, vel=40, acc=20)

# time.sleep(2.0)

# try:
#     # Initialize ROS node first
#     rospy.init_node('My_service_node')
    
#     # Create control object
#     task = CartesianImpedanceControl(file_name="demo_discrete")
#     rospy.sleep(2.0)  # Give time for initialization

#     # Start controller in a separate thread
#     controller_thread = Thread(target=task.run_dmp, args=(75.0, 10.0)) # translation stiff -> N/m, rotational stiffness -> Nm/rad 
#     controller_thread.daemon = True
#     controller_thread.start()
    
#     # Keep the main thread running for the plot
#     while not rospy.is_shutdown():
#         rospy.sleep(0.01)

# except rospy.ROSInterruptException:
#     pass

# finally:
#     # task.save(name="dmp_performed_discrete")  # save data for plotting
#     pass

"""#################################################################################################################"""

def load_demo(name='demo'):
    curr_dir=os.getcwd()
    data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
    q_demo = data['q']    # shape: (N, 6), in rad
    position_demo = 1000 * data['traj']   # shape: (N, 3), convert from m to mm
    orientation_demo = data['ori']   # shape: (N, 4)
    linear_velocity_demo = data['vel']   # shape: (N, 3), in m/s
    angular_velocity_demo = data['omega']   # shape: (N, 3), in rad/s
    N = position_demo.shape[0]   # no of sample points
    return position_demo, orientation_demo, linear_velocity_demo, angular_velocity_demo

# load demo data
position_demo, orientation_demo, linear_velocity_demo, angular_velocity_demo = load_demo(name="demo_discrete")

# Calculate time step between trajectory points (in seconds)
traj_dt = 0.01  # 10Hz = 0.1s between points

# Start DMPS
position_dmp = PositionDMP(no_of_DMPs=3, no_of_basis_func=300, T=1, dt=traj_dt, K=25.0, alpha=1.0)
quaternion_dmp = QuaternionDMP(no_of_basis_func=300, T=1, dt=traj_dt, K=25.0, alpha=1.0)

# learn Weights based on position Demo
X_demo = 0.001 * position_demo.T   # demo position data of shape (3, N) in m
V_demo = linear_velocity_demo.T   # demo velocity data of shape (3, N) in m
position_dmp.learn_dynamics(X_des=X_demo)
position_dmp.reset_state()

# learn Weights based on orientation Demo
Q_demo = orientation_demo.T   # orientation data of shape (4, N)
omega_demo = angular_velocity_demo.T   # demo velocity data of shape (3, N) in m
quaternion_dmp.learn_dynamics(q_des=Q_demo, omega_des=omega_demo)
quaternion_dmp.reset_state()

X_goal = X_demo[:,[-1]]
Q_goal = Q_demo[:,[-1]]

X_track, dX_track = position_dmp.rollout(X_goal)
Q_track, omega_track = quaternion_dmp.rollout(Q_goal)

# ax = plt.figure().add_subplot(projection='3d')
# ax.quiver(X_demo[0,:], X_demo[1,:], X_demo[2,:], V_demo[0,:], V_demo[1,:], V_demo[2,:], length=0.02, color="red", normalize=True)

# ax.quiver(X_track[0,:], X_track[1,:], X_track[2,:], dX_track[0,:], dX_track[1,:], dX_track[2,:], length=0.01, color="blue")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

def quaternion_distance(q1, q2):
    """Calculate the angular distance between two quaternions in degrees."""
    # Ensure quaternions are unit quaternions
    q1 = q1 / np.linalg.norm(q1, axis=0)
    q2 = q2 / np.linalg.norm(q2, axis=0)
    
    # Inner product of quaternions
    dot_product = np.sum(q1 * q2, axis=0)
    
    # Correct for sign (quaternion and its negative represent the same rotation)
    dot_product = np.clip(np.abs(dot_product), -1.0, 1.0)
    
    # Calculate the angle between quaternions
    angle = 2 * np.arccos(dot_product) * 180.0 / np.pi
    
    # Ensure smallest angle (quaternion double covers rotation space)
    angle = np.minimum(angle, 360 - angle)
    
    return angle

def plot_orientation_error():
    """Plot the quaternion orientation error over time."""
    # Calculate quaternion distance at each time step
    distances = quaternion_distance(Q_demo, Q_track)
    
    plt.figure(figsize=(10, 5))
    plt.plot(distances, label='Quaternion Angular Distance')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Distance (degrees)')
    plt.title('Orientation Error Between Demo and Tracked Trajectory')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_quaternion_components():
    """Plot individual quaternion components for comparison."""
    components = ['w', 'x', 'y', 'z']
    
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(Q_demo[i, :], label=f'Demo {components[i]}')
        plt.plot(Q_track[i, :], label=f'Tracked {components[i]}')
        plt.xlabel('Time Step')
        plt.ylabel(f'Quaternion {components[i]} Component')
        plt.title(f'Quaternion {components[i]} Component Comparison')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_euler_angles():
    """Convert quaternions to Euler angles and plot for comparison."""
    # Convert quaternions to Euler angles (in degrees)
    euler_demo = np.zeros((3, Q_demo.shape[1]))
    euler_track = np.zeros((3, Q_track.shape[1]))
    
    for i in range(Q_demo.shape[1]):
        # Convert quaternion [w, x, y, z] to scipy rotation
        # Note: scipy uses [x, y, z, w] order
        rot_demo = R.from_quat([Q_demo[1, i], Q_demo[2, i], Q_demo[3, i], Q_demo[0, i]])
        rot_track = R.from_quat([Q_track[1, i], Q_track[2, i], Q_track[3, i], Q_track[0, i]])
        
        # Get Euler angles in degrees (ZYX convention)
        euler_demo[:, i] = rot_demo.as_euler('zyx', degrees=True)
        euler_track[:, i] = rot_track.as_euler('zyx', degrees=True)
    
    # Plot each Euler angle
    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    plt.figure(figsize=(12, 8))
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(euler_demo[i, :], label=f'Demo {labels[i]}')
        plt.plot(euler_track[i, :], label=f'Tracked {labels[i]}')
        plt.xlabel('Time Step')
        plt.ylabel('Angle (degrees)')
        plt.title(f'{labels[i]} Comparison')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_orientation_frames(step_interval=10):
    """
    Create a 3D visualization of coordinate frames for both demo and tracked orientations.
    Shows frames at specified intervals to avoid cluttering.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get trajectory positions
    X_demo_m = X_demo  # Already in meters
    X_track_m = X_track  # Already in meters
    
    # Plot trajectory
    ax.plot(X_demo_m[0, :], X_demo_m[1, :], X_demo_m[2, :], 'r-', label='Demo Trajectory', alpha=0.5)
    ax.plot(X_track_m[0, :], X_track_m[1, :], X_track_m[2, :], 'b-', label='Tracked Trajectory', alpha=0.5)
    
    # Define axis lengths for the coordinate frames
    axis_length = 0.05  # in meters
    
    # Plot coordinate frames at intervals
    for i in range(0, Q_demo.shape[1], step_interval):
        # Get positions
        pos_demo = X_demo_m[:, i]
        pos_track = X_track_m[:, i]
        
        # Convert quaternions to rotation matrices
        # Note: scipy uses [x, y, z, w] order, our data is [w, x, y, z]
        rot_demo = R.from_quat([Q_demo[1, i], Q_demo[2, i], Q_demo[3, i], Q_demo[0, i]]).as_matrix()
        rot_track = R.from_quat([Q_track[1, i], Q_track[2, i], Q_track[3, i], Q_track[0, i]]).as_matrix()
        
        # Plot x, y, z axes for demo orientation
        ax.quiver(pos_demo[0], pos_demo[1], pos_demo[2], 
                  rot_demo[0, 0] * axis_length, rot_demo[1, 0] * axis_length, rot_demo[2, 0] * axis_length, 
                  color='r', alpha=0.7)
        ax.quiver(pos_demo[0], pos_demo[1], pos_demo[2], 
                  rot_demo[0, 1] * axis_length, rot_demo[1, 1] * axis_length, rot_demo[2, 1] * axis_length, 
                  color='g', alpha=0.7)
        ax.quiver(pos_demo[0], pos_demo[1], pos_demo[2], 
                  rot_demo[0, 2] * axis_length, rot_demo[1, 2] * axis_length, rot_demo[2, 2] * axis_length, 
                  color='b', alpha=0.7)
        
        # Plot x, y, z axes for tracked orientation
        ax.quiver(pos_track[0], pos_track[1], pos_track[2], 
                  rot_track[0, 0] * axis_length, rot_track[1, 0] * axis_length, rot_track[2, 0] * axis_length, 
                  color='darkred', alpha=0.7)
        ax.quiver(pos_track[0], pos_track[1], pos_track[2], 
                  rot_track[0, 1] * axis_length, rot_track[1, 1] * axis_length, rot_track[2, 1] * axis_length, 
                  color='darkgreen', alpha=0.7)
        ax.quiver(pos_track[0], pos_track[1], pos_track[2], 
                  rot_track[0, 2] * axis_length, rot_track[1, 2] * axis_length, rot_track[2, 2] * axis_length, 
                  color='darkblue', alpha=0.7)
    
    # Add some dummy points for the legend
    ax.plot([], [], 'r-', label='Demo Position')
    ax.plot([], [], 'b-', label='Tracked Position')
    ax.plot([], [], 'r-', marker='>', label='Demo Orientation')
    ax.plot([], [], 'darkred-', marker='>', label='Tracked Orientation')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Position and Orientation Comparison')
    ax.legend()
    
    # Equal aspect ratio
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    
    plt.tight_layout()
    plt.show()

def create_animation():
    """Create an animation of both trajectories and orientations."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get trajectory positions
    X_demo_m = X_demo  # Already in meters
    X_track_m = X_track  # Already in meters
    
    # Plot full trajectories with lower alpha
    ax.plot(X_demo_m[0, :], X_demo_m[1, :], X_demo_m[2, :], 'r-', alpha=0.3)
    ax.plot(X_track_m[0, :], X_track_m[1, :], X_track_m[2, :], 'b-', alpha=0.3)
    
    # Define axis lengths for the coordinate frames
    axis_length = 0.05  # in meters
    
    # Fixed points to maintain constant view
    min_x = min(np.min(X_demo_m[0, :]), np.min(X_track_m[0, :]))
    max_x = max(np.max(X_demo_m[0, :]), np.max(X_track_m[0, :]))
    min_y = min(np.min(X_demo_m[1, :]), np.min(X_track_m[1, :]))
    max_y = max(np.max(X_demo_m[1, :]), np.max(X_track_m[1, :]))
    min_z = min(np.min(X_demo_m[2, :]), np.min(X_track_m[2, :]))
    max_z = max(np.max(X_demo_m[2, :]), np.max(X_track_m[2, :]))
    
    # Set axis limits
    ax.set_xlim([min_x - 0.1, max_x + 0.1])
    ax.set_ylim([min_y - 0.1, max_y + 0.1])
    ax.set_zlim([min_z - 0.1, max_z + 0.1])
    
    # Plot settings
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Position and Orientation Comparison')
    
    # Initialize points and frames that will be updated in the animation
    demo_point = ax.plot([X_demo_m[0, 0]], [X_demo_m[1, 0]], [X_demo_m[2, 0]], 'ro', markersize=5)[0]
    track_point = ax.plot([X_track_m[0, 0]], [X_track_m[1, 0]], [X_track_m[2, 0]], 'bo', markersize=5)[0]
    
    # Create initial orientation arrows for both trajectories
    # Convert first quaternions to rotation matrices
    rot_demo = R.from_quat([Q_demo[1, 0], Q_demo[2, 0], Q_demo[3, 0], Q_demo[0, 0]]).as_matrix()
    rot_track = R.from_quat([Q_track[1, 0], Q_track[2, 0], Q_track[3, 0], Q_track[0, 0]]).as_matrix()
    
    # Demo orientation arrows (x, y, z axes)
    demo_arrow_x = ax.quiver(X_demo_m[0, 0], X_demo_m[1, 0], X_demo_m[2, 0],
                             rot_demo[0, 0] * axis_length, rot_demo[1, 0] * axis_length, rot_demo[2, 0] * axis_length, 
                             color='r')
    demo_arrow_y = ax.quiver(X_demo_m[0, 0], X_demo_m[1, 0], X_demo_m[2, 0],
                             rot_demo[0, 1] * axis_length, rot_demo[1, 1] * axis_length, rot_demo[2, 1] * axis_length, 
                             color='g')
    demo_arrow_z = ax.quiver(X_demo_m[0, 0], X_demo_m[1, 0], X_demo_m[2, 0],
                             rot_demo[0, 2] * axis_length, rot_demo[1, 2] * axis_length, rot_demo[2, 2] * axis_length, 
                             color='b')
    
    # Track orientation arrows (x, y, z axes)
    track_arrow_x = ax.quiver(X_track_m[0, 0], X_track_m[1, 0], X_track_m[2, 0],
                              rot_track[0, 0] * axis_length, rot_track[1, 0] * axis_length, rot_track[2, 0] * axis_length, 
                              color='darkred')
    track_arrow_y = ax.quiver(X_track_m[0, 0], X_track_m[1, 0], X_track_m[2, 0],
                              rot_track[0, 1] * axis_length, rot_track[1, 1] * axis_length, rot_track[2, 1] * axis_length, 
                              color='darkgreen')
    track_arrow_z = ax.quiver(X_track_m[0, 0], X_track_m[1, 0], X_track_m[2, 0],
                              rot_track[0, 2] * axis_length, rot_track[1, 2] * axis_length, rot_track[2, 2] * axis_length, 
                              color='darkblue')
    
    # Add legend
    ax.plot([], [], 'r-', label='Demo Trajectory')
    ax.plot([], [], 'b-', label='Tracked Trajectory')
    ax.legend()
    
    # Function to update the animation
    def update(frame):
        # Skip frames to make animation faster but still smooth
        i = min(frame * 3, Q_demo.shape[1] - 1)
        
        # Update points
        demo_point.set_data([X_demo_m[0, i]], [X_demo_m[1, i]])
        demo_point.set_3d_properties([X_demo_m[2, i]])
        
        track_point.set_data([X_track_m[0, i]], [X_track_m[1, i]])
        track_point.set_3d_properties([X_track_m[2, i]])
        
        # Update orientations
        # Convert quaternions to rotation matrices
        rot_demo = R.from_quat([Q_demo[1, i], Q_demo[2, i], Q_demo[3, i], Q_demo[0, i]]).as_matrix()
        rot_track = R.from_quat([Q_track[1, i], Q_track[2, i], Q_track[3, i], Q_track[0, i]]).as_matrix()
        
        # Remove old arrows
        demo_arrow_x.remove()
        demo_arrow_y.remove()
        demo_arrow_z.remove()
        track_arrow_x.remove()
        track_arrow_y.remove()
        track_arrow_z.remove()
        
        # Create new arrows
        nonlocal demo_arrow_x, demo_arrow_y, demo_arrow_z
        nonlocal track_arrow_x, track_arrow_y, track_arrow_z
        
        demo_arrow_x = ax.quiver(X_demo_m[0, i], X_demo_m[1, i], X_demo_m[2, i],
                                 rot_demo[0, 0] * axis_length, rot_demo[1, 0] * axis_length, rot_demo[2, 0] * axis_length, 
                                 color='r')
        demo_arrow_y = ax.quiver(X_demo_m[0, i], X_demo_m[1, i], X_demo_m[2, i],
                                 rot_demo[0, 1] * axis_length, rot_demo[1, 1] * axis_length, rot_demo[2, 1] * axis_length, 
                                 color='g')
        demo_arrow_z = ax.quiver(X_demo_m[0, i], X_demo_m[1, i], X_demo_m[2, i],
                                 rot_demo[0, 2] * axis_length, rot_demo[1, 2] * axis_length, rot_demo[2, 2] * axis_length, 
                                 color='b')
        
        track_arrow_x = ax.quiver(X_track_m[0, i], X_track_m[1, i], X_track_m[2, i],
                                  rot_track[0, 0] * axis_length, rot_track[1, 0] * axis_length, rot_track[2, 0] * axis_length, 
                                  color='darkred')
        track_arrow_y = ax.quiver(X_track_m[0, i], X_track_m[1, i], X_track_m[2, i],
                                  rot_track[0, 1] * axis_length, rot_track[1, 1] * axis_length, rot_track[2, 1] * axis_length, 
                                  color='darkgreen')
        track_arrow_z = ax.quiver(X_track_m[0, i], X_track_m[1, i], X_track_m[2, i],
                                  rot_track[0, 2] * axis_length, rot_track[1, 2] * axis_length, rot_track[2, 2] * axis_length, 
                                  color='darkblue')
        
        return (demo_point, track_point, demo_arrow_x, demo_arrow_y, demo_arrow_z, 
                track_arrow_x, track_arrow_y, track_arrow_z)
    
    # Create animation
    num_frames = min(100, Q_demo.shape[1] // 3)  # Limit to 100 frames for performance
    anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    
    # Save animation (optional)
    # anim.save('orientation_animation.mp4', writer='ffmpeg', fps=20)
    
    plt.tight_layout()
    plt.show()
    
    return anim

# Use these functions to visualize orientation data
plot_quaternion_components()
plot_euler_angles()
plot_orientation_error()
visualize_orientation_frames(step_interval=20)  # Adjust step_interval based on trajectory length
# create_animation()  # Uncomment to create animation (may be computationally intensive)


