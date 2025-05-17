#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot
from scipy.spatial.transform import Rotation

class OSC(Robot):
    def __init__(self, file_name='demo'):
        self.shutdown_flag = False  
        self.file_name = file_name

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        # self.plotter.setup_plots_1()

        # Setup Gain Values
        translational_gain = 300.0
        rotational_gain = 350.0

        self.Kp = np.eye(6)
        self.Kp[:3, :3] *= translational_gain
        self.Kp[3:, 3:] *= rotational_gain
        
        self.Kv = np.eye(6)
        self.Kv[:3, :3] *= 1.25 * np.sqrt(translational_gain)
        self.Kv[3:, 3:] *= 0.5 * np.sqrt(rotational_gain)

        self.Ko = np.diag([0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

        # Current robot state variables
        self.tau_J_d = np.zeros(6)  # Previous desired torque
        
        # Maximum allowed torque rate change
        self.delta_tau_max = 1.0
        
        # Initial estimated frictional torque
        self.tau_f = np.zeros(self.n)

        super().__init__()

    def load_demo(self, name='demo'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.position_demo = 1000 * data['traj']   # shape: (N, 3), convert from m to mm
        self.orientation_demo = data['ori']   # shape: (N, 4)
        self.linear_velocity_demo = data['vel']   # shape: (N, 3), in m/s
        self.angular_velocity_demo = data['omega']   # shape: (N, 3), in rad/s
        self.N = self.position_demo.shape[0]   # no of sample points

    def store_data(self):
        pos = 0.001 * self.Robot_RT_State.actual_tcp_position_abs[:3]    # in m
        orient = self._eul2quat(self.Robot_RT_State.actual_tcp_position_abs[3:])   # quaternions
        self.record_trajectory = np.vstack((self.record_trajectory, pos))  # shape: (N, 3) 
        self.record_orientation = np.vstack((self.record_orientation, orient))  # shape: (N, 4)

        self.record_motor_torque = np.vstack((self.record_motor_torque, self.Robot_RT_State.actual_motor_torque))  # shape: (N, 6) 
        self.record_joint_torque = np.vstack((self.record_joint_torque, self.Robot_RT_State.actual_joint_torque))  # shape: (N, 6)

    def plot_data(self):
        try:
            # self.plotter.update_data_1(self.data.actual_motor_torque, self.data.raw_force_torque, self.data.actual_joint_torque, self.data.raw_joint_torque)
            pass
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def start(self):
        # Set equilibrium point to current state
        # self.position_des = np.array([200, 0, 750])   # (x, y, z) in mm
        # self.orientation_des = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:].copy())   # Convert angles from Euler ZYZ (in degrees) to quaternion  
        
        self.linear_vel_des = None
        self.angular_vel_des = None

        # for plotting
        self.record_trajectory = 0.001 * self.Robot_RT_State.actual_tcp_position_abs[:3]   # in m
        self.record_orientation = self._eul2quat(self.Robot_RT_State.actual_tcp_position_abs[3:])   # quaternions
        self.record_motor_torque = self.Robot_RT_State.actual_motor_torque  # in Nm
        self.record_joint_torque = self.Robot_RT_State.actual_joint_torque  # in Nm

        # load demo trajectory
        self.load_demo(name=self.file_name)
        rospy.loginfo("OperationalSpaceController: Controller started")

    @property
    def current_velocity(self):
        self.q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs  # convert from deg/s to rad/s
        X_dot = self.J @ self.q_dot[:,np.newaxis]
        return X_dot.reshape(-1)   # [Vx, Vy, Vz, ωx, ωy, ωz] in [m/s, rad/s]
    
    @property
    def velocity_error(self):
        # Combine into a single 6D velocity vector
        if self.linear_vel_des is None and self.angular_vel_des is None:
            return None
        else:
            desired_velocity = np.zeros(self.n)  
            desired_velocity[:3] = self.linear_vel_des   #  in m/s
            desired_velocity[3:] = self.angular_vel_des   # in rad/s

            # Calculate velocity error (current - desired)
            return self.current_velocity - desired_velocity
    
    @property
    def position_error(self):
        # actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        current_position = self.Robot_RT_State.actual_tcp_position_abs[:3]   #  (x, y, z) in mm
        return 0.001 * (current_position - self.position_des)  # convert from mm to m
    
    # @property
    # def orientation_error(self):
    #     current_orientation = self._eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # Convert angles from Euler ZYZ (in degrees) to quaternion        

    #     if np.dot(current_orientation, self.orientation_des) < 0.0:
    #         current_orientation = -current_orientation
        
    #     current_rotation = Rotation.from_quat(current_orientation)  # default order: [x,y,z,w]
    #     desired_rotation = Rotation.from_quat(self.orientation_des)  # default order: [x,y,z,w]
        
    #     # Compute the "difference" or quaternion_distance (q_error = q_current^-1 * q_desired)
    #     """https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames"""
    #     error_rotation = current_rotation.inv() * desired_rotation
    #     error_quat = error_rotation.as_quat()  # default order: [x,y,z,w]
        
    #     # Extract x, y, z components of error quaternion
    #     rot_error = error_quat[:3][:, np.newaxis]
        
    #     # Transform orientation error to base frame
    #     current_rotation_matrix = current_rotation.as_matrix()  # this returns a 3x3 rotation matrix
    #     rot_error = current_rotation_matrix @ rot_error
    #     return rot_error.reshape(-1)
    
    @property
    def orientation_error(self):
        """
        rotation vector directly encodes rotation magnitude and axis, making it more intuitive
        It avoids the nonlinear scaling issues that can occur with quaternion components
        For control purposes, the rotation vector components often provide a more useful error 
        signal that's proportional to the rotation needed
        """
        current_orientation = self._eul2quat(self.Robot_RT_State.actual_tcp_position_abs[3:])
        
        if np.dot(current_orientation, self.orientation_des) < 0.0:
            current_orientation = -current_orientation
        
        current_rotation = Rotation.from_quat(current_orientation)
        desired_rotation = Rotation.from_quat(self.orientation_des)
        
        # Compute the difference quaternion
        error_rotation = current_rotation.inv() * desired_rotation
        
        # Use rotation vector instead of quaternion components
        rot_error = error_rotation.as_rotvec()[:, np.newaxis]
        
        # Transform orientation error to base frame
        current_rotation_matrix = current_rotation.as_matrix()
        rot_error = - current_rotation_matrix @ rot_error
        return rot_error.reshape(-1)
        
    def Mx(self, Mq):
        # Mq_inv = np.linalg.inv(Mq)  # This was calculated based on LU-Decomposition which is numerically not very stable
        Mq_inv = self._svd_solve(Mq)  # SVD is more numerically stable when dealing with matrices that might be ill-conditioned
        Mx_inv = self.J @ (Mq_inv @ self.J.T)
        if abs(np.linalg.det(Mx_inv)) >= 1e-4:
            Mx = self._svd_solve(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-5)
        return Mx
    
    def saturate_torque(self, tau, tau_J_d):
        """Limit both the torque rate of change and peak torque values for Doosan A0509 robot"""
        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # First limit rate of change as in your original function
        # tau_rate_limited = np.zeros(self.n)
        # for i in range(self.n):
        #     difference = tau[i] - tau_J_d[i]
        #     tau_rate_limited[i] = tau_J_d[i] + np.clip(difference, -self.delta_tau_max, self.delta_tau_max)
        # tau = tau_rate_limited.copy()

        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.95
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0])  # Nm

        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # Clip torque values to stay within limits (both positive and negative)
        tau_saturated = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau_saturated

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque   # in Nm
        joint_torque = self.Robot_RT_State.actual_joint_torque   # in Nm

        term_1 = np.dot(self.Ko, (motor_torque - joint_torque - self.tau_f)) * 0.005
        self.tau_f = self.tau_f + term_1

    @property
    def control_input(self):
        # define EE-Position & Orientation error in task-space
        error = np.zeros(self.n)
        error[:3] = self.position_error
        error[3:] = self.orientation_error
                
        if self.velocity_error is None:
            return self.Kp @ error[:, np.newaxis]
        else:                
            error_dot = self.velocity_error[:, np.newaxis]
            return self.Kp @ error[:, np.newaxis] + self.Kv @ error_dot

    def run_controller(self):
        self.start()
        tau_task = np.zeros((self.n,1))

        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate

        # Calculate time step between trajectory points (in seconds)
        traj_dt = 1.0/25.0  # 10Hz = 0.1s between points
        
        # Track start time for trajectory indexing
        start_time = rospy.Time.now().to_sec()

        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                # Calculate elapsed time and determine trajectory index
                current_time = rospy.Time.now().to_sec()
                elapsed_time = current_time - start_time
                current_idx = int(elapsed_time / traj_dt)
                
                # Ensure index is within bounds
                if current_idx >= self.N:
                    current_idx = self.N - 1
                    if current_idx == self.N - 1 and np.linalg.norm(self.position_error)*1000 < 10.0:  # if error is less then 10mm break
                        rospy.loginfo("Trajectory complete and position error < 10mm")
                        break

                # Update desired position and orientation
                self.position_des = self.position_demo[current_idx,:]  # (x, y, z) in mm
                self.orientation_des = self.orientation_demo[current_idx,:]

                # Get desired velocity from recorded data
                self.linear_vel_des = self.linear_velocity_demo[current_idx,:]  # Already in m/s
                self.angular_vel_des = self.angular_velocity_demo[current_idx,:]  # Already in rad/s

                # Find Jacobian matrix
                self.J = self.Robot_RT_State.jacobian_matrix

                # Find Inertia matrix in joint space
                Mq = self.Robot_RT_State.mass_matrix
                Mx = self.Mx(Mq)

                if self.velocity_error is None:
                    # if there's no desired velocity in task space, compensate for velocity in joint space 
                    q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg/s to rad/s
                    tau_task = - Mq @ (self.Kv @ q_dot[:,np.newaxis]) - self.J.T @ (Mx @ self.control_input)
                else:
                    tau_task = - self.J.T @ (Mx @ self.control_input)

                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # estimate frictional torque in Nm
                self.calc_friction_torque()
            
                # Compute desired torque
                tau_d = G_torque[:, np.newaxis] + tau_task + self.tau_f[:, np.newaxis]

                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d, self.tau_J_d)

                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()    # target motor torque [Nm]
                writedata.time = 0.0
                self.torque_publisher.publish(writedata)

                self.tau_J_d = tau_d.copy()

                # store actual data for plotting
                self.store_data()

                rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()

    def save(self, name='task_performed'):
        curr_dir=os.getcwd()
        np.savez(curr_dir + '/data/' + str(name) + '.npz',
                traj=self.record_trajectory,
                ori=self.record_orientation,
                motor_torque=self.record_motor_torque,
                joint_torque=self.record_joint_torque,
                )


if __name__ == "__main__":
    # move to initial position first
    p1= posj(0,25,110,0,45,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
    movej(p1, vel=40, acc=20)

    time.sleep(1.0)

    try:
        # Initialize ROS node first
        rospy.init_node('My_service_node')
        
        # Create control object
        task = OSC()
        rospy.sleep(2.0)  # Give time for initialization

        # Start controller in a separate thread
        controller_thread = Thread(target=task.run_controller) 
        controller_thread.daemon = True
        controller_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.01)  # This keeps the plot window responsive

    except rospy.ROSInterruptException:
        pass

    finally:
        task.save()
        # plt.close('all')  # Clean up plots on exit
