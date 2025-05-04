#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot
from scipy.spatial.transform import Rotation
import scipy.linalg as LA

class PTC(Robot):
    """from the paper -> Probabilistic Learning of Torque Controllers from Kinematic and Force Constraints"""
    def __init__(self, file_name='demo'):
        self.shutdown_flag = False  
        self.file_name = file_name

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        
        # Setup initial Gain Values (will be adapted based on uncertainty)
        self.translational_gain_mean = 300.0
        self.rotational_gain_mean = 350.0

        # Initialize mean gain matrices (similar to your OSC implementation)
        self.Kp = np.eye(6)
        self.Kp[:3, :3] *= self.translational_gain_mean
        self.Kp[3:, 3:] *= self.rotational_gain_mean
        
        self.Kv = np.eye(6)
        self.Kv[:3, :3] *= 1.25 * np.sqrt(self.translational_gain_mean)
        self.Kv[3:, 3:] *= 0.5 * np.sqrt(self.rotational_gain_mean)

        # Frictional compensation observer gain
        self.Ko = np.diag([0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

        # Current robot state variables
        self.tau_J_d = np.zeros(6)  # Previous desired torque
        
        # Maximum allowed torque rate change
        self.delta_tau_max = 1.0
        
        # Initial estimated frictional torque
        self.tau_f = np.zeros(6)  # Using self.n caused issues as it's not defined yet

        # Define covariances for probabilistic control
        # Covariance for demonstration data (prior)
        self.Sigma_q = np.eye(6) * 0.01  # Joint position covariance
        self.Sigma_q_dot = np.eye(6) * 0.05  # Joint velocity covariance
        
        # Combined state covariance (q and q_dot)
        self.Sigma_state = np.block([[self.Sigma_q, np.zeros((6, 6))],
                                    [np.zeros((6, 6)), self.Sigma_q_dot]])
        
        # Initialize importance matrix (Gamma) as identity initially
        self.Gamma = np.eye(12)  # For combined state (q, q_dot)
        
        super().__init__()

    def load_demo(self, name='demo'):
        curr_dir = os.getcwd()
        data = np.load(curr_dir + '/data/' + str(name) + '.npz')
        self.q_demo = data['q']    # shape: (N, 6), in rad
        self.q_dot_demo = data['q_dot']    # shape: (N, 6), in rad/s
        self.N = self.q_demo.shape[0]   # no of sample points

        # If covariance data is available, load it
        if 'q_cov' in data:
            self.q_cov_demo = data['q_cov']  # shape: (N, 6, 6)
            self.q_dot_cov_demo = data['q_dot_cov']  # shape: (N, 6, 6)
        else:
            # Initialize with default covariances
            self.q_cov_demo = np.tile(self.Sigma_q, (self.N, 1, 1))
            self.q_dot_cov_demo = np.tile(self.Sigma_q_dot, (self.N, 1, 1))

    def store_data(self):
        pos = 0.001 * self.Robot_RT_State.actual_tcp_position[:3]    # in m
        orient = self._eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # quaternions
        
        # Store position and orientation
        if not hasattr(self, 'record_trajectory'):
            self.record_trajectory = pos.reshape(1, -1)
            self.record_orientation = orient.reshape(1, -1)
            self.record_motor_torque = self.Robot_RT_State.actual_motor_torque.reshape(1, -1)
            self.record_joint_torque = self.Robot_RT_State.actual_joint_torque.reshape(1, -1)
        else:
            self.record_trajectory = np.vstack((self.record_trajectory, pos))  # shape: (N, 3) 
            self.record_orientation = np.vstack((self.record_orientation, orient))  # shape: (N, 4)
            self.record_motor_torque = np.vstack((self.record_motor_torque, self.Robot_RT_State.actual_motor_torque))
            self.record_joint_torque = np.vstack((self.record_joint_torque, self.Robot_RT_State.actual_joint_torque))
        
    def start(self):
        # load demo trajectory
        self.load_demo(name=self.file_name)
        rospy.loginfo("ProbabilisticTorqueController: Controller started")
    
    @property
    def position_error(self):
        # actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        current_position = self.Robot_RT_State.actual_tcp_position[:3]   #  (x, y, z) in mm
        return 0.001 * (current_position - self.position_des)  # convert from mm to m
    
    def saturate_torque(self, tau, tau_J_d):
        """Limit both the torque rate of change and peak torque values for Doosan A0509 robot"""
        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # Apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.95
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0])  # Nm

        # Clip torque values to stay within limits (both positive and negative)
        tau_saturated = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau_saturated

    def calc_friction_torque(self):
        """Estimate friction torque using a disturbance observer approach"""
        motor_torque = self.Robot_RT_State.actual_motor_torque   # in Nm
        joint_torque = self.Robot_RT_State.actual_joint_torque   # in Nm

        term_1 = np.dot(self.Ko, (motor_torque - joint_torque - self.tau_f)) * 0.005
        self.tau_f = self.tau_f + term_1

    @property
    def Ax(self):
        return np.block([[self.Kp, self.Kv]])
    
    @property
    def bx(self):
        current_q = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg to rad
        current_q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg/s to rad/s
        current_state = np.concatenate((current_q, current_q_dot))[:, np.newaxis]
        return self.Ax @ current_state
    
    # @property
    # def mu_state(self):  # this should come from some learning algorithm (GMM, GPR), as given in the paper
    #     return np.concatenate((self.q_des, self.q_dot_des))[:, np.newaxis]

    def update_importance_matrix(self):
        """Update the importance matrix (Gamma) based on equation (13)"""
        # Get the current covariance of the reference trajectory
        idx = min(self.current_idx, self.N - 1)
        
        # Construct the full state covariance matrix at the current time
        Sigma_full = np.block([[self.q_cov_demo[idx], np.zeros((6, 6))],
                              [np.zeros((6, 6)), self.q_dot_cov_demo[idx]]])
        
        # Compute Gamma using equation (13): Gamma = Sigma^(-1)
        # Add small regularization term to ensure invertibility
        reg_term = 1e-6 * np.eye(Sigma_full.shape[0])
        self.Gamma = LA.inv(Sigma_full + reg_term)
        return self.Gamma
    
    def compute_ptc_control(self):
        """Compute the probabilistic torque control based on equations (9) and (12)"""
        # Update importance matrix based on current covariance
        # self.update_importance_matrix()
        
        # Compute the mean torque command using equation (9):
        # tau_q = A_q * mu_q + b_q
        self.tau_ptc = self.Ax @ self.mu_state + self.bx
        
        # Compute the covariance of the torque command using equation (9): Sigma_tau = A_q * Sigma_q * A_q^T
        # Use the current state covariance from the demonstration
        idx = min(self.current_idx, self.N - 1)
        Sigma_full = np.block([[self.q_cov_demo[idx], np.zeros((6, 6))],
                              [np.zeros((6, 6)), self.q_dot_cov_demo[idx]]])
        
        # Apply importance weighting (equation 13)
        # Sigma_weighted = Gamma^(-1) = Sigma
        # We've already calculated self.Gamma = Sigma^(-1)
        
        # Compute the torque covariance
        self.Sigma_tau = self.Ax @ Sigma_full @ self.Ax.T
        
        # Adapt control gains based on uncertainty
        # Lower gain in directions with high uncertainty
        for i in range(6):
            uncertainty_factor = np.sqrt(self.Sigma_tau[i, i])
            scaling = 1.0 / (1.0 + 0.1 * uncertainty_factor)  # Apply a scaling that reduces gain for high uncertainty
            scaling = max(0.5, scaling)  # Limit the minimum scaling to 0.5 to ensure some control action
            self.tau_ptc[i] *= scaling
        
        # For debugging - print variance of torque commands
        # if self.current_idx % 100 == 0:
        #     print(f"Torque variances: {np.diag(self.Sigma_tau)}")
        
        return self.tau_ptc
    
    def run_controller(self):
        self.start()
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
                self.current_idx = int(elapsed_time / traj_dt)
                
                # Ensure index is within bounds
                if self.current_idx >= self.N:
                    self.current_idx = self.N - 1
                    if self.current_idx == self.N - 1 and np.linalg.norm(self.position_error)*1000 < 10.0:
                        rospy.loginfo("Trajectory complete and position error < 10mm")
                        break

                # Update desired position and orientation from demonstration
                self.q_des = self.q_demo[self.current_idx,:]  # (x, y, z) in rad

                # Get desired velocity from recorded data
                self.q_dot_des = self.q_dot_demo[self.current_idx,:]  # Already in rad/s

                # Update the Jacobian
                self.J = self.Robot_RT_State.jacobian_matrix

                # Compute the probabilistic torque control action
                tau_task = self.compute_ptc_control()

                # Compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque[:, np.newaxis]

                # Estimate frictional torque in Nm
                self.calc_friction_torque()
            
                # Compute desired torque
                tau_d = G_torque + tau_task + self.tau_f[:, np.newaxis]

                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d, self.tau_J_d)

                # Publish to ROS
                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()    # target motor torque [Nm]
                writedata.time = 0.0
                self.torque_publisher.publish(writedata)

                self.tau_J_d = tau_d.copy()

                # Store data for analysis
                self.store_data()

                rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()


if __name__ == "__main__":
    # Move to initial position first
    p1 = posj(0, 25, 110, 0, 45, 0)  # posj(q1, q2, q3, q4, q5, q6) Joint angles in degrees
    movej(p1, vel=40, acc=20)

    time.sleep(1.0)

    try:
        # Initialize ROS node first
        rospy.init_node('My_service_node')
        
        # Create control object
        task = PTC()
        rospy.sleep(2.0)  # Give time for initialization

        # Start controller in a separate thread
        controller_thread = Thread(target=task.run_controller) 
        controller_thread.daemon = True
        controller_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.01)  # Keep plot window responsive

    except rospy.ROSInterruptException:
        pass

    finally:
        task.save()