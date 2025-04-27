#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot, Filters
from scipy.spatial.transform import Rotation

class CTC(Robot):
    def __init__(self):
        self.is_rt_connected = False
        self.shutdown_flag = False  # Add flag to track shutdown state

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        # self.plotter.setup_plots_2()

        super().__init__()

    def plot_data(self):
        try:
            # self.plotter.update_imdepdance(self.data.actual_motor_torque, 
            #                                self.data.raw_force_torque, 
            #                                self.data.actual_joint_torque, 
            #                                self.impedance_force.reshape(-1))
            pass
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def saturate_torque(self, tau, tau_J_d):
        """
        Limit both the torque rate of change and peak torque values for Doosan A0509 robot
        """
        # First limit rate of change as in your original function
        # tau_rate_limited = np.zeros(self.n)
        # for i in range(len(tau)):
        #     difference = tau[i] - tau_J_d[i]
        #     tau_rate_limited[i] = tau_J_d[i] + np.clip(difference, -self.delta_tau_max, self.delta_tau_max)
        # tau = tau_rate_limited.copy()
        
        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.95
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0]) # Nm

        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # Clip torque values to stay within limits (both positive and negative)
        tau = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau

    def run_controller(self, Kp, Kd, qd, qd_dot, qd_ddot):
        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate
        i = 0
        
        total_points = qd.shape[1]
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag and i < total_points:            
                # Store current desired position for plotting
                self.current_desired_position = qd[:,i]

                G_torque = self.Robot_RT_State.gravity_torque
                C = self.Robot_RT_State.coriolis_matrix
                M = self.Robot_RT_State.mass_matrix

                q =  0.0174532925 * self.Robot_RT_State.actual_joint_position_abs    # convert from deg to rad
                q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg/s to rad/s

                # Calculate errors
                E = qd[:,i] - q
                E_dot = qd_dot[:,i] - q_dot

                # Print progress
                if i % 500 == 0:
                    print(f"Progress: {i}/{total_points} points ({(i/total_points)*100:.1f}%)")
                    # print(f"Position error (deg): {E}")
                    print(f"Max error: {np.max(np.abs(E))}")
                    print("---------------------------------------------------------------------")

                # Feed-back PD-control Input with reduced gains
                u = Kp @ E[:, np.newaxis] + Kd @ E_dot[:, np.newaxis]

                # Compute control torque
                Torque = M @ (qd_ddot[:,[i]] + u) + C @ q_dot[:, np.newaxis] + G_torque[:, np.newaxis]
                
                # Saturate torque to avoid limit breach
                Torque = self.saturate_torque(Torque, self.tau_J_d)

                # Send torque command
                writedata = TorqueRTStream()
                writedata.tor = Torque
                writedata.time = 0.0
                self.torque_publisher.publish(writedata)

                self.tau_J_d = Torque.copy()

                i += 1

                rate.sleep()
                
            print(f"Control loop finished. Completed {i}/{total_points} points")
            
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()


def generate_quintic_trajectory(q0, qf, t0, tf, dt=0.005):
    """
    Generate a quintic polynomial trajectory between two points.
    
    Args:
        q0: Initial position
        qf: Final position
        t0: Initial time
        tf: Final time
        num_points: Number of points in the trajectory
    
    Returns:
        t: Time points
        q: Position trajectory
        qd: Velocity trajectory
        qdd: Acceleration trajectory
    """
    # Time vector
    t = np.arange(t0, tf, dt)
    
    # Time parameters
    T = tf - t0
    
    # Quintic polynomial coefficients
    a0 = q0
    a1 = 0  # Initial velocity = 0
    a2 = 0  # Initial acceleration = 0
    a3 = 10 * (qf - q0) / T**3
    a4 = -15 * (qf - q0) / T**4
    a5 = 6 * (qf - q0) / T**5
    
    # Compute position, velocity, and acceleration
    q = a0 + a1*(t-t0) + a2*(t-t0)**2 + a3*(t-t0)**3 + a4*(t-t0)**4 + a5*(t-t0)**5
    qd = a1 + 2*a2*(t-t0) + 3*a3*(t-t0)**2 + 4*a4*(t-t0)**3 + 5*a5*(t-t0)**4
    qdd = 2*a2 + 6*a3*(t-t0) + 12*a4*(t-t0)**2 + 20*a5*(t-t0)**3
    return t, q, qd, qdd

def pre_process_trajectory(tf, dt):
    # Define initial and final joint angles (in degrees)
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qf = np.array([90.0, 45.0, -45.0, 45.0, -45.0, 180.0])

    # Increased time for slower motion
    t0 = 0.0
    t = np.arange(t0, tf, dt)
    q_list = np.zeros((6, len(t)))
    q_dot_list = np.zeros((6, len(t)))
    q_ddot_list = np.zeros((6, len(t)))

    for i in range(6):
        _, q, qd, qdd = generate_quintic_trajectory(q0[i], qf[i], t0, tf, dt)
        q_list[i, :len(q)] = q
        q_dot_list[i, :len(qd)] = qd
        q_ddot_list[i, :len(qdd)] = qdd
    return t, q_list, q_dot_list, q_ddot_list

if __name__ == "__main__":
    try:
        # Initialize ROS node first
        rospy.init_node('My_service_node')
        dt = 0.002
        t, qd, qd_dot, qd_ddot = pre_process_trajectory(tf=10.0, dt=dt)
        
        # Create control object
        task = CTC()
        rospy.sleep(2.0)  # Give time for initialization

        Kp = np.diag([2.5, 2.5, 3.0, 3.5, 30.0, 300.0]) 
        Kd = np.diag([0.5, 0.5, 0.5, 0.5, 2.0, 20.0])
        
        # Start impedance control in a separate thread
        control_thread = Thread(target=lambda: task.run_controller(Kp, Kd, qd, qd_dot, qd_ddot))
        control_thread.daemon = True
        control_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.01)  # This keeps the plot window responsive
            
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')  # Clean up plots on exit