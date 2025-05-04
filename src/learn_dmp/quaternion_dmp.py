#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import rk4_step
from .orientation_dmp import OrientationDMP


# DMP Explained : https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics/
class QuaternionDMP(OrientationDMP):
    def __init__(self, no_of_basis_func, dt=0.01, T=1, q_0=None, alpha=3, K=1050, D=None, W=None):
        super().__init__(no_of_basis_func, dt=dt, T=T, q_0=q_0, alpha=alpha, K=K, D=D, W=W)

    ####################################################################################################################################################

    def learn_dynamics_1(self, q_des, omega_des):
        """
        Takes in a desired trajectory and generates the set of system parameters that best realize this path.
        X_des: the desired trajectories of each DMP should be shaped [no_of_dmps, num_timesteps]
        """
        # set initial and goal state
        self.q_0 = q_des[:,[0]].copy()  # [[x0],[y0]]
        self.q_goal = q_des[:,[-1]].copy()  # [[xn],[yn]]

        t_des = np.linspace(0, self.cs.run_time, q_des.shape[1])  # demo trajectory timing
        std_time = np.linspace(0, self.cs.run_time, self.cs.time_steps)  # map demo timing to standard time

        # --------------------------------  Using Vector ---------------------------------------
        path_gen = scipy.interpolate.interp1d(t_des, q_des, kind="quadratic")

        # Evaluation of the interpolant
        q_des = path_gen(std_time)  # (4, N)

        # --------------------------------  Using Vector ---------------------------------------
        vel_gen = scipy.interpolate.interp1d(t_des, omega_des, kind="quadratic")

        # Evaluation of the interpolant
        eta_des = vel_gen(std_time)   # η = omega -> (3, N)
        # --------------------------------------------------------------------------------------

        # calculate acceleration of y_des (gradient of dX_des is computed using second order accurate central differences)
        eta_dot_des = np.gradient(eta_des, self.cs.dt, axis=1, edge_order=2)  # (3, N)
       
        theta_track = self.cs.rollout()
    
        ## Find the force required to move along this trajectory
        f_target = np.zeros([self.cs.time_steps, 3])
        log_q_des = np.zeros((3, q_des.shape[1]))   # (3, N)
        
        for i in range(f_target.shape[0]):
            q_conj = self.quaternion_conjugate(q_des[:,i])  # shape (4,)
            q_product = self.quaternion_multiply(self.q_goal, q_conj)  # shape (4,)
            log_q_des[:,i] = self.quaternion_logarithm(q_product)  # shape (3,)
        
        for idx in range(3):
            f_target[:,idx] = eta_dot_des[idx,:] - 2 * self.K * log_q_des[idx,:] + self.D * eta_des[idx,:]  # (101,2)
        
        # generate weights to realize f_target
        self.generate_weights(f_target, theta_track, log_q_des)
        self.reset_state()
    
    def step_1(self, q_goal, gamma, tau=1):
        """
        ----------------------- Run the DMP system for a single timestep ---------------------------
        Based on eqn (43) from the paper, Dynamic Movement Primitives in Robotics: A Tutorial Survey
        τ*dη = 2*K*log(q_goal x q_conj) - D*η + f(θ),   # ω = 2*log(q_goal x q_conj)
        τ*dq = 0.5*(S(η)*q)

        In matrix form; q_product = q_goal x q_conj
            |dη_x|       |K 0 0|   log(q_product)_x   |D 0 0|   |η_x|   |f_x|
        τ * |dη_y| = 2 * |0 K 0| * log(q_product)_y - |0 D 0| * |η_y| + |f_y|       
            |dη_z|       |0 0 K|   log(q_product)_z   |0 0 D|   |η_z|   |f_z|      

            |dq_x|         |0  -ηx -ηy -ηz|   |q_x|
        τ * |dq_y| = 0.5 * |ηx  0   ηz -ηy| * |q_y|
            |dq_z|         |ηy -ηz  0   ηx|   |q_z|
            |dq_w|         |ηz  ηy -ηx   0|   |q_w|
        """
        K_matrix = self.K * np.eye(3)
        D_matrix = self.D * np.eye(3)

        psi = self.gaussian_basis_func(self.cs.theta)  # update basis function
        
        # update forcing term using weights learnt while imitating given trajectory
        sum_psi = np.sum(psi[:,[0]])
        if np.abs(sum_psi) <= 1e-6:  # avoid division by 0
            f = 0.0 * np.dot(self.W, psi[:,[0]])
        else:
            f = (np.dot(self.W, psi[:,[0]]) / sum_psi) * self.cs.theta

        q_conj = self.quaternion_conjugate(self.q)
        q_product = self.quaternion_multiply(q_goal, q_conj)
        log_q = self.quaternion_logarithm(q_product)
        
        η = tau * self.omega
        dη = 2 * K_matrix @ log_q[:, np.newaxis] - D_matrix @ η + f
        
        self.omeaga_dot = dη / tau
        # self.omega = self.omega + self.omeaga_dot * (gamma*self.cs.dt)
        dq = (1/tau) * self.q_dot(self.omega, self.q)
        
        self.q = self.q + dq * (gamma*self.cs.dt)
        self.omega = self.omega + self.omeaga_dot * (gamma*self.cs.dt)
        
        self.cs.step(tau=tau)  # update theta
        return self.q, self.omega
    
    ####################################################################################################################################################
    
