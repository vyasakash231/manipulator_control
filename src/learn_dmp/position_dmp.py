#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import rk4_step, forward_euler
from .discrete_dmp import DiscreteDMP


class PositionDMP(DiscreteDMP):
    def __init__(self, no_of_DMPs, no_of_basis_func, dt=0.01, T=1, X_0=None, alpha=3, K=1050, D=None, W=None):
        super().__init__(no_of_DMPs, no_of_basis_func, dt=dt, T=T, X_0=X_0, alpha=alpha, K=K, D=D, W=W)

    ####################################################################################################################################################

    def learn_dynamics(self, X_des):
        """
        Takes in a desired trajectory and generates the set of system parameters that best realize this path.
        X_des: the desired trajectories of each DMP should be shaped [no_of_dmps, num_timesteps]
        """
        # set initial and goal state
        self.X_0 = X_des[:,[0]].copy()  # [[x0],[y0]]
        self.X_g = X_des[:,[-1]].copy()  # [[xn],[yn]]

        t_des = np.linspace(0, self.cs.run_time, X_des.shape[1])  # demo trajectory timing

        # -------------------- plot uniform trajectory from demo data --------------------
        path = np.zeros((self.no_of_DMPs, self.cs.time_steps))
        for i in range(self.no_of_DMPs):
            path_gen = scipy.interpolate.interp1d(t_des, X_des[i,:], kind="quadratic")
            for j in range(self.cs.time_steps):
                path[i, j] = path_gen(j * self.cs.dt)  # map demo timing to standard time
        
        # Evaluation of the interpolant
        X_des = path # [[x0,x1,x2....], [y0,y1,y2,....]] -> (3,N)
        # --------------------------------------------------------------------------------
        
        # calculate acceleration of y_des (gradient of dX_des is computed using second order accurate central differences)
        dX_des = np.gradient(X_des, self.cs.dt, axis=1, edge_order=2)  # (3,N)
        dX_des[:,0] = np.zeros(3)  # make initial velocity 0
        
        # calculate acceleration of y_des (gradient of dX_des is computed using second order accurate central differences)
        ddX_des = np.gradient(dX_des, self.cs.dt, axis=1, edge_order=2)  # (3,N)
        ddX_des[:,0] = np.zeros(3)  # make initial acceleration 0

        theta_track = self.cs.rollout()
    
        ## Find the force required to move along this trajectory
        f_target = np.zeros([self.cs.time_steps, self.no_of_DMPs])
        for idx in range(self.no_of_DMPs):
            f_target[:,idx] = (ddX_des[idx,:] / self.K) - (self.X_g[idx] - X_des[idx,:]) + (self.D / self.K) * dX_des[idx,:] + (self.X_g[idx] - self.X_0[idx]) * theta_track  # (101,2)
        
        # generate weights to realize f_target
        self.generate_weights(f_target, theta_track)
        self.reset_state()

    def rollout(self, X_d, gamma=1):
        """Generate a system trial, no feedback is incorporated."""
        self.reset_state()

        # set up tracking vectors
        y_track = np.zeros((self.no_of_DMPs, self.cs.time_steps))
        dy_track = np.zeros((self.no_of_DMPs, self.cs.time_steps))
      
        for t in range(self.cs.time_steps):
            y_track[:,[t]], dy_track[:,[t]] = self.step(X_d, gamma)   # run and record timestep
        return y_track, dy_track
    
    def step(self, X_g, gamma=1, tau=1):
        """
        Based on eqn (1) from the paper, Learning and Generalization of Motor Skills by Learning from Demonstration
        DMP 2nd order system in vector form (for 3 DOF system);
        τ*dV = K*(X_g - X) - D*V - K*(X_g - X_0)*θ + K*f
        τ*dX = V

        In matrix form;
            |dV_x|   |K 0 0|   |g_x - X_x|   |D 0 0|   |V_x|   |K 0 0|   |g_x - X_x0|       |K 0 0|   |f_x|
        τ * |dV_y| = |0 K 0| * |g_y - X_y| - |0 D 0| * |V_y| - |0 K 0| * |g_y - X_y0| * θ + |0 K 0| * |f_y|       
            |dV_z|   |0 0 K|   |g_z - X_z|   |0 0 D|   |V_z|   |0 0 K|   |g_z - X_z0|       |0 0 K|   |f_z|      

            |dX_x|   |V_x|
        τ * |dX_y| = |V_y|
            |dX_z|   |V_z|

        State-Space form for 3 DOF/No_of_DMPs system;
        state_vector, Y = [y1, y2, y3, y4, y5, y6] = [V_x, X_x, V_y, X_y, V_z, X_z]
        
                |dy1|   |dV_x|         |-D -K  0  0  0  0|   |V_x|         |g_x - (g_x - X_x0)*θ + f_x|
                |dy2|   |dX_x|         | 1  0  0  0  0  0|   |X_x|         |           0              |
        dY_dt = |dy3| = |dV_y| = 1\τ * | 0  0 -D -K  0  0| * |V_y| + K\τ * |g_y - (g_y - X_y0)*θ + f_y| 
                |dy4|   |dX_y|         | 0  0  1  0  0  0|   |X_y|         |           0              |
                |dy5|   |dV_z|         | 0  0  0  0 -D -K|   |V_z|         |g_z - (g_z - X_z0)*θ + f_z|
                |dy6|   |dX_z|         | 0  0  0  0  1  0|   |X_z|         |           0              |

        dY_dt = A @ Y + B   (A-matrix must have constant coeff for DS to be linear)
        """

        # define state vector (Y)
        Y = np.zeros((2*self.no_of_DMPs, 1))  # Y = [[0], [0], [0], [0]]
        Y[range(0,2*self.no_of_DMPs, 2),:] = copy.deepcopy(self.dX)  # [[Vx], [0], [Vy], [0]]
        Y[range(1,2*self.no_of_DMPs, 2),:] = copy.deepcopy(self.X)  # [[Vx], [x], [Vy], [y]]
        
        # define A-matrix
        A = np.zeros((2*self.no_of_DMPs, 2*self.no_of_DMPs))
        A[range(0, 2*self.no_of_DMPs, 2), range(0, 2*self.no_of_DMPs, 2)] = -self.D / tau
        A[range(0, 2*self.no_of_DMPs, 2), range(1, 2*self.no_of_DMPs, 2)] = -self.K / tau
        A[range(1, 2*self.no_of_DMPs, 2), range(0, 2*self.no_of_DMPs, 2)] = 1 / tau

        """Run the DMP system for a single timestep"""
        psi = self.gaussian_basis_func(self.cs.theta)  # update basis function
        
        # update forcing term using weights learnt while imitating given trajectory
        sum_psi = np.sum(psi[:,[0]])
        if np.abs(sum_psi) <= 1e-6:  # avoid division by 0
            f = 0.0 * np.dot(self.W, psi[:,[0]])
        else:
            f = (np.dot(self.W, psi[:,[0]]) / sum_psi) * self.cs.theta

        # define B-matrix
        B = np.zeros((2 * self.no_of_DMPs ,1))
        B[0::2,:] = (self.K/tau) * (X_g - (X_g - self.X_0) * self.cs.theta + f)

        # solve above dynamical system using Euler-forward method / Runge-kutta 4th order / Exponential Integrators 
        dY_dt = rk4_step(Y,A,B,gamma*self.cs.dt)
        # dY_dt = forward_euler(Y,A,B,self.cs.dt)

        Y = Y + dY_dt * (gamma*self.cs.dt)
        
        # extract position-X, velocity-V, acceleration data from current state vector-Y values
        self.dX = Y[0::2, :] # extract velocity data from state vector Y
        self.X = Y[1::2, :]   # extract position data from state vector Y
        
        self.cs.step(tau=tau)  # update theta
        return self.X, self.dX   
    
