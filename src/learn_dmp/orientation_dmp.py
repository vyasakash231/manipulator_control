#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from .canonical_system import Canonical_System


# DMP Explained : https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics/
class OrientationDMP:
    def __init__(self, no_of_DMPs, no_of_basis_func, dt=0.01, T=1, q_0=None, q_goal=None, alpha=3, K=1050, D=None, W=None):
        """
        no_of_DMPs         : number of dynamic movement primitives (i.e. dimensions)
        no_of_basis_func   : number of basis functions per DMP (actually, they will be one more)
        dt             : timestep for simulation
        q_0            : initial quaternion state of DMPs
        q_goal         : goal quaternion state of DMPs
        T              : final time
        K              : elastic parameter in the dynamical system
        D              : damping parameter in the dynamical system
        w              : associated weights
        alpha          : constant of the Canonical System
        """
        self.no_of_DMPs = no_of_DMPs
        self.no_of_basis_func = no_of_basis_func

        # Set up the DMP system
        if q_0 is None:
            q_0 = np.zeros(4)
        self.q_0 = copy.deepcopy(q_0)

        if q_goal is None:
            q_goal = np.ones(4)
        self.q_goal = copy.deepcopy(q_goal)

        self.K = K  # stiffness
        if D is None:
            self.D = 2 * np.sqrt(self.K)  # damping 
        else:
            self.D = D

        self.cs = Canonical_System(dt=dt, alpha=alpha, run_time=T)  # setup a canonical system
        
        self.reset_state()  # set up the DMP system

        self.center_of_gaussian()  # centers of Gaussian basis functions distributed along the phase of the movement
        self.variance_of_gaussian()  # width/variance of Gaussian basis functions distributed along the phase of the movement
        
        # If no weights are give, set them to zero (default, f = 0)
        if W is None:  
            W = np.zeros((3, self.no_of_basis_func))
        self.W = W

    def center_of_gaussian(self):
        self.c = np.exp(-self.cs.alpha * np.linspace(0, self.cs.run_time, self.no_of_basis_func + 1))  #  centers are exponentially spaced

    def variance_of_gaussian(self):
        """width/variance of gaussian distribution"""
        self.width = np.zeros(self.no_of_basis_func)
        for i in range(self.no_of_basis_func):
            self.width[i] = 1 / ((self.c[i+1] - self.c[i])**2)
        self.width = np.append(self.width, self.width[-1])                                                                                              

    def reset_state(self):
        """Reset the system state"""
        self.q = self.q_0.copy()
        self.dq = np.zeros((4, 1))
        self.ddq = np.zeros((4, 1))
        self.cs.reset()

    def gaussian_basis_func(self, theta):
        """Generates the activity of the basis functions for a given canonical system rollout"""
        c = np.reshape(self.c, [self.no_of_basis_func + 1, 1])
        h = np.reshape(self.width, [self.no_of_basis_func + 1, 1])
        Psi_basis_func = np.exp(-h * (theta - c)**2)
        return Psi_basis_func
    
    """from, eqn (10) of Adaptation of manipulation skills in physical contact with the environment to reference force profiles"""
    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions
        q = [x, y, z, w] = (u + v)
        where, u = [x, y, z], v = [w]
        """
        u1, v1 = q1[:3], q1[-1]
        u2, v2 = q2[:3], q2[-1]

        # quaternion product => q1 * q2 = (v1 + u1) * (v2 + u2)
        v = v1*v2 - np.dot(u1, u2)
        u = v1*u2 + v2*u1 + np.cross(u1, u2)
        return np.append(u, v)
    
    """from, eqn (11) of Adaptation of manipulation skills in physical contact with the environment to reference force profiles"""
    def quaternion_logarithm(self, q):
        """ 
        log(q) : S^3 → R^3 
        q = [x, y, z, w] = (u + v)
        where, u = [x, y, z], v = [w]
        """
        u, v = q[:3], q[-1]
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:  # Almost zero
            return np.array([0, 0, 0])
        
        u_unit = u / u_norm
        angle = np.arccos(np.clip(v, -1.0, 1.0))
        return angle * u_unit
    
    def quaternion_conjugate(self, q):
        u, v = q[:3], q[-1]
        return np.append(-u, v)

    """from, eqn (15) of Adaptation of manipulation skills in physical contact with the environment to reference force profiles"""
    def generate_weights(self, f_target, theta_track, log_q):
        """
        Generate a set of weights over the basis functions such that the target forcing 
        term trajectory is matched (f_target - f(θ), shape -> [3 x time_steps])
                    / ∑ W * ψ(θ) \              / W.T @ ψ(θ) \       
        f(θ) = D @ |--------------| * θ => D @ |--------------| * θ  
                    \   ∑ ψ(θ)   /              \   ∑ ψ(θ)   /       
        
               /  ψ(θ)  \     
        W.T @ |----------| * θ => D^(-1) @ f(θ) = A
               \ ∑ ψ(θ) /            
                                 
                  | /  ψ(θ)  \     |^(-1)
        W = A.T @ ||----------| * θ|
                  | \ ∑ ψ(θ) /     |
        """
        # generate Basis functions
        psi = self.gaussian_basis_func(theta_track)

        # scaling factor
        A = np.zeros_like(f_target)
        for i in range(f_target.shape[1]):
            D = np.diag(log_q[:,i])  # shape (3,3)
            A[:,[i]] = np.linalg.inv(D) @ f_target[:,[i]]

        # calculate basis function weights using "linear regression"
        sum_psi = np.sum(psi,0)
        self.W = np.nan_to_num(A.T @ np.linalg.pinv((psi / sum_psi) * theta_track))
