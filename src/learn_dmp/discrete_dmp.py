#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from .canonical_system import CanonicalSystem


# DMP Explained : https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics/
class DiscreteDMP:
    def __init__(self, no_of_DMPs, no_of_basis_func, dt=0.01, T=1, X_0=None, alpha=3, K=1050, D=None, W=None):
        """
        no_of_DMPs         : number of dynamic movement primitives (i.e. dimensions)
        no_of_basis_func   : number of basis functions per DMP (actually, they will be one more)
        dt             : timestep for simulation
        X_0            : initial state of DMPs
        X_g            : X_g state of DMPs
        T              : final time
        K              : elastic parameter in the dynamical system
        D              : damping parameter in the dynamical system
        w              : associated weights
        alpha          : constant of the Canonical System
        """
        self.no_of_DMPs = no_of_DMPs
        self.no_of_basis_func = no_of_basis_func

        # Set up the DMP system
        if X_0 is None:
            X_0 = np.zeros(self.no_of_DMPs)
        self.X_0 = copy.deepcopy(X_0)

        self.K = K  # stiffness
        if D is None:
            self.D = 2 * np.sqrt(self.K)  # damping 
        else:
            self.D = D

        self.cs = CanonicalSystem(dt=dt, alpha=alpha)  # setup a canonical system

        self.reset_state()  # set up the DMP system

        self.center_of_gaussian()  # centers of Gaussian basis functions distributed along the phase of the movement
        self.variance_of_gaussian()  # width/variance of Gaussian basis functions distributed along the phase of the movement
        
        # If no weights are give, set them to zero (default, f = 0)
        if W is None:  
            W = np.zeros((self.no_of_DMPs, self.no_of_basis_func))
        self.W = W

    def center_of_gaussian(self):
        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.no_of_basis_func + 1)
        self.c = np.exp(-self.cs.alpha * des_c)  #  centers are exponentially spaced

    def variance_of_gaussian(self):
        """width/variance of gaussian distribution"""
        self.width = np.zeros(self.no_of_basis_func)
        for i in range(self.no_of_basis_func):
            self.width[i] = 1 / ((self.c[i+1] - self.c[i])**2)
        self.width = np.append(self.width, self.width[-1])                                                                                              

    def reset_state(self):
        """Reset the system state"""
        self.X = self.X_0.copy()
        self.dX = np.zeros((self.no_of_DMPs, 1))
        self.ddX = np.zeros((self.no_of_DMPs, 1))
        self.cs.reset()

    def gaussian_basis_func(self, theta):
        """Generates the activity of the basis functions for a given canonical system rollout"""
        c = np.reshape(self.c, [self.no_of_basis_func + 1, 1])
        h = np.reshape(self.width, [self.no_of_basis_func + 1, 1])
        Psi_basis_func = np.exp(-h * (theta - c)**2)
        return Psi_basis_func

    def generate_weights(self, f_target, theta_track):
        """
        Generate a set of weights over the basis functions such that the target forcing 
        term trajectory is matched (f_target - f(θ), shape -> [no_of_DMPs x time_steps])
                / ∑ W * ψ(θ) \          / W.T @ ψ(θ) \                /  ψ(θ)  \       
        f(θ) = |--------------| * θ => |--------------| * θ => W.T @ |----------| * θ  
                \   ∑ ψ(θ)   /          \   ∑ ψ(θ)   /                \ ∑ ψ(θ) /       
        
                     | /  ψ(θ)  \     |^(-1)
        W = f(θ).T @ ||----------| * θ|
                     | \ ∑ ψ(θ) /     |
        """
        # generate Basis functions
        psi = self.gaussian_basis_func(theta_track)
        
        # calculate basis function weights using "linear regression"
        sum_psi = np.sum(psi,0)
        self.W = np.nan_to_num(f_target.T @ np.linalg.pinv((psi / sum_psi) * theta_track))

        for i in range(self.no_of_DMPs):
            for j in range(self.no_of_basis_func):
                pass
