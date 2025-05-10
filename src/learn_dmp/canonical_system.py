import numpy as np

class CanonicalSystem:
    """
    τ * (dθ/dt) = - alpha * θ
    
    Intergrate both sides;
    ∫(dθ/θ) = -(alpha/τ) * ∫dt
    ln(θ2) - ln(θ1) = -(alpha/τ) * (t2 - t1)

    ln(θ2/θ1) = -(alpha/τ) * (t2 - t1)
    θ2/θ1 = exp(-(alpha/τ) * dt)
    θ2 = exp(-(alpha/τ) * dt) * θ1
    """
    def __init__(self, dt, alpha, run_time = 1):
        self.dt = dt
        self.alpha = alpha
        self.run_time = run_time  # T
        self.time_steps = int(self.run_time/self.dt)  # T/dt = 1/0.005 = 200 time steps

        self.reset()

    def reset(self):
        """Reset the system state"""
        self.theta = 1  # at t = 0, theta = 1

    def step(self,tau=1):
        """Perform single step integration"""
        self.theta = np.exp(-(self.alpha/tau)*self.dt) * self.theta

    def rollout(self,tau=1):
        self.theta_track = np.zeros(self.time_steps)
        self.reset()
        for i in range(self.time_steps):
            self.theta_track[i] = self.theta
            self.step(tau)
        return self.theta_track

    