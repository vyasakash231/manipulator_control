import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

class Robot:
    def __init__(self, position, goal, velocity=None):
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity, dtype=float)
        self.history = [self.position.copy()]
        
        # Control parameters
        self.speed = 0.5  # Max speed
        self.dt = 0.1     # Time step
        
    def set_velocity(self, velocity):
        self.velocity = np.array(velocity, dtype=float)
        
    def update_position(self):
        self.position += self.velocity * self.dt
        self.history.append(self.position.copy())
        
    def default_velocity(self):
        """Calculate velocity vector towards goal"""
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            return self.speed * direction / distance
        return np.zeros(3)

class Obstacle:
    def __init__(self, position, radius=0.5):
        self.center = np.array(position, dtype=float)
        self.radius = radius
        
    def distance_to(self, agent_position):
        """Calculate distance from obstacle to position"""
        return np.linalg.norm(self.center - agent_position) - self.radius

class ObstacleAvoidance:
    def __init__(self, robot, obstacles, influence_radius=2.0):
        self.robot = robot
        self.obstacles = obstacles
        self.influence_radius = influence_radius
        
        # Parameters for the manifold projection method
        self.lambda_prox = 1.0   # Proximity weighting parameter
        self.beta_damp = 0.5     # Overall damping strength
        
    def compute_avoidance_vectors(self):
        """Compute avoidance vectors for each obstacle"""
        v_ref = self.robot.default_velocity()
        if np.linalg.norm(v_ref) < 1e-6:
            return v_ref, []  # No motion needed
            
        v_ref_norm = v_ref / np.linalg.norm(v_ref)
        avoidance_vectors = []
        
        for obstacle in self.obstacles:
            # Calculate vector to obstacle
            to_obstacle = obstacle.position - self.robot.position
            distance = np.linalg.norm(to_obstacle) - obstacle.radius
            
            if distance > self.influence_radius:
                continue  # Obstacle too far to influence
                
            # Normalize the obstacle direction
            if np.linalg.norm(to_obstacle) > 0:
                to_obstacle_norm = to_obstacle / np.linalg.norm(to_obstacle)
            else:
                # Obstacle at same position as robot (rare but handle it)
                to_obstacle_norm = np.array([0, 0, 1])  # Arbitrary direction
                
            # Create avoidance direction (away from obstacle)
            avoid_dir = -to_obstacle_norm
            
            # Store avoidance vector and distance
            avoidance_vectors.append((avoid_dir, distance))
            
        return v_ref_norm, avoidance_vectors
        
    def manifold_projection(self):
        """Implement the manifold-based velocity projection for obstacle avoidance"""
        v_ref, avoidance_vectors = self.compute_avoidance_vectors()
        
        if not avoidance_vectors:
            return self.robot.default_velocity()  # No obstacles to avoid
            
        # Process each avoidance vector
        w_combined = np.zeros(3)
        gamma_sum = 0
        gamma_list = []
        w_i = []
        
        for i, (v_i, distance) in enumerate(avoidance_vectors):
            # Compute the angle
            cos_theta = np.clip(np.dot(v_ref, v_i), -1.0, 1.0)
            theta_i = np.arccos(cos_theta)
            
            # Compute the orthogonal component
            n_i = v_i - (np.dot(v_ref, v_i) * v_ref)
            n_i_norm = np.linalg.norm(n_i)
            
            if n_i_norm > 1e-6:
                u_i = n_i / n_i_norm
            else:
                # If n_i is near zero, create a perpendicular vector
                u_i = self.perpendicular_vector(v_ref)
            
            # Compute proximity-based damping factor
            gamma_i = np.exp(-self.lambda_prox * distance) / (distance**2 + 0.05)  # Add 0.05 to avoid division by zero
            gamma_sum += gamma_i
            
            gamma_list.append(gamma_i)
            
            # Map onto tangent space with damping
            w_i.append(theta_i * u_i)
        
        for i in range(len(gamma_list)):
            w_combined += w_i[i] * gamma_list[i]
            
        # Compute overall velocity damping
        gamma_combined = min(1.0, 1.0 / (1.0 + self.beta_damp * gamma_sum))
        
        # Map back to the sphere with damping
        w_norm = np.linalg.norm(w_combined)
        
        if w_norm > 1e-6:
            v_new = np.cos(w_norm) * v_ref + np.sin(w_norm) * (w_combined / w_norm)
            v_final = gamma_combined * self.robot.speed * (v_new / np.linalg.norm(v_new))
            return v_final
        else:
            return gamma_combined * self.robot.default_velocity()

    def perpendicular_vector(self, v):
        """Create a vector perpendicular to v"""
        if abs(v[0]) < abs(v[1]):
            return np.array([0, -v[2], v[1]])
        else:
            return np.array([-v[2], 0, v[0]])
        
    def update(self):
        """Update robot velocity based on obstacle avoidance"""
        new_velocity = self.manifold_projection()
        self.robot.set_velocity(new_velocity)
        self.robot.update_position()

# Simulation setup
def create_simulation(dimensions=3):
    # Create robot and obstacles
    robot = Robot(position=[-4, -4, 0], goal=[4, 4, 0])
    
    obstacles = [
        Obstacle(position=[-1, 0, 0], radius=0.7),
        Obstacle(position=[0.8, 2, 0], radius=0.5),
        Obstacle(position=[0, -2, 0], radius=0.6),
        # Obstacle(position=[1.8, 0, 0], radius=0.6),
        # Obstacle(position=[1.5, 1.0, 0], radius=0.7),
        # Obstacle(position=[2.2, 2.0, 0], radius=0.6),
        # Obstacle(position=[0, -2, 0], radius=0.6)
    ]
    
    avoidance_system = ObstacleAvoidance(robot, obstacles)
    return robot, obstacles, avoidance_system, dimensions

class RobotAnimator:
    def __init__(self, dimensions=2):
        self.dimensions = dimensions
        self.robot, self.obstacles, self.avoidance_system, _ = create_simulation(dimensions)
        self.fig = None
        self.ax = None  # Main axis
        self.robot_plot = None
        self.path_plot = None
        self.goal_plot = None
        self.vel_arrow = None
        self.obstacle_circles = []
        self.anim = None
        
    def setup_animation(self, figsize=(10, 8)):
        """Set up the animation figure and axis"""
        self.fig = plt.figure(figsize=figsize)
        
        if self.dimensions == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._set_3d_plot_properties(self.ax)
        else:
            self.ax = self.fig.add_subplot(111)
            self._set_2d_plot_properties(self.ax)
            
        return self
    
    def _set_2d_plot_properties(self, ax):
        """Set the basic properties for a 2D plot"""
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Manifold-Based Obstacle Avoidance')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def _set_3d_plot_properties(self, ax, elevation=30, azimuth=-60):
        """Set properties for a 3D plot"""
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elevation, azimuth)
        ax.set_title('Robot Trajectory')
    
    def init_animation(self):
        """Initialize the animation"""
        if self.dimensions == 2:
            # Create initial 2D plots
            self.robot_plot, = self.ax.plot([], [], 'bo', markersize=10)
            self.path_plot, = self.ax.plot([], [], 'b-', alpha=0.5)
            self.goal_plot, = self.ax.plot([self.robot.goal[0]], [self.robot.goal[1]], 'g*', markersize=15)
            
            # Plot obstacles
            self.obstacle_circles = []
            for obstacle in self.obstacles:
                circle = patches.Circle((obstacle.position[0], obstacle.position[1]), obstacle.radius, fc='red', alpha=0.6)
                self.ax.add_patch(circle)
                self.obstacle_circles.append(circle)
                
            # Initial velocity arrow (empty)
            self.vel_arrow = self.ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc='blue', ec='blue')
            
            return [self.robot_plot, self.path_plot, self.goal_plot] + self.obstacle_circles + [self.vel_arrow]
        else:
            # 3D initialization would go here
            pass
    
    def update_animation(self, frame):
        """Update the animation for each frame"""
        # Update robot position using obstacle avoidance
        self.avoidance_system.update()
        
        if self.dimensions == 2:
            # Update robot position plot
            self.robot_plot.set_data([self.robot.position[0]], [self.robot.position[1]])
            
            # Update path plot
            path_x = [p[0] for p in self.robot.history]
            path_y = [p[1] for p in self.robot.history]
            self.path_plot.set_data(path_x, path_y)
            
            # Remove old velocity arrow and create new one
            self.vel_arrow.remove()
            vel_scale = 1.0
            vel_x = self.robot.velocity[0] * vel_scale
            vel_y = self.robot.velocity[1] * vel_scale
            self.vel_arrow = self.ax.arrow(self.robot.position[0], self.robot.position[1], vel_x, vel_y, head_width=0.2, head_length=0.3, fc='blue', ec='blue')
            
            # Check if robot reached goal
            dist_to_goal = np.linalg.norm(self.robot.position[:2] - self.robot.goal[:2])
            if dist_to_goal < 0.5:
                self.anim.event_source.stop()
                self.ax.set_title('Goal Reached!')
            
            return [self.robot_plot, self.path_plot, self.goal_plot] + self.obstacle_circles + [self.vel_arrow]
        else:
            # 3D animation update would go here
            pass
    
    def create_animation(self, frames=200, interval=50):
        """Create the animation"""
        self.anim = FuncAnimation(
            self.fig, 
            self.update_animation, 
            frames=frames, 
            init_func=self.init_animation, 
            blit=True,  # Important for performance
            interval=interval,
            repeat=False)
        return self.fig, self.anim
    
    def show(self):
        """Display the animation"""
        plt.tight_layout()
        plt.show()

# To run the simulation
if __name__ == "__main__":
    # Create animator for 2D simulation
    animator = RobotAnimator(dimensions=2)
    animator.setup_animation()
    fig, anim = animator.create_animation()
    animator.show()