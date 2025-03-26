import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
import random
import time

class RRTBase:
    """Base class with shared functionality for RRT variants."""
    
    def __init__(self, sim, joint_limits=None, obstacles=None):
        """
        Initialize the base RRT planner with shared functionality.
        
        Args:
            sim: Simulation object to access robot and environment data
            joint_limits: List of tuples [(min_J0, max_J0), ..., (min_J4, max_J4)]
            obstacles: List of obstacles in the workspace
        """
        self.sim = sim
        np.random.seed(42)  # For reproducibility
        
        # Default joint limits for 5-DOF Kuka YouBot arm if not provided
        self.joint_limits = joint_limits or [
            (-(169/180)*np.pi, (169/180)*np.pi),     # Joint 0
            (-(65/180)*np.pi, (90/180)*np.pi),       # Joint 1
            (-(150/180)*np.pi, (146/180)*np.pi),     # Joint 2
            (-(102.5/180)*np.pi, (102.5/180)*np.pi), # Joint 3
            (-(167.5/180)*np.pi, (167.5/180)*np.pi)  # Joint 4
        ]
        
        # Get obstacles from simulation if not provided
        self.obstacles = obstacles or self.get_obstacles_from_simulation()
        
        # Define end-effector home configuration matrix
        self.M = np.array([
            [1, 0, 0, 0.033],
            [0, 1, 0, 0],
            [0, 0, 1, 0.6546],
            [0, 0, 0, 1]
        ])
        
        # Define screw axes in body frame for arm
        self.B_list_arm = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0],
            [1, 0, 0, 0, 1],
            [0, -0.5076, -0.3526, -0.2176, 0],
            [0.033, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    
    def get_obstacles_from_simulation(self):
        """Retrieve obstacle positions and dimensions from the simulation."""
        try:
            obstacle_handles = self.sim.getObject('/Obstacles')
            num_children = self.sim.getObjectChildCount(obstacle_handles)
            obstacles = []
            
            for i in range(num_children):
                child_handle = self.sim.getObjectChild(obstacle_handles, i)
                pos = self.sim.getObjectPosition(child_handle, -1)
                dims = self.sim.getObjectBoundingBox(child_handle)
                
                obstacles.append({
                    'position': pos,
                    'dimensions': dims
                })
            
            return obstacles
        except Exception as e:
            print(f"Error retrieving obstacles: {e}")
            return []
    
    def forward_kinematics(self, config):
        """
        Compute end-effector position from joint configuration using FK.
        
        Args:
            config: Joint angles [J0-J4]
            
        Returns:
            End-effector position [x, y, z]
        """
        T_ee = mr.FKinBody(self.M, self.B_list_arm, np.array(config))
        return T_ee[:3, -1]  # Extract x,y,z position
    
    def inverse_kinematics(self, target_pose, initial_guess=None):
        """
        Solve inverse kinematics to compute joint angles for a given end-effector pose.
        
        Args:
            target_pose: Target SE(3) pose of the end-effector (4x4 matrix)
            initial_guess: Initial guess for joint angles
            
        Returns:
            Joint angles [J0, J1, J2, J3, J4] or None if no solution is found
        """
        # Default initial guess
        if initial_guess is None:
            initial_guess = [0.0, -np.pi/4, np.pi/6, -np.pi/6, np.pi/12]
        
        # Use Modern Robotics' IK solver
        joint_angles, success = mr.IKinBody(
            self.B_list_arm, self.M, target_pose, initial_guess, eomg=1e-3, ev=1e-3
        )
        
        if success:
            return self.normalize_joint_angles(joint_angles.tolist())
        else:
            print("IK failed to converge")
            return None
    
    def is_collision_free(self, config):
        """
        Check if a given joint configuration is collision-free.
        
        Args:
            config: Joint configuration [J0-J4]
            
        Returns:
            True if collision-free; False otherwise
        """
        # Compute end-effector position using forward kinematics
        end_effector_pos = self.forward_kinematics(config)
        
        for obstacle in self.obstacles:
            # Get obstacle position and dimensions
            obstacle_pos = np.array(obstacle['position'])
            obstacle_dims = np.array(obstacle['dimensions'])
            
            # Define obstacle boundaries (bounding box)
            x_min = obstacle_pos[0] - obstacle_dims[0] / 2
            x_max = obstacle_pos[0] + obstacle_dims[0] / 2
            y_min = obstacle_pos[1] - obstacle_dims[1] / 2
            y_max = obstacle_pos[1] + obstacle_dims[1] / 2
            z_min = obstacle_pos[2]
            z_max = obstacle_pos[2] + obstacle_dims[2]
            
            # Check if end-effector is within the bounding box
            if (x_min <= end_effector_pos[0] <= x_max and
                y_min <= end_effector_pos[1] <= y_max and
                z_min <= end_effector_pos[2] <= z_max):
                return False  # Collision detected
        
        return True  # No collision detected
    
    @staticmethod
    def normalize_joint_angles(joint_angles):
        """
        Normalize joint angles to the range [-π, π].
        
        Args:
            joint_angles: List or array of joint angles in radians
            
        Returns:
            Normalized joint angles in the range [-π, π]
        """
        return [(angle + np.pi) % (2 * np.pi) - np.pi for angle in joint_angles]
    
    def sample_configuration(self):
        """Sample a random configuration within joint limits."""
        return np.array([np.random.uniform(lim[0], lim[1]) for lim in self.joint_limits])
    
    def nearest_node(self, tree, config):
        """Find the nearest node in the tree to the given configuration."""
        distances = [np.linalg.norm(node - config) for node in tree]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx], nearest_idx
    
    def steer(self, from_node, to_node, step_size=0.05):
        """
        Steer from one node toward another within a small step size.
        
        Args:
            from_node: Starting node (joint configuration)
            to_node: Target node (joint configuration)
            step_size: Maximum distance to move
            
        Returns:
            New node closer to the target node
        """
        direction = to_node - from_node
        distance = np.linalg.norm(direction)
        
        if distance < step_size:
            return to_node
        else:
            new_node = from_node + direction / distance * step_size
            # Clip to joint limits
            return np.clip(new_node, 
                           [lim[0] for lim in self.joint_limits],
                           [lim[1] for lim in self.joint_limits])
    
    def reconstruct_path(self, edges, goal_idx, start_idx):
        """
        Reconstruct the path from start to goal using the edges dictionary.
        
        Args:
            edges: Dictionary of {child_idx: parent_idx}
            goal_idx: Index of the goal node
            start_idx: Index of the start node
            
        Returns:
            List of node indices representing the path
        """
        path = [goal_idx]
        current_idx = goal_idx
        
        while current_idx != start_idx:
            current_idx = edges[current_idx]
            path.append(current_idx)
        
        return path[::-1]  # Reverse to get start-to-goal order
    
    def smooth_path(self, path, max_iter=100):
        """
        Smooth the path by removing unnecessary waypoints.
        
        Args:
            path: List of configurations
            max_iter: Maximum number of iterations
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        smoothed_path = path.copy()
        
        for _ in range(max_iter):
            if len(smoothed_path) <= 2:
                break
            
            # Randomly select two points
            i, j = sorted(random.sample(range(len(smoothed_path)), 2))
            
            if j > i + 1:
                # Check if direct connection is collision-free
                direct_path_valid = True
                steps = max(int(np.linalg.norm(smoothed_path[j] - smoothed_path[i]) / 0.05), 5)
                
                for step in range(1, steps):
                    interp_config = smoothed_path[i] + (smoothed_path[j] - smoothed_path[i]) * (step / steps)
                    if not self.is_collision_free(interp_config):
                        direct_path_valid = False
                        break
                
                if direct_path_valid:
                    # Remove points between i and j
                    smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
        
        return smoothed_path
    
    def visualize_trajectory(self, path, title="Trajectory"):
        """
        Visualize the end-effector trajectory and obstacles in 3D space.
        
        Args:
            path: List of joint configurations
            title: Plot title
        """
        # Extract end-effector positions using forward kinematics
        ee_positions = [self.forward_kinematics(config) for config in path]
        x_traj, y_traj, z_traj = zip(*ee_positions)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot end-effector trajectory
        ax.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label="End-Effector Path")
        ax.scatter(x_traj[0], y_traj[0], z_traj[0], color='g', s=100, label="Start")
        ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='r', s=100, label="Goal")
        
        # Plot obstacles
        for obs in self.obstacles:
            pos = obs['position']
            dims = obs['dimensions']
            
            # Create cube vertices
            x_min, x_max = pos[0] - dims[0]/2, pos[0] + dims[0]/2
            y_min, y_max = pos[1] - dims[1]/2, pos[1] + dims[1]/2
            z_min, z_max = pos[2], pos[2] + dims[2]
            
            # Plot obstacle as wireframe cuboid
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [y_min, y_min, y_max, y_max, y_min],
                    [z_min, z_min, z_min, z_min, z_min], 'k-')
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [y_min, y_min, y_max, y_max, y_min],
                    [z_max, z_max, z_max, z_max, z_max], 'k-')
            for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
                ax.plot([x, x], [y, y], [z_min, z_max], 'k-')
        
        # Set labels and title
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.show()


class RRT(RRTBase):
    """Standard RRT implementation for robotic arm planning."""
    
    def __init__(self, sim, joint_limits=None, obstacles=None):
        """Initialize the RRT planner."""
        super().__init__(sim, joint_limits, obstacles)
    
    def plan(self, start_config, goal_config, max_iterations=1000, goal_bias=0.1, step_size=0.05):
        """
        Plan a collision-free path using RRT.
        
        Args:
            start_config: Initial joint configuration
            goal_config: Goal joint configuration
            max_iterations: Maximum number of iterations
            goal_bias: Probability of sampling the goal
            step_size: Maximum step size for extending the tree
            
        Returns:
            List of joint configurations representing the path
        """
        # Normalize configurations
        start = np.array(self.normalize_joint_angles(start_config))
        goal = np.array(self.normalize_joint_angles(goal_config))
        
        # Check if start and goal are collision-free
        if not self.is_collision_free(start):
            raise ValueError("Start configuration is in collision!")
        
        if not self.is_collision_free(goal):
            raise ValueError("Goal configuration is in collision!")
        
        # Initialize tree and edges
        tree = [start]
        edges = {}  # {child_idx: parent_idx}
        
        # Start timer
        start_time = time.time()
        
        # Main RRT loop
        for i in range(max_iterations):
            # Sample random configuration with goal bias
            if np.random.random() < goal_bias:
                rand_config = goal
            else:
                rand_config = self.sample_configuration()
            
            # Find nearest node in the tree
            nearest, nearest_idx = self.nearest_node(tree, rand_config)
            
            # Steer towards the sampled configuration
            new_config = self.steer(nearest, rand_config, step_size)
            
            # Check if new configuration is collision-free
            if self.is_collision_free(new_config):
                # Add new configuration to the tree
                tree.append(new_config)
                new_idx = len(tree) - 1
                edges[new_idx] = nearest_idx
                
                # Check if goal is reached
                if np.linalg.norm(new_config - goal) < step_size:
                    # Goal is close enough, add direct connection to goal if possible
                    if self.is_collision_free(self.steer(new_config, goal, step_size)):
                        tree.append(goal)
                        goal_idx = len(tree) - 1
                        edges[goal_idx] = new_idx
                        
                        # Reconstruct and smooth the path
                        path_indices = self.reconstruct_path(edges, goal_idx, 0)
                        path = [tree[idx] for idx in path_indices]
                        smoothed_path = self.smooth_path(path)
                        
                        elapsed_time = time.time() - start_time
                        print(f"Path found in {elapsed_time:.2f} seconds after {i+1} iterations")
                        return smoothed_path
        
        raise Exception(f"Failed to find a path after {max_iterations} iterations")


class InformedRRT(RRTBase):
    """Informed RRT* implementation for robotic arm planning."""
    
    def __init__(self, sim, joint_limits=None, obstacles=None):
        """Initialize the Informed RRT* planner."""
        super().__init__(sim, joint_limits, obstacles)
    
    def plan(self, start_config, goal_config, max_iterations=1000, goal_bias=0.1, step_size=0.05, search_radius=0.5):
        """
        Plan a collision-free path using Informed RRT*.
        
        Args:
            start_config: Initial joint configuration
            goal_config: Goal joint configuration
            max_iterations: Maximum number of iterations
            goal_bias: Probability of sampling the goal
            step_size: Maximum step size for extending the tree
            search_radius: Radius for finding nearby nodes
            
        Returns:
            List of joint configurations representing the path
        """
        # Normalize configurations
        start = np.array(self.normalize_joint_angles(start_config))
        goal = np.array(self.normalize_joint_angles(goal_config))
        
        # Check if start and goal are collision-free
        if not self.is_collision_free(start):
            raise ValueError("Start configuration is in collision!")
        
        if not self.is_collision_free(goal):
            raise ValueError("Goal configuration is in collision!")
        
        # Initialize tree, edges, and costs
        tree = [start]
        edges = {}  # {child_idx: parent_idx}
        costs = {0: 0.0}  # Cost from start to each node
        
        # Start timer
        start_time = time.time()
        
        # Best solution so far
        best_cost = float('inf')
        best_path = None
        
        # Main RRT* loop
        for i in range(max_iterations):
            # Sample random configuration with goal bias
            if np.random.random() < goal_bias:
                rand_config = goal
            else:
                if best_cost < float('inf'):
                    # Use informed sampling once a solution is found
                    rand_config = self.informed_sample(start, goal, best_cost)
                else:
                    rand_config = self.sample_configuration()
            
            # Find nearest node in the tree
            nearest, nearest_idx = self.nearest_node(tree, rand_config)
            
            # Steer towards the sampled configuration
            new_config = self.steer(nearest, rand_config, step_size)
            
            # Check if new configuration is collision-free
            if self.is_collision_free(new_config):
                # Find nearby nodes within search_radius
                nearby_indices = [j for j, node in enumerate(tree) 
                                 if np.linalg.norm(node - new_config) < search_radius]
                
                # Add new configuration to the tree
                tree.append(new_config)
                new_idx = len(tree) - 1
                
                # Initialize with connection to nearest node
                min_cost = costs[nearest_idx] + np.linalg.norm(new_config - nearest)
                min_idx = nearest_idx
                
                # Check for better connections from nearby nodes
                for j in nearby_indices:
                    cost = costs[j] + np.linalg.norm(tree[j] - new_config)
                    if cost < min_cost and self.is_collision_free_path(tree[j], new_config):
                        min_cost = cost
                        min_idx = j
                
                # Connect new node to best parent
                edges[new_idx] = min_idx
                costs[new_idx] = min_cost
                
                # Rewire the tree
                for j in nearby_indices:
                    cost = costs[new_idx] + np.linalg.norm(tree[j] - new_config)
                    if cost < costs[j] and self.is_collision_free_path(new_config, tree[j]):
                        edges[j] = new_idx
                        costs[j] = cost
                
                # Check if goal is reached
                dist_to_goal = np.linalg.norm(new_config - goal)
                if dist_to_goal < step_size:
                    # Goal is close enough, add direct connection to goal if possible
                    direct_cost = costs[new_idx] + dist_to_goal
                    
                    if direct_cost < best_cost and self.is_collision_free(self.steer(new_config, goal, step_size)):
                        # Add goal to tree
                        tree.append(goal)
                        goal_idx = len(tree) - 1
                        edges[goal_idx] = new_idx
                        costs[goal_idx] = direct_cost
                        
                        # Update best solution
                        best_cost = direct_cost
                        path_indices = self.reconstruct_path(edges, goal_idx, 0)
                        best_path = [tree[idx] for idx in path_indices]
                        
                        print(f"Found better path with cost {best_cost:.4f}")
        
        elapsed_time = time.time() - start_time
        
        if best_path is None:
            raise Exception(f"Failed to find a path after {max_iterations} iterations")
        
        print(f"Final path found in {elapsed_time:.2f} seconds with cost {best_cost:.4f}")
        smoothed_path = self.smooth_path(best_path)
        return smoothed_path
    
    def informed_sample(self, start, goal, c_best):
        """
        Sample from an ellipsoidal region that may contain better paths.
        
        Args:
            start: Start configuration
            goal: Goal configuration
            c_best: Cost of best path found so far
            
        Returns:
            Sampled configuration
        """
        # Standard sampling (simplified, without true ellipsoidal sampling)
        # With low probability, sample from the entire space
        if np.random.random() < 0.2:
            return self.sample_configuration()
        
        # Otherwise, sample near the direct path between start and goal
        alpha = np.random.random()
        direct_config = start + alpha * (goal - start)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, len(start))
        noisy_config = direct_config + noise
        
        # Clip to joint limits
        return np.clip(noisy_config,
                      [lim[0] for lim in self.joint_limits],
                      [lim[1] for lim in self.joint_limits])
    
    def is_collision_free_path(self, from_node, to_node, steps=5):
        """
        Check if the path between two configurations is collision-free.
        
        Args:
            from_node: Start configuration
            to_node: End configuration
            steps: Number of interpolation steps
            
        Returns:
            True if path is collision-free, False otherwise
        """
        for i in range(1, steps):
            t = i / steps
            interp_config = from_node + t * (to_node - from_node)
            if not self.is_collision_free(interp_config):
                return False
        return True


# Example usage:
def main():
    # Placeholder for simulation object - in real use you'd have your CoppeliaSim interface
    class SimPlaceholder:
        def getObject(self, name):
            return 0  # Placeholder
        
        def getObjectChildCount(self, handle):
            return 0  # Placeholder
        
        def getObjectChild(self, handle, index):
            return 0  # Placeholder
        
        def getObjectPosition(self, handle, frame):
            return [0, 0, 0]  # Placeholder
        
        def getObjectBoundingBox(self, handle):
            return [0.1, 0.1, 0.1]  # Placeholder
        
        def getJointPosition(self, handle):
            return 0.0  # Placeholder
    
    sim = SimPlaceholder()
    
    # Create some obstacles for testing
    obstacles = [
        {'position': [0.5, 0.0, 0.2], 'dimensions': [0.1, 0.1, 0.2]},
        {'position': [0.0, 0.5, 0.2], 'dimensions': [0.1, 0.1, 0.2]},
        {'position': [-0.3, -0.3, 0.1], 'dimensions': [0.2, 0.2, 0.2]}
    ]
    
    # Create RRT and Informed RRT planners
    rrt_planner = RRT(sim, obstacles=obstacles)
    informed_rrt_planner = InformedRRT(sim, obstacles=obstacles)
    
    # Define start and goal configurations
    start_config = [0.0, 0.0, 0.0, 0.0, 0.0]
    goal_config = [1.0, 0.5, -0.5, 0.8, 0.2]
    
    # Plan paths
    try:
        print("Planning with standard RRT...")
        rrt_path = rrt_planner.plan(start_config, goal_config)
        rrt_planner.visualize_trajectory(rrt_path, "RRT Path")
        
        print("\nPlanning with Informed RRT*...")
        informed_path = informed_rrt_planner.plan(start_config, goal_config)
        informed_rrt_planner.visualize_trajectory(informed_path, "Informed RRT* Path")
        
        # Compare path lengths
        rrt_length = sum(np.linalg.norm(rrt_path[i+1] - rrt_path[i]) for i in range(len(rrt_path)-1))
        informed_length = sum(np.linalg.norm(informed_path[i+1] - informed_path[i]) for i in range(len(informed_path)-1))
        
        print(f"\nRRT path length: {rrt_length:.4f}")
        print(f"Informed RRT* path length: {informed_length:.4f}")
        print(f"Improvement: {(1 - informed_length/rrt_length)*100:.2f}%")
        
    except Exception as e:
        print(f"Planning failed: {e}")

if __name__ == "__main__":
    main()


# import numpy as np
# import modern_robotics as mr
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# class InformedRRTStar:
#     def __init__(self, sim):
#         """
#         Initialize Informed RRT* planner for 5-DOF robotic arm.

#         Args:
#             sim: Simulation object to access obstacle and configuration data.
#             start_config: Current joint configuration [J0, J1, J2, J3, J4].
#             goal_config: Goal joint configuration [J0, J1, J2, J3, J4].
#             joint_limits: List of tuples [(min_J0, max_J0), ..., (min_J4, max_J4)].
#         """
#         self.sim = sim
#         np.random.seed(42)

#         self.joint_limits = [
#                 (-(169/180)*np.pi, (169/180)*np.pi),
#                 (-(65/180)*np.pi, (90/180)*np.pi),
#                 (-(150/180)*np.pi, (146/180)*np.pi),
#                 (-(102.5/180)*np.pi, (102.5/180)*np.pi),
#                 (-(167.5/180)*np.pi, (167.5/180)*np.pi),
#             ]
        
#         self.obstacles = self.get_obstacles_from_simulation()  # Get obstacles from simulation

#     def get_obstacles_from_simulation(self):
#         """
#         Retrieve obstacle positions and dimensions from the simulation using object handles.

#         Returns:
#             List of obstacles with positions and dimensions.
#         """
#         obstacle_handles = self.sim.getObject('/Obstacles')  # Parent group containing all obstacles
#         num_children = self.sim.getObjectChildCount(obstacle_handles)

#         obstacles = []
#         for i in range(num_children):
#             child_handle = self.sim.getObjectChild(obstacle_handles, i)

#             # Get position of the obstacle in world coordinates
#             pos = self.sim.getObjectPosition(child_handle, -1)  # [x, y, z]

#             # Get dimensions (bounding box size)
#             dims = self.sim.getObjectBoundingBox(child_handle)  # [x_length, y_width, z_height]

#             obstacles.append({
#                 'position': pos,
#                 'dimensions': dims,
#             })
#         return obstacles

#     def is_collision_free(self, config):
#         """
#         Check if a given joint configuration is collision-free.

#         Args:
#             config: Joint configuration [J0-J1-J2-J3-J4].

#         Returns:
#             True if collision-free; False otherwise.
#         """
#         # Compute end-effector position using forward kinematics
#         end_effector_pos = self.forward_kinematics(config)

#         for obstacle in self.obstacles:
#             # Get obstacle position and dimensions
#             obstacle_pos = np.array(obstacle['position'])  # [x, y, z]
#             obstacle_dims = np.array(obstacle['dimensions'])  # [length, width, height]

#             # Define obstacle boundaries (bounding box)
#             x_min = obstacle_pos[0] - obstacle_dims[0] / 2
#             x_max = obstacle_pos[0] + obstacle_dims[0] / 2
#             y_min = obstacle_pos[1] - obstacle_dims[1] / 2
#             y_max = obstacle_pos[1] + obstacle_dims[1] / 2
#             z_min = obstacle_pos[2]
#             z_max = obstacle_pos[2] + obstacle_dims[2]

#             # Check if end-effector is within the bounding box
#             if (
#                 x_min <= end_effector_pos[0] <= x_max and
#                 y_min <= end_effector_pos[1] <= y_max and
#                 z_min <= end_effector_pos[2] <= z_max
#             ):
#                 return False  # Collision detected

#         return True  # No collision detected

#     @staticmethod
#     def forward_kinematics(config):
#         """
#         Compute end-effector position from joint configuration using FK logic.

#         Args:
#             config: Joint angles [J0-J4].

#         Returns:
#             End-effector position [x, y, z].
#         """
#         M = np.array([[1, 0, 0, 0.033],
#                       [0, 1, 0, 0],
#                       [0, 0, 1, 0.6546],
#                       [0, 0, 0, 1]])
        
#         B_list_arm = np.array([[0 , 0 , 0 , 0 , 0],
#                                [0 ,-1 ,-1 ,-1 ,-1],
#                                [1 , 0 , 0 , 0 , 1],
#                                [0 ,-0.5076 ,-0.3526 ,-0.2176 ,-0.2176],
#                                [0.033 , 0 , 0 , 0 , 0],
#                                [0 , 0 , 0 , 0 , 0]])
        
#         T_ee = mr.FKinBody(M,B_list_arm,np.array(config))
        
#         return T_ee[:3,-1]  # Extract x,y,z position
    
#     @staticmethod
#     def inverse_kinematics(target_pose):
#         """
#         Solve inverse kinematics to compute joint angles for a given end-effector pose.

#         Args:
#             target_pose: Target SE(3) pose of the end-effector (4x4 matrix).

#         Returns:
#             Joint angles [J0, J1, J2, J3, J4] or None if no solution is found.
#         """
#         # Define home configuration of the end-effector
#         M = np.array([[1, 0, 0, 0.033],
#                     [0, 1, 0, 0],
#                     [0, 0, 1, 0.6546],
#                     [0, 0, 0, 1]])
        
#         # Define screw axes in body frame
#         B_list_arm = np.array([[0,0,0,0,0],
#                         [0,-1,-1,-1,0],
#                         [1,0,0,0,1],
#                         [0,-0.5076,-0.3526,-0.2176,0],
#                         [0.033,0,0,0,0],
#                         [0,0,0,0,0]])
        
#         # Initial guess for joint angles
#         initial_guess = [0.0, -np.pi/4, np.pi/6, -np.pi/6, np.pi/12]

#         # Use Modern Robotics' IK solver
#         joint_angles, success = mr.IKinBody(B_list_arm, M, target_pose, initial_guess, eomg=1e-3, ev=1e-3)
        
#         if success:
#             return joint_angles.tolist()
#         else:
#             raise Exception("Failed to find a valid joint configuration.")
    
#     @staticmethod
#     def normalize_joint_angles(joint_angles):
#         """
#         Normalize joint angles to the range [-π, π].

#         Args:
#             joint_angles: List or array of joint angles in radians.

#         Returns:
#             Normalized joint angles in the range [-π, π].
#         """
#         return [(angle + np.pi) % (2 * np.pi) - np.pi for angle in joint_angles]

#     @staticmethod
#     def sample(joint_limits):
#         """
#         Sample a random configuration within joint limits.

#         Returns:
#             Random joint configuration [J0-J4].
#         """
#         return np.array([np.random.uniform(lim[0], lim[1]) for lim in joint_limits])

#     @staticmethod
#     def nearest(tree, config):
#         """
#         Find the nearest node in the tree to the given configuration.

#         Args:
#             config: Joint configuration [J0-J4].

#         Returns:
#             Nearest node in the tree.
#         """
#         distances = [np.linalg.norm(node - config) for node in tree]
#         nearest_idx = np.argmin(distances)
#         return tree[nearest_idx]

#     @staticmethod
#     def steer(joint_limits, from_node, to_node):
#         """
#         Steer from one node toward another within a small step size.

#         Args:
#             from_node: Starting node (joint configuration).
#             to_node: Target node (joint configuration).

#         Returns:
#             New node closer to the target node.
#         """
#         direction = to_node - from_node
#         step_size = 0.05
#         new_node = from_node + direction / np.linalg.norm(direction) * step_size
#         return np.clip(new_node,
#                        [lim[0] for lim in joint_limits],
#                        [lim[1] for lim in joint_limits])

#     def plan(self, goal_config, start_config=None, base_path=None):
#         """
#         Plan a collision-free path using Informed RRT*.

#         Returns:
#             List of joint configurations representing the path.
#         """
#         max_iterations = 1000  # Maximum number of iterations for RRT*

#         if start_config is None:
#             arm_joint_handles = [
#                 self.sim.getObject('/youBotArmJoint0'),
#                 self.sim.getObject('/youBotArmJoint1'),
#                 self.sim.getObject('/youBotArmJoint2'),
#                 self.sim.getObject('/youBotArmJoint3'),
#                 self.sim.getObject('/youBotArmJoint4'),
#             ]
#             start_config = [self.sim.getJointPosition(joint) for joint in arm_joint_handles]

#         start_config = self.normalize_joint_angles(start_config)
#         print("Start Configuration:", start_config)
#         start = np.array(start_config)

#         goal_config = self.inverse_kinematics(goal_config)
#         goal_config = self.normalize_joint_angles(goal_config)
#         print("Goal Configuration:", goal_config)
#         goal = np.array(goal_config)

#         tree = [start]  # Initialize tree with start configuration
#         edges = {}  # Store parent-child relationships for backtracking

#         for _ in range(max_iterations):
#             # Sample a random configuration within joint limits
#             sample_node = self.sample(self.joint_limits)
            
#             # Find the nearest node in the tree to the sampled configuration
#             nearest_node = self.nearest(tree, sample_node)
            
#             # Steer towards the sampled configuration
#             new_node = self.steer(self.joint_limits, nearest_node, sample_node)
            
#             # Check if the new node is collision-free
#             if self.is_collision_free(new_node):
#                 # Add the new node to the tree and record its parent
#                 tree.append(new_node)
#                 edges[tuple(new_node)] = tuple(nearest_node)
                
#                 # Check if the goal configuration is reached
#                 if np.linalg.norm(new_node - goal) < 0.05:  # Threshold for goal proximity
#                     path = [goal]
#                     while tuple(path[-1]) != tuple(start):
#                         path.append(edges[tuple(path[-1])])

#                     self.visualize_trajectory(path[::-1], self.obstacles)
#                     return path[::-1]  # Return path in start-to-goal order
            
#         raise Exception("Failed to find a collision-free path within maximum iterations.")

#     def visualize_trajectory(self, path, obstacles):
#         """
#         Visualize the end-effector trajectory and obstacles in 3D space.

#         Args:
#             path: List of joint configurations representing the planned trajectory.
#             obstacles: List of obstacle positions [[x1, y1, z1], [x2, y2, z2], ...].
#         """
#         # Extract end-effector positions from joint configurations using FK
#         x_traj, y_traj, z_traj = [], [], []
#         for config in path:
#             end_effector_pos = self.forward_kinematics(config)
#             x_traj.append(end_effector_pos[0])
#             y_traj.append(end_effector_pos[1])
#             z_traj.append(end_effector_pos[2])
        
#         # Create a 3D plot
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Plot end-effector trajectory
#         ax.plot(x_traj, y_traj, z_traj, label="End-Effector Trajectory", color='blue', linewidth=2)
        
#         # Plot obstacles as red points
#         obstacle_x = [obs[0] for obs in obstacles]
#         obstacle_y = [obs[1] for obs in obstacles]
#         obstacle_z = [obs[2] for obs in obstacles]
#         ax.scatter(obstacle_x, obstacle_y, obstacle_z, label="Obstacles", color='red', s=50)
        
#         # Add labels and legend
#         ax.set_xlabel("X-axis")
#         ax.set_ylabel("Y-axis")
#         ax.set_zlabel("Z-axis")
#         ax.set_title("End-Effector Trajectory with Obstacles")
#         ax.legend()
        
#         plt.show()

