import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class InformedRRTStar:
    def __init__(self, sim):
        """
        Initialize Informed RRT* planner for 5-DOF robotic arm.

        Args:
            sim: Simulation object to access obstacle and configuration data.
            start_config: Current joint configuration [J0, J1, J2, J3, J4].
            goal_config: Goal joint configuration [J0, J1, J2, J3, J4].
            joint_limits: List of tuples [(min_J0, max_J0), ..., (min_J4, max_J4)].
        """
        self.sim = sim
        np.random.seed(42)

        self.joint_limits = [
                (-(169/180)*np.pi, (169/180)*np.pi),
                (-(65/180)*np.pi, (90/180)*np.pi),
                (-(150/180)*np.pi, (146/180)*np.pi),
                (-(102.5/180)*np.pi, (102.5/180)*np.pi),
                (-(167.5/180)*np.pi, (167.5/180)*np.pi),
            ]
        
        self.obstacles = self.get_obstacles_from_simulation()  # Get obstacles from simulation

    def get_obstacles_from_simulation(self):
        """
        Retrieve obstacle positions and dimensions from the simulation using object handles.

        Returns:
            List of obstacles with positions and dimensions.
        """
        obstacle_handles = self.sim.getObject('/Obstacles')  # Parent group containing all obstacles
        num_children = self.sim.getObjectChildCount(obstacle_handles)

        obstacles = []
        for i in range(num_children):
            child_handle = self.sim.getObjectChild(obstacle_handles, i)

            # Get position of the obstacle in world coordinates
            pos = self.sim.getObjectPosition(child_handle, -1)  # [x, y, z]

            # Get dimensions (bounding box size)
            dims = self.sim.getObjectBoundingBox(child_handle)  # [x_length, y_width, z_height]

            obstacles.append({
                'position': pos,
                'dimensions': dims,
            })
        return obstacles

    def is_collision_free(self, config):
        """
        Check if a given joint configuration is collision-free.

        Args:
            config: Joint configuration [J0-J1-J2-J3-J4].

        Returns:
            True if collision-free; False otherwise.
        """
        # Compute end-effector position using forward kinematics
        end_effector_pos = self.forward_kinematics(config)

        for obstacle in self.obstacles:
            # Get obstacle position and dimensions
            obstacle_pos = np.array(obstacle['position'])  # [x, y, z]
            obstacle_dims = np.array(obstacle['dimensions'])  # [length, width, height]

            # Define obstacle boundaries (bounding box)
            x_min = obstacle_pos[0] - obstacle_dims[0] / 2
            x_max = obstacle_pos[0] + obstacle_dims[0] / 2
            y_min = obstacle_pos[1] - obstacle_dims[1] / 2
            y_max = obstacle_pos[1] + obstacle_dims[1] / 2
            z_min = obstacle_pos[2]
            z_max = obstacle_pos[2] + obstacle_dims[2]

            # Check if end-effector is within the bounding box
            if (
                x_min <= end_effector_pos[0] <= x_max and
                y_min <= end_effector_pos[1] <= y_max and
                z_min <= end_effector_pos[2] <= z_max
            ):
                return False  # Collision detected

        return True  # No collision detected

    @staticmethod
    def forward_kinematics(config):
        """
        Compute end-effector position from joint configuration using FK logic.

        Args:
            config: Joint angles [J0-J4].

        Returns:
            End-effector position [x, y, z].
        """
        M = np.array([[1, 0, 0, 0.033],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0.6546],
                      [0, 0, 0, 1]])
        
        B_list_arm = np.array([[0 , 0 , 0 , 0 , 0],
                               [0 ,-1 ,-1 ,-1 ,-1],
                               [1 , 0 , 0 , 0 , 1],
                               [0 ,-0.5076 ,-0.3526 ,-0.2176 ,-0.2176],
                               [0.033 , 0 , 0 , 0 , 0],
                               [0 , 0 , 0 , 0 , 0]])
        
        T_ee = mr.FKinBody(M,B_list_arm,np.array(config))
        
        return T_ee[:3,-1]  # Extract x,y,z position
    
    @staticmethod
    def inverse_kinematics(target_pose):
        """
        Solve inverse kinematics to compute joint angles for a given end-effector pose.

        Args:
            target_pose: Target SE(3) pose of the end-effector (4x4 matrix).

        Returns:
            Joint angles [J0, J1, J2, J3, J4] or None if no solution is found.
        """
        # Define home configuration of the end-effector
        M = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])
        
        # Define screw axes in body frame
        B_list_arm = np.array([[0,0,0,0,0],
                        [0,-1,-1,-1,0],
                        [1,0,0,0,1],
                        [0,-0.5076,-0.3526,-0.2176,0],
                        [0.033,0,0,0,0],
                        [0,0,0,0,0]])
        
        # Initial guess for joint angles
        initial_guess = [0.0, -np.pi/4, np.pi/6, -np.pi/6, np.pi/12]

        # Use Modern Robotics' IK solver
        joint_angles, success = mr.IKinBody(B_list_arm, M, target_pose, initial_guess, eomg=1e-3, ev=1e-3)
        
        if success:
            return joint_angles.tolist()
        else:
            raise Exception("Failed to find a valid joint configuration.")
    
    @staticmethod
    def normalize_joint_angles(joint_angles):
        """
        Normalize joint angles to the range [-π, π].

        Args:
            joint_angles: List or array of joint angles in radians.

        Returns:
            Normalized joint angles in the range [-π, π].
        """
        return [(angle + np.pi) % (2 * np.pi) - np.pi for angle in joint_angles]

    @staticmethod
    def sample(joint_limits):
        """
        Sample a random configuration within joint limits.

        Returns:
            Random joint configuration [J0-J4].
        """
        return np.array([np.random.uniform(lim[0], lim[1]) for lim in joint_limits])

    @staticmethod
    def nearest(tree, config):
        """
        Find the nearest node in the tree to the given configuration.

        Args:
            config: Joint configuration [J0-J4].

        Returns:
            Nearest node in the tree.
        """
        distances = [np.linalg.norm(node - config) for node in tree]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx]

    @staticmethod
    def steer(joint_limits, from_node, to_node):
        """
        Steer from one node toward another within a small step size.

        Args:
            from_node: Starting node (joint configuration).
            to_node: Target node (joint configuration).

        Returns:
            New node closer to the target node.
        """
        direction = to_node - from_node
        step_size = 0.05
        new_node = from_node + direction / np.linalg.norm(direction) * step_size
        return np.clip(new_node,
                       [lim[0] for lim in joint_limits],
                       [lim[1] for lim in joint_limits])

    def plan(self, goal_config, start_config=None, base_path=None):
        """
        Plan a collision-free path using Informed RRT*.

        Returns:
            List of joint configurations representing the path.
        """
        max_iterations = 1000  # Maximum number of iterations for RRT*

        if start_config is None:
            arm_joint_handles = [
                self.sim.getObject('/youBotArmJoint0'),
                self.sim.getObject('/youBotArmJoint1'),
                self.sim.getObject('/youBotArmJoint2'),
                self.sim.getObject('/youBotArmJoint3'),
                self.sim.getObject('/youBotArmJoint4'),
            ]
            start_config = [self.sim.getJointPosition(joint) for joint in arm_joint_handles]

        start_config = self.normalize_joint_angles(start_config)
        print("Start Configuration:", start_config)
        start = np.array(start_config)

        goal_config = self.inverse_kinematics(goal_config)
        goal_config = self.normalize_joint_angles(goal_config)
        print("Goal Configuration:", goal_config)
        goal = np.array(goal_config)

        tree = [start]  # Initialize tree with start configuration
        edges = {}  # Store parent-child relationships for backtracking

        for _ in range(max_iterations):
            # Sample a random configuration within joint limits
            sample_node = self.sample(self.joint_limits)
            
            # Find the nearest node in the tree to the sampled configuration
            nearest_node = self.nearest(tree, sample_node)
            
            # Steer towards the sampled configuration
            new_node = self.steer(self.joint_limits, nearest_node, sample_node)
            
            # Check if the new node is collision-free
            if self.is_collision_free(new_node):
                # Add the new node to the tree and record its parent
                tree.append(new_node)
                edges[tuple(new_node)] = tuple(nearest_node)
                
                # Check if the goal configuration is reached
                if np.linalg.norm(new_node - goal) < 0.05:  # Threshold for goal proximity
                    path = [goal]
                    while tuple(path[-1]) != tuple(start):
                        path.append(edges[tuple(path[-1])])

                    self.visualize_trajectory(path[::-1], self.obstacles)
                    return path[::-1]  # Return path in start-to-goal order
            
        raise Exception("Failed to find a collision-free path within maximum iterations.")

    def visualize_trajectory(self, path, obstacles):
        """
        Visualize the end-effector trajectory and obstacles in 3D space.

        Args:
            path: List of joint configurations representing the planned trajectory.
            obstacles: List of obstacle positions [[x1, y1, z1], [x2, y2, z2], ...].
        """
        # Extract end-effector positions from joint configurations using FK
        x_traj, y_traj, z_traj = [], [], []
        for config in path:
            end_effector_pos = self.forward_kinematics(config)
            x_traj.append(end_effector_pos[0])
            y_traj.append(end_effector_pos[1])
            z_traj.append(end_effector_pos[2])
        
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot end-effector trajectory
        ax.plot(x_traj, y_traj, z_traj, label="End-Effector Trajectory", color='blue', linewidth=2)
        
        # Plot obstacles as red points
        obstacle_x = [obs[0] for obs in obstacles]
        obstacle_y = [obs[1] for obs in obstacles]
        obstacle_z = [obs[2] for obs in obstacles]
        ax.scatter(obstacle_x, obstacle_y, obstacle_z, label="Obstacles", color='red', s=50)
        
        # Add labels and legend
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title("End-Effector Trajectory with Obstacles")
        ax.legend()
        
        plt.show()

