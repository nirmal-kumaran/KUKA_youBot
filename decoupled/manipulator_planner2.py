import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import modern_robotics as mr

class RRT:
    class Node:
        """RRT Node for 5DOF configuration"""
        def __init__(self, theta_config):
            self.q_values = theta_config  # List of 5 joint angles
            self.path_q_values = []       # Intermediate configurations
            self.parent = None

    def __init__(self, sim, obstacles=None):
        
        self.goal_sample_rate = 15
        self.path_resolution = 0.5
        self.max_iter = 500
        self.expand_dis = 0.5
        self.sim = sim

        # Get obstacles from simulation if not provided
        self.obstacles = obstacles or self.get_obstacles_from_simulation()
        
        self.arm_joints = [
        sim.getObject('/youBotArmJoint0'),
        sim.getObject('/youBotArmJoint1'),
        sim.getObject('/youBotArmJoint2'),
        sim.getObject('/youBotArmJoint3'),
        sim.getObject('/youBotArmJoint4')
        ]
        
        # Create arm collection properly
        self.arm_collection = sim.createCollection(0)
        for joint in self.arm_joints:
            sim.addItemToCollection(self.arm_collection, sim.handle_single, joint, 0)
        
        # Create obstacle collection
        self.obstacle_collection = sim.createCollection(0)
        for obs in self.obstacles:
            sim.addItemToCollection(self.obstacle_collection, sim.handle_single, obs['handle'], 0)

        self.joint_limits = [
            (-(169/180)*np.pi, (169/180)*np.pi),     # Joint 0
            (-(65/180)*np.pi, (90/180)*np.pi),       # Joint 1
            (-(150/180)*np.pi, (146/180)*np.pi),     # Joint 2
            (-(102.5/180)*np.pi, (102.5/180)*np.pi), # Joint 3
            (-(167.5/180)*np.pi, (167.5/180)*np.pi)  # Joint 4
        ]

    def measure_time_for_iterations(self, start_config, goal_config, iterations_list):
        """Measure planning time for different iteration counts"""
        results = {}
        
        # Store original parameters
        original_max_iter = self.max_iter
        original_goal_sample_rate = self.goal_sample_rate
        
        for max_iter in iterations_list:
            # Configure planner for this test
            self.max_iter = max_iter
            self.goal_sample_rate = 5  # More aggressive goal sampling
            
            # Warm-up: Solve once to initialize data structures
            _ = self.planning(start_config, goal_config)
            
            # Time measurement
            start_time = time.time()
            path = self.planning(start_config, goal_config)
            elapsed_time = time.time() - start_time
            
            # Store results
            results[max_iter] = {
                'time': elapsed_time,
                'success': len(path) > 0,
                'path_length': len(path)
            }
        
        # Restore original parameters
        self.max_iter = original_max_iter
        self.goal_sample_rate = original_goal_sample_rate
        
        return results
    
    def get_base_transform(self):
        """Compute base's transformation matrix in world frame"""
        x, y, z = self.base_position
        return np.array([
            [np.cos(self.base_orientation), -np.sin(self.base_orientation), 0, x],
            [np.sin(self.base_orientation), np.cos(self.base_orientation), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def normalize_joint_angles(joint_angles):
        """
        Normalize joint angles to the range [-π, π].
        """
        return [(angle + np.pi) % (2 * np.pi) - np.pi for angle in joint_angles]
    
    def get_ee_positions(self, path):
        """Convert joint configurations to end-effector positions"""
        ee_positions = []
        for config in path:
            position = self.forward_kinematics(config)
            ee_positions.append(position.tolist())
        return ee_positions

    ###------------------------------------------###
    def inverse_kinematics(self, target_pose_world, base_position=(0,0,0.0963), base_orientation=0, initial_guess=None):
        """
        Solve inverse kinematics to compute joint angles for a given end-effector pose.
        
        Args:
            target_pose: Target SE(3) pose of the end-effector (4x4 matrix)
            initial_guess: Initial guess for joint angles
            
        Returns:
            Joint angles [J0, J1, J2, J3, J4] or None if no solution is found
        """
        self.base_position, self.base_orientation = np.array(base_position), base_orientation
        # Default initial guess
        if initial_guess is None:
            initial_guess = [0, 0 , np.pi/4, 0, 0]  
        
        # Use Modern Robotics' IK solver
        M = np.array([[1, 0, 0, 0.033],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0.6546],
                      [0, 0, 0, 1]])
        B_list_arm = np.array([[0,0,0,0,0],
                        [0,-1,-1,-1,0],
                        [1,0,0,0,1],
                        [0,-0.5076,-0.3526,-0.2176,0],
                        [0.033,0,0,0,0],
                        [0,0,0,0,0]])
        # joint_angles, success = mr.IKinBody(
        #     B_list_arm, M, target_pose, initial_guess, eomg=1e-3, ev=1e-3
        # )
        
        # if success:
        #     return self.normalize_joint_angles(joint_angles.tolist())
        # else:
        #     print("IK failed to converge")
        #     return None
        T_base_world = self.get_base_transform()
        T_target_base = np.linalg.inv(T_base_world) @ target_pose_world
        # print(T_base_world,"\n", T_target_base)
        
        joint_angles, success = mr.IKinBody(
            B_list_arm, M, T_target_base,
            initial_guess,
            eomg=0.1, ev=0.05
        )
        if success:
            joint_angles = self.normalize_joint_angles(joint_angles)
            verify = self.forward_kinematics(joint_angles)
            # print("IK joint angles:", joint_angles)
            print("IK end-effector position:", verify)
            return joint_angles 
        else:
            return []
    ###------------------------------------------###

    def forward_kinematics(self, config):
        """Compute end-effector position in WORLD frame"""
        M = np.array([[1, 0, 0, 0.033],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0.6546],
                      [0, 0, 0, 1]])
        B_list_arm = np.array([[0 , 0 , 0 , 0 , 0],
                               [0 ,-1 ,-1 ,-1 ,-1],
                               [1 , 0 , 0 , 0 , 1],
                               [0 ,-0.5076 ,-0.3526 ,-0.2176 , 0],
                               [0.033 , 0 , 0 , 0 , 0],
                               [0 , 0 , 0 , 0 , 0]])

        T_ee_base = mr.FKinBody(M, B_list_arm, np.array(config))
        T_base_world = self.get_base_transform()
        T_ee_world = T_base_world @ T_ee_base
        return T_ee_world[:3, -1] 

    # def check_collision(self, config):
    #     """Check collision using CoppeliaSim's collision detection"""
    #     # Set arm joints to test configuration
    #     for i, angle in enumerate(config):
    #         self.sim.setJointPosition(self.arm_joints[i], angle)
        
    #     # Check collision between arm and obstacles
    #     collision = self.sim.checkCollision(self.arm_collection, self.obstacle_collection)
    #     return not collision  # Return True if collision-free
    
    def check_collision(self, config):
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
            min_x, max_x, min_y, max_y, min_z, max_z = np.array(obstacle['dimensions'])
            
            # Check if end-effector is within the bounding box
            if (min_x <= end_effector_pos[0] <= max_x and
                min_y <= end_effector_pos[1] <= max_y and
                min_z <= end_effector_pos[2] <= max_z):
                return False  # Collision detected
        
        return True  # No collision detected

    def get_obstacles_from_simulation(self):
        """Retrieve obstacle positions and dimensions from the simulation."""
        obstacles = []
        obstacle_names = [
            '/Obstacles/Cylinder[0]',
            '/Obstacles/Cylinder[1]',
            '/Obstacles/Cylinder[2]',
            '/Obstacles/Wall[0]',
            '/Obstacles/Wall[1]',
            '/Obstacles/Wall[2]',
            '/Obstacles/Wall[3]'
        ]
        
        try:
            for name in obstacle_names:
                handle = self.sim.getObject(name)
                if handle != -1:
                    # Get position in world coordinates
                    position = self.sim.getObjectPosition(handle, -1)

                    # CORRECTED: Use separate parameters for min/max
                    min_x = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_x)
                    max_x = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_x)
                    min_y = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_y)
                    max_y = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_y)
                    min_z = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_z)
                    max_z = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_z)

                    # Convert to world coordinates
                    world_min_x = position[0] + min_x
                    world_max_x = position[0] + max_x
                    world_min_y = position[1] + min_y
                    world_max_y = position[1] + max_y
                    world_min_z = position[2] + min_z
                    world_max_z = position[2] + max_z

                    obstacles.append({
                        'handle': handle,
                        'position': position,
                        'dimensions': [world_min_x, world_max_x,
                                    world_min_y, world_max_y,
                                    world_min_z, world_max_z]
                    })
            return obstacles
        except Exception as e:
            print(f"Error retrieving obstacles: {e}")
            return []


    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        q_from = np.array(from_node.q_values)
        q_to = np.array(to_node.q_values)
        distance = np.linalg.norm(q_to - q_from)
        direction = (q_to - q_from) / distance if distance > 0 else np.zeros_like(q_from)
        return distance, direction
        
    def get_random_node(self):
        """Generate random node within joint limits or sample goal directly"""
        if random.randint(0, 100) > self.goal_sample_rate:
            q_random = [random.uniform(joint[0], joint[1]) for joint in self.joint_limits]
            return self.Node(q_random)
        else:
            return self.Node(self.goal.q_values.copy())

    def get_nearest_node_index(self, rnd_node):
        dlist = [np.linalg.norm(np.array(node.q_values) - np.array(rnd_node.q_values))
                 for node in self.node_list]
        return dlist.index(min(dlist))

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.q_values.copy())
        d, direction = self.calc_distance_and_angle(from_node, to_node)

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)
        new_node.path_q_values = [new_node.q_values.copy()]

        for _ in range(n_expand):
            new_node.q_values = [q + self.path_resolution * dq 
                                 for q, dq in zip(new_node.q_values, direction)]
            new_node.path_q_values.append(new_node.q_values.copy())

        if np.linalg.norm(np.array(new_node.q_values) - np.array(to_node.q_values)) <= self.path_resolution:
            new_node.q_values = to_node.q_values.copy()
            new_node.path_q_values.append(to_node.q_values.copy())

        new_node.parent = from_node
        return new_node

    # def check_collision(self, node):
    #     """Placeholder for collision checking - always returns True"""
    #     return True

    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.q_values)
            node = node.parent
        path.append(node.q_values)
        # print(path[::-1])
        return path[::-1]
    
    def planning(self, start_config, goal_config):
        self.start = self.Node(start_config)
        self.goal = self.Node(goal_config)
        self.node_list = [self.start]
        
        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node.q_values):
                self.node_list.append(new_node)
                
                # Check if goal is reached
                if np.linalg.norm(np.array(new_node.q_values) - np.array(self.goal.q_values)) <= self.path_resolution:
                    # print("Goal reached. Terminating early.")
                    return self.generate_final_course(len(self.node_list) - 1)  # Exit immediately

        # Final check after loop
        last_node = self.node_list[-1]
        if np.linalg.norm(np.array(last_node.q_values) - np.array(self.goal.q_values)) <= self.expand_dis:
            return self.generate_final_course(len(self.node_list) - 1)
        
        return []


    # def visualize_trajectory(self, path, obstacles):
    #     """
    #     Visualize the RRT tree, end-effector trajectory, start & goal positions, and obstacles.
    #     """
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # --- Plot RRT Tree ---
    #     for node in self.node_list:
    #         if node.parent is not None:
    #             parent_pos = self.forward_kinematics(node.parent.q_values)
    #             current_pos = self.forward_kinematics(node.q_values)
    #             ax.plot([parent_pos[0], current_pos[0]],
    #                     [parent_pos[1], current_pos[1]],
    #                     [parent_pos[2], current_pos[2]],
    #                     color='gray', linewidth=0.5, alpha=0.6)

    #     # --- Plot final trajectory ---
    #     x_traj, y_traj, z_traj = [], [], []
    #     for config in path:
    #         pos = self.forward_kinematics(config)
    #         x_traj.append(pos[0])
    #         y_traj.append(pos[1])
    #         z_traj.append(pos[2])
    #     ax.plot(x_traj, y_traj, z_traj, label="Planned Trajectory", color='blue', linewidth=2)

    #     # --- Start and Goal ---
    #     start_pos = self.forward_kinematics(path[0])
    #     goal_pos = self.forward_kinematics(path[-1])
    #     ax.scatter(*start_pos, color='green', s=80, label="Start", marker='o')
    #     ax.scatter(*goal_pos, color='magenta', s=80, label="Goal", marker='x')

    #     # --- Obstacles ---
    #     if obstacles:
    #         obs_x = [obs[0] for obs in obstacles]
    #         obs_y = [obs[1] for obs in obstacles]
    #         obs_z = [obs[2] for obs in obstacles]
    #         ax.scatter(obs_x, obs_y, obs_z, label="Obstacles", color='red', s=50)

    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title("RRT Tree Expansion & End-Effector Path")
    #     ax.legend()
    #     plt.show()



# def main():
#     start_config = [0, 0, 0, 0, 0]
#     goal_config = [0.5, 0.3, -0.4, 0.2, 0.1]

#     rrt = YouBotRRT(start_config, goal_config, goal_sample_rate=10, max_iter=1000)

#     print("Planning path...")
#     start_time = time.time()
#     path = rrt.planning()
#     planning_time = time.time() - start_time

#     if path:
#         print(f"✅ Path found in {planning_time:.2f} seconds")
#         print(f"Path has {len(path)} waypoints")
#         obstacles = [[0.3, 0.2, 0.5], [0.2, -0.1, 0.6]]  # Dummy obstacles
#         rrt.visualize_trajectory(path, obstacles)
#     else:
#         print("❌ No path found.")

if __name__ == "__main__":
    print("This module is not meant to be run directly. It is intended for use within a simulation environment.")
