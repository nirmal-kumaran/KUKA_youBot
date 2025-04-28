import numpy as np
import math
import random
import time
import modern_robotics as mr

random.seed(42)

class PlannerManipulator:
    """Base class for RRT-based planners"""
    class Node:
        def __init__(self, q_values):
            self.q_values = q_values  # Joint angles
            self.parent = None
            self.cost = 0.0  # For RRT*

    def __init__(self, sim, obstacles=None):
        self.sim = sim
        
        self.arm_joints = [
            sim.getObject('/youBotArmJoint0'),
            sim.getObject('/youBotArmJoint1'),
            sim.getObject('/youBotArmJoint2'),
            sim.getObject('/youBotArmJoint3'),
            sim.getObject('/youBotArmJoint4')
        ]
        
        self.obstacles = obstacles or self.get_obstacles_from_simulation()
        self.joint_limits = [
            (-np.deg2rad(169), np.deg2rad(169)),
            (-np.deg2rad(65), np.deg2rad(90)),
            (-np.deg2rad(150), np.deg2rad(146)),
            (-np.deg2rad(102.5), np.deg2rad(102.5)),
            (-np.deg2rad(167.5), np.deg2rad(167.5))
        ]

        self.arm_collection = sim.createCollection(0)
        for joint in self.arm_joints:
            sim.addItemToCollection(self.arm_collection, sim.handle_single, joint, 0)
        
        self.obstacle_collection = sim.createCollection(0)
        for obs in self.obstacles:
            sim.addItemToCollection(self.obstacle_collection, sim.handle_single, obs['handle'], 0)

    def measure_time_for_iterations(self, start_config, goal_config, iterations_list):
        """Measure planning time for different iteration counts"""
        results = {}
        
        original_max_iter = self.max_iter
        original_goal_sample_rate = self.goal_sample_rate
        
        for max_iter in iterations_list:
            self.max_iter = max_iter
            self.goal_sample_rate = 15
            
            _ = self.planning(start_config, goal_config)
            
            start_time = time.time()
            path = self.planning(start_config, goal_config)
            elapsed_time = time.time() - start_time
            
            results[max_iter] = {
                'time': elapsed_time,
                'success': len(path) > 0,
                'path_length': len(path)
            }
        
        self.max_iter = original_max_iter
        self.goal_sample_rate = original_goal_sample_rate
        
        return results

    def get_obstacles_from_simulation(self):
        """
        Retrieve obstacle positions and dimensions from the simulation.
        """
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
                    position = self.sim.getObjectPosition(handle, -1)

                    min_x = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_x)
                    max_x = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_x)
                    min_y = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_y)
                    max_y = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_y)
                    min_z = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_min_z)
                    max_z = self.sim.getObjectFloatParam(handle, self.sim.objfloatparam_objbbox_max_z)

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
        
    ###------------------------------------------###
    def inverse_kinematics(self, target_pose_world, base_position=(0,0,0.0963), base_orientation=0, initial_guess=None):
        """
        Solve inverse kinematics to compute joint angles for a given end-effector pose in bodyframe.
        """
        self.base_position, self.base_orientation = np.array(base_position), base_orientation
        if initial_guess is None:
            initial_guess = [0, 0 , np.pi/4, 0, 0]  
        
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

        T_base_world = self.get_base_transform()
        T_target_base = np.linalg.inv(T_base_world) @ target_pose_world
        # print(T_base_world,"\n", T_target_base)
        
        joint_angles, success = mr.IKinBody(
            B_list_arm, M, T_target_base,
            initial_guess,
            eomg=0.01, ev=0.05
        )
        if success:
            joint_angles = self.normalize_joint_angles(joint_angles)
            verify = self.forward_kinematics(joint_angles)
            print("IK end-effector position:", verify)
            return joint_angles 
        else:
            return []

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
    
    def forward_kinematics_full(self, config):
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
        return T_ee_world
    
    def check_collision(self, config):
        """
        Check if a given joint configuration is collision-free.
        """
        end_effector_pos = self.forward_kinematics(config)
        
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle['position'])
            min_x, max_x, min_y, max_y, min_z, max_z = np.array(obstacle['dimensions'])
            if (min_x <= end_effector_pos[0] <= max_x and
                min_y <= end_effector_pos[1] <= max_y and
                min_z <= end_effector_pos[2] <= max_z):
                return False
        
        return True

    def get_random_node(self):
        """Generate random node within joint limits or sample goal directly"""
        if random.randint(0, 100) > self.goal_sample_rate:
            q_random = [random.uniform(joint[0], joint[1]) for joint in self.joint_limits]
            return self.Node(q_random)
        else:
            return self.Node(self.goal.q_values.copy())
    
    def get_nearest_node_index(self, rnd_node):
        # return np.argmin([
        #     self.pose_distance(n.q_values, rnd_node.q_values)
        #     for n in self.node_list
        # ])
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

    def get_base_transform(self):
        """Compute base's transformation matrix in world frame"""
        x, y, z = self.base_position
        return np.array([
            [np.cos(self.base_orientation), -np.sin(self.base_orientation), 0, x],
            [np.sin(self.base_orientation), np.cos(self.base_orientation), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    def pose_distance(self, q1_values, q2_values):
        """Compare closeness based on end-effector pose"""
        p1 = self.forward_kinematics(q1_values)  # Pose of node q1
        p2 = self.forward_kinematics(q2_values)  # Pose of node q2
        
        # Position difference
        position_diff = np.linalg.norm(p1 - p2)
        
        # Orientation difference (using Frobenius norm)
        # orientation_diff = np.linalg.norm(R1 - R2)
        
        # Weighted combination (adjust weights based on task)
        return position_diff #+ 0.5 * orientation_diff
    
    def calc_dist_to_goal(self, current_q):
        """Calculate distance to goal in joint configuration space"""
        q_diff = np.array(current_q) - np.array(self.goal.q_values)
        return np.linalg.norm(q_diff)
    
    def is_goal_reached(self, node):
        """Check if node satisfies both position and orientation constraints"""
        current_pose = self.forward_kinematics_full(node.q_values)
        pos_error = np.linalg.norm(current_pose[:3, -1] - self.goal[:3, -1])
        orient_error = np.linalg.norm(current_pose[:3, :3] - self.goal[:3, :3])
        return pos_error < 0.01 and orient_error < 0.1
    
    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.q_values)
            node = node.parent
        path.append(node.q_values)
        # print(path[::-1])
        return path[::-1]
    
    @staticmethod
    def normalize_joint_angles(joint_angles):
        """
        Normalize joint angles to the range [-π, π].
        """
        return [(angle + np.pi) % (2 * np.pi) - np.pi for angle in joint_angles]
    
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        q_from = np.array(from_node.q_values)
        q_to = np.array(to_node.q_values)
        distance = np.linalg.norm(q_to - q_from)
        direction = (q_to - q_from) / distance if distance > 0 else np.zeros_like(q_from)
        return distance, direction
    ###------------------------------------------###

    def get_ee_positions(self, path):
        """
        Convert joint configurations to end-effector positions
        """
        ee_positions = []
        for config in path:
            position = self.forward_kinematics(config)
            ee_positions.append(position.tolist()) #.tolist()
        return ee_positions
    
class RRT(PlannerManipulator):
    """Basic RRT Implementation"""
    def __init__(self, sim, obstacles=None):
        super().__init__(sim, obstacles)  # Proper parent init
        self.goal_sample_rate = 10
        self.path_resolution = 0.1
        self.max_iter = 500
        self.expand_dis = 0.5

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
                # if self.is_goal_reached(new_node):
                    return self.generate_final_course(len(self.node_list) - 1)  # Exit immediately

        # Final check after loop
        last_node = self.node_list[-1]
        if np.linalg.norm(np.array(last_node.q_values) - np.array(self.goal.q_values)) <= self.expand_dis:
        # if self.is_goal_reached(last_node):
            return self.generate_final_course(len(self.node_list) - 1)
        
        return []

class RRTStar(PlannerManipulator):
    """RRT* implementation with cost optimization"""
    def __init__(self, sim, obstacles=None):
        super().__init__(sim, obstacles)
        self.expand_dis=0.1
        self.path_resolution=0.01
        self.goal_sample_rate=15
        self.max_iter=1000
        self.connect_circle_dist=0.05
        self.robot_radius=0.0
        # self.search_until_max_iter = True
        
    def planning(self, start_config, goal_config):
        self.start = self.Node(start_config)
        self.goal = self.Node(goal_config)
        self.node_list = [self.start]

        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd_node)
            new_node = self.steer(self.node_list[nearest_ind], rnd_node, self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + np.linalg.norm(np.array(new_node.q_values) - np.array(near_node.q_values))
            # print(new_node.q_values, new_node.cost)
            if self.check_collision(new_node.q_values):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            # if ((not self.search_until_max_iter)
            #             and new_node):
            #         last_index = self.search_best_goal_node()
            #         if last_index is not None:
            #             return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return []

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node.q_values):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.q_values) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal)
            if self.check_collision(t_node.q_values):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        safe_goal_costs = [self.node_list[i].cost +
                           self.calc_dist_to_goal(self.node_list[i].q_values)
                           for i in safe_goal_inds]

        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [np.linalg.norm(np.array(node.q_values) - np.array(new_node.q_values))
                    for node in self.node_list] 
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree

                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node.q_values)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)