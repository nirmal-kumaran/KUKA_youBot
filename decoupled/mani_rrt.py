import numpy as np
import math
import random

class Node:
    def __init__(self, joint_angles):
        self.q = np.array(joint_angles)  # Joint angles (5-dimensional)
        self.cost = 0.0
        self.parent = None

class InformedRRTStar5DOF:

    def __init__(self, start, goal, joint_limits, obstacle_check_fn, link_length,
                 expand_dis=0.1, goal_sample_rate=10, max_iter=5000):
        
        self.start = Node(start)
        self.goal = Node(goal)
        self.joint_limits = joint_limits  # [(min1,max1), (min2,max2),..., (min5,max5)]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_check_fn = obstacle_check_fn
        self.node_list = None

        self.link_length = link_length # List of link lengths

    
    def convert_joint_to_cartesian_space(self, joint_angles):
    
        x = self.link_length[4] * np.cos(joint_angles[4] + joint_angles[3] + joint_angles[2] + joint_angles[1] + joint_angles[0]) 
        + self.link_length[3] * np.cos(joint_angles[3] + joint_angles[2] + joint_angles[1] + joint_angles[0]) 
        + self.link_length[2] * np.cos(joint_angles[2] + joint_angles[1] + joint_angles[0])
        + self.link_length[1] * np.cos(joint_angles[1] + joint_angles[0])
        + self.link_length[0] * np.cos(joint_angles[0])

        y = self.link_length[4] * np.sin(joint_angles[4] + joint_angles[3] + joint_angles[2] + joint_angles[1] + joint_angles[0]) 
        + self.link_length[3] * np.sin(joint_angles[3] + joint_angles[2] + joint_angles[1] + joint_angles[0]) 
        + self.link_length[2] * np.sin(joint_angles[2] + joint_angles[1] + joint_angles[0])
        + self.link_length[1] * np.sin(joint_angles[1] + joint_angles[0])
        + self.link_length[0] * np.sin(joint_angles[0])

        return [x, y]
    
    def informed_rrt_star_search(self):

        self.node_list = [self.start]
        c_best = float('inf')
        solution_node = None
        path = None

        c_min = np.linalg.norm(self.start.q - self.goal.q)
        q_center = (self.start.q + self.goal.q) / 2.0
        a1 = (self.goal.q - self.start.q) / c_min

        # Compute rotation matrix
        id1_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
        m = a1.reshape(5, 1) @ id1_t
        u, s, vh = np.linalg.svd(m, True, True)
        c = u @ np.diag([1.0, 1.0, 1.0, 1.0, np.linalg.det(u) * np.linalg.det(vh)]) @ vh

        for _ in range(self.max_iter):
            rnd = self.informed_sample(c_best, c_min, q_center, c)
            nearest_node = self.get_nearest_node(rnd)
            new_node = self.steer(nearest_node, rnd)

            if not self.check_collision(nearest_node.q, new_node.q):
                continue

            near_inds = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_inds)
            self.node_list.append(new_node)
            self.rewire(new_node, near_inds)

            if self.is_near_goal(new_node):
                temp_cost = new_node.cost + np.linalg.norm(new_node.q - self.goal.q)
                if temp_cost < c_best:
                    solution_node = new_node
                    c_best = temp_cost

        return self.extract_path(solution_node)

    def informed_sample(self, c_best, c_min, x_center, c):
        if c_best < float('inf'):
            r = np.array([c_best / 2.0, math.sqrt(c_best**2 - c_min**2) / 2.0, 1, 1, 1])
            L = np.diag(r)
            rnd = np.random.randn(5) @ L @ c + x_center
        else:
            rnd = np.array([random.uniform(lim[0], lim[1]) for lim in self.joint_limits])
        return rnd 
    
    def get_nearest_node(self, rnd_joint):
        distances = [np.linalg.norm(node.q - rnd_joint) for node in self.node_list]
        return self.node_list[np.argmin(distances)]
    
    def steer(self, from_node, to_joint):
        direction = to_joint - from_node.q
        distance = np.linalg.norm(direction)
        direction_unit = direction / distance if distance != 0 else np.zeros(5)
        new_q = from_node.q + self.expand_dis * direction_unit
        new_q = np.clip(new_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
        new_node = Node(new_q)
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.expand_dis
        return new_node

    def check_collision(self, q_start, q_end):
        return self.obstacle_check_fn(q_start, q_end)

    def find_near_nodes(self, new_node):
        r = min(10.0 * math.sqrt(math.log(len(self.node_list)) / len(self.node_list)), self.expand_dis * 2.0)
        return [i for i, node in enumerate(self.node_list) if np.linalg.norm(node.q - new_node.q) <= r]

    def choose_parent(self, new_node, near_inds):
        best_parent = None
        min_cost = float('inf')
        for i in near_inds:
            node = self.node_list[i]
            cost = node.cost + np.linalg.norm(node.q - new_node.q)
            if cost < min_cost and self.check_collision(node.q, new_node.q):
                min_cost = cost
                best_parent = node
        if best_parent:
            new_node.parent = best_parent
            new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            potential_cost = new_node.cost + np.linalg.norm(new_node.q - near_node.q)
            if near_node.cost > potential_cost and self.check_collision(near_node.q, new_node.q):
                near_node.parent = new_node
                near_node.cost = potential_cost

    def is_near_goal(self, node):
        return np.linalg.norm(node.q - self.goal.q) < self.expand_dis

    def extract_path(self, solution_node):
        path = []
        node = solution_node
        while node is not None:
            path.append(node.q)
            node = node.parent
        return path[::-1]

def no_obstacle_check(q_start, q_end):
    return True  # Always return True (no obstacles for now)

start_q = [0.0, 0.0, 0.0, 0.0, 0.0]  # Initial joint angles
goal_q = [np.pi/2, np.pi/4, np.pi/6, np.pi/3, np.pi/8]  # Target joint angles

joint_limits = [
    (-np.pi, np.pi),  # Joint 1 limits
    (-np.pi, np.pi),  # Joint 2 limits
    (-np.pi, np.pi),  # Joint 3 limits
    (-np.pi, np.pi),  # Joint 4 limits
    (-np.pi, np.pi)   # Joint 5 limits
]

link_lengths = [1.0, 1.0, 1.0, 1.0, 1.0]  # Example link lengths

planner = InformedRRTStar5DOF(start_q, goal_q, joint_limits, no_obstacle_check, link_lengths)

path = planner.informed_rrt_star_search()

if path:
    print("Path found!")
    for i, joint_config in enumerate(path):
        print(f"Step {i}: {joint_config}")
else:
    print("No path found.")




        







    