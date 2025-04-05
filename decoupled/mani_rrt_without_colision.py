import numpy as np
import random
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class YouBotRRT:
    
    class Node:
        """RRT Node for 5DOF configuration"""
        def __init__(self, theta_config):
            self.q_values = theta_config  # List of 5 joint angles
            self.path_q_values = []  # For storing intermediate configurations
            self.parent = None

    def __init__(self, start_config, goal_config, goal_sample_rate=10, 
                 path_resolution=0.017, max_iter=500, expand_dis=0.5):
        """
        Initialize RRT planner for YouBot 5DOF arm
        
        Parameters:
        start_config: list of 5 joint angles (radians)
        goal_config: list of 5 joint angles (radians)
        goal_sample_rate: % probability to sample goal directly
        path_resolution: step size for steering (radians)
        max_iter: maximum iterations
        expand_dis: maximum expansion distance
        """
        self.start = self.Node(start_config)
        self.goal = self.Node(goal_config)
        self.goal_sample_rate = goal_sample_rate
        self.path_resolution = path_resolution
        self.max_iter = max_iter
        self.expand_dis = expand_dis
        self.node_list = [self.start]
        
        # KUKA YouBot joint limits (radians)
        self.joint_limits = [
            (-(169/180)*np.pi, (169/180)*np.pi),     # Joint 0
            (-(65/180)*np.pi, (90/180)*np.pi),       # Joint 1
            (-(150/180)*np.pi, (146/180)*np.pi),     # Joint 2
            (-(102.5/180)*np.pi, (102.5/180)*np.pi), # Joint 3
            (-(167.5/180)*np.pi, (167.5/180)*np.pi)  # Joint 4
        ]

    def get_random_node(self):
        """Generate random node within joint limits or sample goal directly"""
        if random.randint(0, 100) > self.goal_sample_rate:
            # Random configuration within joint limits
            q_random = [random.uniform(joint[0], joint[1]) for joint in self.joint_limits]
            return self.Node(q_random)
        else:
            # Sample goal configuration
            return self.Node(self.goal.q_values.copy())

    def get_nearest_node_index(self, rnd_node):
        """Find nearest node in the tree to the random node"""
        dlist = [np.linalg.norm(np.array(node.q_values) - np.array(rnd_node.q_values))
                for node in self.node_list]
        return dlist.index(min(dlist))

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """Calculate distance and direction between two nodes in joint space"""
        q_from = np.array(from_node.q_values)
        q_to = np.array(to_node.q_values)
        distance = np.linalg.norm(q_to - q_from)
        direction = (q_to - q_from) / distance if distance > 0 else np.zeros_like(q_from)
        return distance, direction

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Steer from from_node toward to_node in joint space"""
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

    def check_collision(self, node):
        """Placeholder for collision checking - always returns True for now"""
        return True

    def generate_final_course(self, goal_ind):
        """Generate the final path from goal to start"""
        path = [self.goal.q_values]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.q_values)
            node = node.parent
        path.append(node.q_values)
        return path[::-1]  # Reverse to get start->goal

    def planning(self):
        """Main RRT planning algorithm"""
        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node):
                self.node_list.append(new_node)

            if np.linalg.norm(np.array(new_node.q_values) - np.array(self.goal.q_values)) <= self.path_resolution:
                final_node = self.steer(new_node, self.goal, self.path_resolution)
                if self.check_collision(final_node):
                    self.node_list.append(final_node)
                    return self.generate_final_course(len(self.node_list) - 1)
        return None  # No path found

def test_rrt_planner():
    
    # Define start and goal configurations (in radians)
    start_config = [0, 0, 0, 0, 0]
    goal_config = [0.5, 0.3, -0.4, 0.2, 0.1]  # Example goal
    
    # Create RRT planner
    rrt_planner = YouBotRRT(start_config, goal_config, 
                           goal_sample_rate=10, 
                           max_iter=1000)
    
    # Plan path
    print("Planning path...")
    start_time = time.time()
    path = rrt_planner.planning()
    planning_time = time.time() - start_time
    
    if path:
        print(f"Path found in {planning_time:.2f} seconds")
        print(f"Path length: {len(path)} waypoints")
        
    else:
        print("No path found")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    import time
    test_rrt_planner()