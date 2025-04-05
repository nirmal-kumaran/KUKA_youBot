import numpy as np
import random
import math
import modern_robotics as mr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class RRT:
    class Node:
        def __init__(self, q_values):
            self.q_values = np.array(q_values)  # Joint configuration [J0-J4]
            self.parent = None

    def __init__(self, start_config, goal_config, joint_limits, 
                 goal_sample_rate=10, path_resolution=0.1, max_iter=1000):
        self.start = self.Node(start_config)
        self.goal = self.Node(goal_config)
        self.joint_limits = np.array(joint_limits)
        self.goal_sample_rate = goal_sample_rate
        self.path_resolution = path_resolution
        self.max_iter = max_iter
        self.node_list = [self.start]
        
        # Modern Robotics FK parameters
        self.M = np.array([[1, 0, 0, 0.033],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0.6546],
                          [0, 0, 0, 1]])
        self.B_list = np.array([[0, 0, 0, 0, 0],
                               [0, -1, -1, -1, -1],
                               [1, 0, 0, 0, 1],
                               [0, -0.5076, -0.3526, -0.2176, -0.2176],
                               [0.033, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]]).T

    def plan(self, visualizer=None, simulator=None):
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if self.check_collision(new_node):
                self.node_list.append(new_node)
                
                # Update visualization
                if visualizer and i % 50 == 0:
                    visualizer.update_tree(self.node_list)
                    visualizer.update_robot(new_node.q_values)
                
                # Check goal
                if self.is_goal_reached(new_node):
                    final_path = self.generate_path(new_node)
                    if visualizer:
                        visualizer.show_path(final_path)
                    if simulator:
                        simulator.execute_path(final_path)
                    return final_path

        return None

    def get_random_node(self):
        if random.random() < self.goal_sample_rate/100:
            return self.Node(self.goal.q_values)
        return self.Node(np.random.uniform(
            low=self.joint_limits[:,0], 
            high=self.joint_limits[:,1]
        ))

    def get_nearest_node(self, node):
        distances = [np.linalg.norm(n.q_values - node.q_values) 
                    for n in self.node_list]
        return self.node_list[np.argmin(distances)]

    def steer(self, from_node, to_node):
        direction = to_node.q_values - from_node.q_values
        distance = np.linalg.norm(direction)
        step = min(distance, self.path_resolution)
        new_q = from_node.q_values + (direction/distance) * step
        
        new_node = self.Node(new_q)
        new_node.parent = from_node
        return new_node

    def check_collision(self, node):
        # TODO: Implement actual collision checking
        return True

    def is_goal_reached(self, node):
        return np.linalg.norm(node.q_values - self.goal.q_values) < self.path_resolution

    def generate_path(self, node):
        path = []
        while node:
            path.append(node.q_values.tolist())
            node = node.parent
        return path[::-1]

    def forward_kinematics(self, config):
        """Calculate end-effector position using Modern Robotics library"""
        T = mr.FKinBody(self.M, self.B_list, np.array(config))
        return T[:3, 3]

class Visualizer:
    def __init__(self, joint_limits):
        self.fig = plt.figure(figsize=(12, 6))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122)
        
        # Workspace settings
        self.ax1.set_xlim([-1, 1])
        self.ax1.set_ylim([-1, 1])
        self.ax1.set_zlim([0, 1.5])
        
        # Configuration space settings
        self.ax2.set_xlim(joint_limits[0])
        self.ax2.set_ylim(joint_limits[1])
        
        self.tree_lines = []
        self.robot_line = None

    def update_tree(self, nodes):
        # Clear previous tree
        for line in self.tree_lines:
            line.remove()
        self.tree_lines = []
        
        # Plot tree
        for node in nodes:
            if node.parent:
                x = [node.parent.q_values[0], node.q_values[0]]
                y = [node.parent.q_values[1], node.q_values[1]]
                self.tree_lines.append(self.ax2.plot(x, y, 'gray', alpha=0.3)[0])
        
        # Plot start/goal
        self.tree_lines.append(self.ax2.plot(
            self.start.q_values[0], self.start.q_values[1], 'go')[0])
        self.tree_lines.append(self.ax2.plot(
            self.goal.q_values[0], self.goal.q_values[1], 'ro')[0])
        
        plt.draw()
        plt.pause(0.001)

    def update_robot(self, config):
        # Clear previous robot
        if self.robot_line:
            self.robot_line.remove()
        
        # Calculate joint positions
        joint_pos = []
        T = self.M.copy()
        joint_pos.append(T[:3,3])
        for i in range(5):
            T = T @ mr.MatrixExp6(mr.VecTose3(self.B_list[:,i]*config[i]))
            joint_pos.append(T[:3,3])
        joint_pos = np.array(joint_pos)
        
        # Plot robot
        self.robot_line = self.ax1.plot3D(
            joint_pos[:,0], joint_pos[:,1], joint_pos[:,2], 
            'o-', color='blue')[0]
        
        plt.draw()
        plt.pause(0.001)

class YouBotSimulator:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.joints = [self.sim.getObject(f'/youBotArmJoint{i}') for i in range(5)]
        self.sim.startSimulation()
        self.sim.setStepping(True)

    def execute_path(self, path, delay=0.1):
        for q in path:
            for i, joint in enumerate(self.joints):
                self.sim.setJointTargetPosition(joint, q[i])
            self.sim.step()
            time.sleep(delay)

def main():
    # Joint limits (radians)
    joint_limits = [
        (-2.9496, 2.9496),  # J0
        (-1.1345, 1.5708),  # J1
        (-2.6179, 2.6179),  # J2
        (-1.7890, 1.7890),  # J3
        (-2.9234, 2.9234)   # J4
    ]

    # Initialize components
    visualizer = Visualizer(joint_limits)
    simulator = YouBotSimulator()
    
    # Create RRT planner
    rrt = RRT(
        start_config=[0, 0, 0, 0, 0],
        goal_config=[0.5, -0.3, 0.4, 0.2, 0],
        joint_limits=joint_limits,
        goal_sample_rate=10,
        max_iter=1000
    )

    # Plan and execute
    path = rrt.plan(visualizer=visualizer, simulator=simulator)
    
    if path:
        print(f"Path found with {len(path)} waypoints")
        plt.show()
    else:
        print("No path found")

    input("Press Enter to exit...")
    simulator.sim.stopSimulation()

if __name__ == "__main__":
    import time
    main()