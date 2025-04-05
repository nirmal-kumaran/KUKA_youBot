import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import modern_robotics as mr

class YouBotRRT:
    class Node:
        """RRT Node for 5DOF configuration"""
        def __init__(self, theta_config):
            self.q_values = theta_config  # List of 5 joint angles
            self.path_q_values = []       # Intermediate configurations
            self.parent = None

    def __init__(self, start_config, goal_config, goal_sample_rate=10, 
                 path_resolution=0.017, max_iter=500, expand_dis=0.5):
        
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

    def forward_kinematics(self, config):
        """Compute end-effector position from joint configuration using FK logic"""
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
        T_ee = mr.FKinBody(M, B_list_arm, np.array(config))
        return T_ee[:3, -1]

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

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        q_from = np.array(from_node.q_values)
        q_to = np.array(to_node.q_values)
        distance = np.linalg.norm(q_to - q_from)
        direction = (q_to - q_from) / distance if distance > 0 else np.zeros_like(q_from)
        return distance, direction

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

    def check_collision(self, node):
        """Placeholder for collision checking - always returns True"""
        return True

    def generate_final_course(self, goal_ind):
        path = [self.goal.q_values]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.q_values)
            node = node.parent
        path.append(node.q_values)
        return path[::-1]

    def planning(self):
        """Main RRT planning loop"""
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
        return None

    def visualize_trajectory(self, path, obstacles):
        """
        Visualize the RRT tree, end-effector trajectory, start & goal positions, and obstacles.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # --- Plot RRT Tree ---
        for node in self.node_list:
            if node.parent is not None:
                parent_pos = self.forward_kinematics(node.parent.q_values)
                current_pos = self.forward_kinematics(node.q_values)
                ax.plot([parent_pos[0], current_pos[0]],
                        [parent_pos[1], current_pos[1]],
                        [parent_pos[2], current_pos[2]],
                        color='gray', linewidth=0.5, alpha=0.6)

        # --- Plot final trajectory ---
        x_traj, y_traj, z_traj = [], [], []
        for config in path:
            pos = self.forward_kinematics(config)
            x_traj.append(pos[0])
            y_traj.append(pos[1])
            z_traj.append(pos[2])
        ax.plot(x_traj, y_traj, z_traj, label="Planned Trajectory", color='blue', linewidth=2)

        # --- Start and Goal ---
        start_pos = self.forward_kinematics(path[0])
        goal_pos = self.forward_kinematics(path[-1])
        ax.scatter(*start_pos, color='green', s=80, label="Start", marker='o')
        ax.scatter(*goal_pos, color='magenta', s=80, label="Goal", marker='x')

        # --- Obstacles ---
        if obstacles:
            obs_x = [obs[0] for obs in obstacles]
            obs_y = [obs[1] for obs in obstacles]
            obs_z = [obs[2] for obs in obstacles]
            ax.scatter(obs_x, obs_y, obs_z, label="Obstacles", color='red', s=50)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("RRT Tree Expansion & End-Effector Path")
        ax.legend()
        plt.show()



def main():
    start_config = [0, 0, 0, 0, 0]
    goal_config = [0.5, 0.3, -0.4, 0.2, 0.1]

    rrt = YouBotRRT(start_config, goal_config, goal_sample_rate=10, max_iter=1000)

    print("Planning path...")
    start_time = time.time()
    path = rrt.planning()
    planning_time = time.time() - start_time

    if path:
        print(f"✅ Path found in {planning_time:.2f} seconds")
        print(f"Path has {len(path)} waypoints")
        obstacles = [[0.3, 0.2, 0.5], [0.2, -0.1, 0.6]]  # Dummy obstacles
        rrt.visualize_trajectory(path, obstacles)
    else:
        print("❌ No path found.")

if __name__ == "__main__":
    main()
