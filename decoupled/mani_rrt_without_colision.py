#RRT without collision on a 5dof manipulator

# rrt = RRT(
#         start=[0, 0],
#         goal=[gx, gy],
#         rand_area=[-2, 15],
#         obstacle_list=obstacleList,
#         # play_area=[0, 10, 0, 14]
#         robot_radius=0.8
#         )
#     path = rrt.planning(animation=show_animation)

import numpy as np
import random
import math

class RRT:

    class Node:
            """
            RRT Node
            """

            def __init__(self, theta_config):
                
                self.q_values = theta_config

                self.path_q_values = []

                self.parent = None

    # class AreaBounds:

    #     def __init__(self, area):
    #         self.xmin = float(area[0])
    #         self.xmax = float(area[1])
    #         self.ymin = float(area[2])
    #         self.ymax = float(area[3])
    #         self.ymin = float(area[4])
    #         self.ymax = float(area[5])


    def __init__(self, start_config_list, goal_config_list, goal_sample_rate, play_area = None, path_resolution = 0.017, max_iter=500):
        
        self.start = self.Node(start_config_list)
        self.goal = self.Node(goal_config_list)


        self.joint_limits = [
            (-(169/180)*np.pi, (169/180)*np.pi),     # Joint 0
            (-(65/180)*np.pi, (90/180)*np.pi),       # Joint 1
            (-(150/180)*np.pi, (146/180)*np.pi),     # Joint 2
            (-(102.5/180)*np.pi, (102.5/180)*np.pi), # Joint 3
            (-(167.5/180)*np.pi, (167.5/180)*np.pi)  # Joint 4
        ]
        self.goal_sample_rate = goal_sample_rate  # % probability of sampling goal
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.path_resolution = path_resolution
    def get_random_node(self, goal_config=None):
        """Generates a random node within joint limits or samples the goal."""
        if random.randint(0, 100) > self.goal_sample_rate:
            # Sample within joint limits
            q_random = [random.uniform(joint[0], joint[1]) for joint in self.joint_limits]
            rnd = self.Node(q_random)
        else:
            # Sample goal directly
            rnd = self.Node(goal_config)  # `goal_config` should be a list of 5 joint values
        
        return rnd
    
    def get_nearest_node_index(self, node_list, rnd_node):
    
        dlist = [np.linalg.norm(np.array(node.q_values) - np.array(rnd_node.q_values))
                for node in node_list]
        
        return dlist.index(min(dlist))

    def calc_distance_and_angle(from_node, to_node):
        """
        Computes the Euclidean distance and direction (unit vector) in a 5-DOF joint space.

        :param from_node: Node with q_values (list of 5 joint angles)
        :param to_node: Node with q_values (list of 5 joint angles)
        :return: (distance, unit direction vector)
        """
        # Convert to numpy arrays for vector operations
        q_from = np.array(from_node.q_values)
        q_to = np.array(to_node.q_values)
        
        # Compute Euclidean distance in joint space
        distance = np.linalg.norm(q_to - q_from)
        
        # Compute unit direction vector (joint-wise change direction)
        if distance > 0:
            direction = (q_to - q_from) / distance  # Normalize to get unit vector
        else:
            direction = np.zeros_like(q_from)  # No movement if nodes are identical

        return distance, direction
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Moves from `from_node` towards `to_node` in 5D joint space, up to `extend_length`.

        :param from_node: Start node (with q_values list)
        :param to_node: Target node (with q_values list)
        :param extend_length: Max step length in joint space
        :return: New node extended in the direction of `to_node`
        """
        
        new_node = self.Node(from_node.q_values)  # Copy joint angles
        d, direction = self.calc_distance_and_angle(from_node, to_node)  # Compute distance & direction
        
        # Limit the extension length
        if extend_length > d:
            extend_length = d
        
        # Compute number of small steps to take
        n_expand = math.floor(extend_length / self.path_resolution)

        new_node.path_q_values = [new_node]  # Store path in joint space

        for _ in range(n_expand):
            # Move each joint incrementally in the direction of `to_node`
            new_node.q_values = [q + self.path_resolution * dq for q, dq in zip(new_node.q_values, direction)]
            new_node.path_q_values.append(new_node.q_values)  # Save intermediate configurations

        # If close enough to `to_node`, snap to it
        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.q_values = to_node.q_values
            new_node.path_q_values.append(to_node.q_values)

        new_node.parent = from_node  # Link parent node

        return new_node

# def check_if_outside_play_area(node, play_area):

#         if play_area is None:
#             return True  # no play_area was defined, every pos should be ok

#         if node.x < play_area.xmin or node.x > play_area.xmax or \
#            node.y < play_area.ymin or node.y > play_area.ymax:
#             return False  # outside - bad
#         else:
#             return True  # inside - ok

    def check_collision(self):
        return True

    def generate_final_course(self, goal_ind):
            
            path = [[self.goal.q_values]]
            node = self.node_list[goal_ind]
            while node.parent is not None:
                path.append([node.q_values])
                node = node.parent
            path.append([node.q_values])

            return path

    def planning(self):
        """
        rrt path planning

        animation: flag for animation on or off
        """
        self.node_list = [self.start.q_values]

        for i in range(self.max_iter):
            # Step 1: Sample a random node (or goal)
            rnd_node = self.get_random_node(self.goal.q_values)

            # Step 2: Find the nearest node in the tree
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # Step 3: Steer towards the random node within `expand_dis`
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # Step 4: Collision checking in joint space
            if self.check_collision():
                self.node_list.append(new_node)  # Add valid node to the tree

            # Step 5: Check if close enough to goal
            if np.linalg.norm(np.array(new_node.q_values) - np.array(self.goal.q_values)) <= self.path_resolution:
                final_node = self.steer(new_node, self.goal, self.path_resolution)
                if self.check_collision():
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # No path found