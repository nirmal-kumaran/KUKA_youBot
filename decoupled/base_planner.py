import numpy as np
from queue import PriorityQueue
import math


class AStar:
    def __init__(self, occupancy_grid, map_dims, cell_size, heuristic=0):
        self.maze = occupancy_grid
        self.heuristic = heuristic
        self.map_dims = map_dims
        self.cell_size = cell_size

    @staticmethod
    def get_neighbors(maze, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] != 1:
                neighbors.append((nx, ny))
        # print(neighbors)
        return neighbors

    @staticmethod
    def manhattan_heuristic(node, goal):
        return abs(goal[0] - node[0]) + abs(goal[1] - node[1])

    @staticmethod
    def euclidean_heuristic(node, goal):
        return np.linalg.norm(np.array(goal) - np.array(node))
    
    @staticmethod
    def waypoints_theta(astar_path):
        """
        Compute orientation (theta) for each waypoint in an A* path.
        
        Args:
            astar_path: List of waypoints [(x1, y1), (x2, y2), ...].
        
        Returns:
            List of waypoints with orientation [(x1, y1, theta1), (x2, y2, theta2), ...].
        """
        base_waypoints = []
        for i in range(len(astar_path) - 1):
            # Current and next waypoints
            x_current, y_current = astar_path[i]
            x_next, y_next = astar_path[i + 1]
            
            # Compute change in x and y
            delta_x = x_next - x_current
            delta_y = y_next - y_current
            
            # Calculate theta using atan2
            angle_rad = np.arctan2(delta_y, delta_x)
            theta = (angle_rad + math.pi) % (2 * math.pi) - math.pi
            # Append current waypoint with theta
            base_waypoints.append((x_current, y_current, theta))
        
        # Handle last waypoint (same theta as previous or task-defined)
        base_waypoints.append((astar_path[-1][0], astar_path[-1][1], base_waypoints[-1][2]))
        print(f"Start Position - x:{base_waypoints[0][0]}, y:{base_waypoints[0][1]}, theta{base_waypoints[0][2]}")
        print(f"Goal Position - x:{base_waypoints[-1][0]}, y:{base_waypoints[-1][1]}, theta{base_waypoints[-1][2]}")
       
        return base_waypoints

    def astar(self, start, goal):
        print("Start Position:", start)
        print("Goal Position:", goal)

        path = []
        fringe = PriorityQueue()
        closed_set = set()
        parent_node = {start: None}
        g_cost = {start: 0}
        f_cost = {}

        if self.heuristic == 0:
            heuristic_cost = self.manhattan_heuristic(start, goal)
        elif self.heuristic == 1:
            heuristic_cost = self.euclidean_heuristic(start, goal)
        elif self.heuristic >= 2:
            heuristic_cost = self.heuristic * self.manhattan_heuristic(start, goal)

        f_cost[start] = g_cost[start] + heuristic_cost
        fringe.put((f_cost[start], start))

        while not fringe.empty():
            _, current_node = fringe.get()

            if current_node in closed_set:
                continue

            closed_set.add(current_node)

            if current_node == goal:
                break

            for neighbor in self.get_neighbors(self.maze, current_node):
                if neighbor in closed_set:
                    continue

                actual_cost = g_cost[current_node] + 1

                if self.heuristic == 0:
                    heuristic_cost = self.manhattan_heuristic(neighbor, goal)
                elif self.heuristic == 1:
                    heuristic_cost = self.euclidean_heuristic(neighbor, goal)
                elif self.heuristic >= 2:
                    heuristic_cost = self.heuristic * self.manhattan_heuristic(neighbor, goal)

                f_cost_neighbor = actual_cost + heuristic_cost

                if neighbor not in g_cost or actual_cost < g_cost[neighbor]:
                    g_cost[neighbor] = actual_cost
                    f_cost[neighbor] = f_cost_neighbor
                    fringe.put((f_cost_neighbor, neighbor))
                    parent_node[neighbor] = current_node

        if goal in parent_node:
            node = goal
            while node is not None:
                path.append(node)
                node = parent_node[node]
            return path[::-1]
        
        return path
    
    def astar_trajectory(self, start, goal):
        """
        Convert a list of waypoints in grid indices to world coordinates.
        Args:
            path: List of waypoints [(i, j)] in grid indices.
            grid: OccupancyGrid object.
        Returns:
            List of waypoints [(x, y)] in world coordinates.
        """
        min_x, max_x, min_y, max_y = self.map_dims
        cell_size = self.cell_size
        astar_path = self.astar(start, goal)

        base_coordinates = []
        for i, j in astar_path:
            x = min_x + (j + 0.5) * cell_size  # Convert column index (j) to x-coordinate
            y = max_y - (i + 0.5) * cell_size  # Convert row index (i) to y-coordinate
            base_coordinates.append([x, y])

        coordinates_theta = self.waypoints_theta(base_coordinates)

        return coordinates_theta, astar_path
    
if __name__ == "__main__":
    print("This is just an A* class implementation. It cannot be run independently.")
    
    
