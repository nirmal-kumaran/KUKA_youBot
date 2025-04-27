from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from occupancy_grid import OccupancyGrid
from base_planner import AStar
from manipulator_planner import InformedRRTStar
from controller_test import YouBotController
import numpy as np
from scipy.interpolate import BSpline

class SimulationManager:
    def __init__(self, port=23000):
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.require('sim')
        self.running = False

    def start_simulation(self):
        if not self.running:
            self.sim.startSimulation()
            self.running = True
            print("Simulation started.")

            # Compute base trajectory using A*
            occupancy_grid = OccupancyGrid(self.sim)
            maze = occupancy_grid.grid
            start = occupancy_grid.start_position
            payload = occupancy_grid.payload_position
            goal = occupancy_grid.goal_position
            
            astar = AStar(maze, occupancy_grid.map_dims, occupancy_grid.cell_size, heuristic=1)

            start_payload_theta, start_payload = astar.astar_trajectory(start, payload)
            payload_goal_theta, payload_goal = astar.astar_trajectory(payload, goal)

            base_path_theta, base_path = start_payload_theta + payload_goal_theta[1:], start_payload + payload_goal[1:]
            # occupancy_grid.visualize(path=base_path)
            self.visualize_path_in_sim(base_path_theta)
            print("Base path waypoints:", base_path_theta)
            controller = YouBotController(sim_manager.sim)
            controller.follow_path(base_path_theta)
            controller.set_arm_position_youbot(0)

    #         # Compute manipulator trajectory using InformedRRTStar
    #         manipulator = InformedRRTStar(self.sim)
    #         cube_payload_handle = self.sim.getObject('/Cuboid_initial')
    #         cube_payload_position = self.sim.getObjectPosition(cube_payload_handle, -1)
            
    #         Tsc_i = np.array([[1, 0, 0, cube_payload_position[0]],
    #                         [0, 1, 0, cube_payload_position[1]],
    #                         [0, 0, 1, cube_payload_position[2]],
    #                         [0, 0, 0,                     1]])
    #         Tce_s = np.array([[np.cos(3*np.pi/4), 0, np.sin(3*np.pi/4),       0],
    #                         [               0,   1,               0,       0],
    #                         [-np.sin(3*np.pi/4),   0, np.cos(3*np.pi/4),   .15],
    #                         [               0,   0,               00]])
    #         Tce_g = np.array([[np.cos(3*np.pi/4), 0, np.sin(3*np.pi/4), 0],
    #                   [0, 1, 0, 0],
    #                   [-np.sin(3*np.pi/4), 0, np.cos(3*np.pi/4), 0],
    #                   [0, 0,  0, 1]])
            
    #          # Retrieve pose of Cuboid_goal dynamically
    #         cuboid_goal_handle = self.sim.getObject('/Cuboid_goal')
    #         cuboid_goal_position = self.sim.getObjectPosition(cuboid_goal_handle, -1)  # [x, y, z]
    #         cuboid_goal_orientation = self.sim.getObjectOrientation(cuboid_goal_handle, -1)  # [roll, pitch, yaw]
    #         R = self.euler_to_rotation_matrix(cuboid_goal_orientation)
    #         Tsc_f = np.eye(4)
    #         Tsc_f[:3, :3] = R
    #         Tsc_f[:3, 3] = cuboid_goal_position
            
    #         waypoint1 = Tsc_i@Tce_s
    #         waypoint2 = Tsc_i@Tce_g
    #         waypoint3 = Tsc_i@Tce_g
    #         waypoint4 = Tsc_i@Tce_s
    #         waypoint5 = Tsc_f@Tce_s
    #         waypoint6 = Tsc_f@Tce_g
    #         waypoint7 = Tsc_f@Tce_g
    #         waypoint8 = Tsc_f@Tce_s
    #         m_se = manipulator.plan(waypoint1)
    #         print("Manipulator path waypoints:", m_se)
    #         m_ge = manipulator.plan(waypoint2, waypoint1)

    #         # Get current robot position from simulation
    # def get_Tbaseworld(self):
    #     robot_handle = self.sim.getObject('/youBot')
    #     current_position = self.sim.getObjectPosition(robot_handle, -1)[:2]  # Get x, y coordinates
        
    #     # Find waypoints within MAX_ARM_EXTENSION of target
    #     reachable_waypoints = []
        
    #     for i, waypoint in enumerate(base_path_theta):
    #         x, y, theta = waypoint
            
    #         # Calculate distance from waypoint to target
    #         distance_to_target = np.sqrt((x - cube_payload_position[0])**2 + (y - cube_payload_position[1])**2)
            
    #         if distance_to_target <= 0.5:
    #             # Calculate distance from waypoint to current position
    #             distance_to_robot = np.sqrt((x - current_position[0])**2 + (y - current_position[1])**2)
    #             reachable_waypoints.append((i, waypoint, distance_to_target, distance_to_robot))
        
    #     if reachable_waypoints:
    #         # Sort by distance to robot (closest first)
    #         reachable_waypoints.sort(key=lambda x: x[3])
            
    #         closest_idx, closest_waypoint, distance_to_target, distance_to_robot = reachable_waypoints[0]
            
    #         print(f"Found closest reachable waypoint: {closest_waypoint} (index {closest_idx})")
    #         print(f"Distance to target: {distance_to_target:.3f}m")
    #         print(f"Distance to robot: {distance_to_robot:.3f}m")
            
    #         # Here you would compute IK for this waypoint
    #         x_base, y_base, theta_base = closest_waypoint
            
    #         # Transform for IK calculation
    #         T_base_world = self.get_base_transformation(x_base, y_base, theta_base)
    #         # Now you can use this for IK calculations
            
    #     else:
    #         print("No waypoints within arm's reach of target")
            
    # def get_base_transformation(self, x, y, theta):
    #     """Calculate the transformation matrix for the robot base."""
    #     T_base_world = np.eye(4)
    #     T_base_world[:3, :3] = [
    #         [np.cos(theta), -np.sin(theta), 0],
    #         [np.sin(theta), np.cos(theta), 0],
    #         [0, 0, 1]
    #     ]
    #     T_base_world[:3, 3] = [x, y, 0.0963]  # Base height
    #     return T_base_world

    def visualize_path_in_sim(self, path_theta):
        # Store handles to remove them later
        self.drawing_handles = []
        
        # Set the height slightly above the floor
        floor_height = 0.001
        
        # Create point container
        point_handle = self.sim.addDrawingObject(
            self.sim.drawing_points,  # objectType
            8,                        # size
            0,                        # duplicateTolerance
            -1,                       # parentObjectHandle
            len(path_theta),          # maxItemCount
            [0, 1, 0]                 # color (green)
        )
        self.drawing_handles.append(point_handle)
        
        # Create line container
        line_handle = self.sim.addDrawingObject(
            self.sim.drawing_lines,   # objectType
            4,                        # size
            0,                        # duplicateTolerance
            -1,                       # parentObjectHandle
            len(path_theta)-1,        # maxItemCount
            [1, 0, 0]                 # color (red)
        )
        self.drawing_handles.append(line_handle)
        
        # Add points and line segments
        for i, waypoint in enumerate(path_theta):
            x, y, _ = waypoint
            # Add point
            self.sim.addDrawingObjectItem(point_handle, [x, y, floor_height])
            
            # Add line segment (connecting current point to next point)
            if i < len(path_theta) - 1:
                next_x, next_y, _ = path_theta[i+1]
                self.sim.addDrawingObjectItem(line_handle, [x, y, floor_height, next_x, next_y, floor_height])



    @staticmethod
    def euler_to_rotation_matrix(euler_angles):
        """
        Convert Euler angles (roll, pitch, yaw) to a rotation matrix.

        Args:
            euler_angles: List or array of [roll, pitch, yaw] in radians.

        Returns:
            Rotation matrix (3x3).
        """
        roll, pitch, yaw = euler_angles

        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        # Combined rotation matrix
        R = R_z @ R_y @ R_x
        return R

    def stop_simulation(self):
        if self.running:
            self.sim.stopSimulation()
            self.running = False
            print("Simulation stopped.")

if __name__ == "__main__":
    sim_manager = SimulationManager()
    sim_manager.start_simulation()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        sim_manager.stop_simulation()
