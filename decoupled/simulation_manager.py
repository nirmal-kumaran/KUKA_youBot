from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from occupancy_grid import OccupancyGrid
from base_planner import AStar
# from manipulator_planner import RRT
# from controller import YouBotController
# from manipulator_planner2 import RRT 
from manipulator_planner3 import RRT, RRTStar
import numpy as np
import matplotlib.pyplot as plt
import time
# from scipy.interpolate import BSpline

class SimulationManager:
    def __init__(self, port=23000):
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.require('sim')
        self.running = False

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

        R = R_z @ R_y @ R_x
        return R

    def get_current_arm_config(self):
        """
        Retrieve current arm configuration.

        Returns:
            Angle list (1x5).
        """
        arm_joint_handles = [
            self.sim.getObject('/youBotArmJoint0'),
            self.sim.getObject('/youBotArmJoint1'),
            self.sim.getObject('/youBotArmJoint2'),
            self.sim.getObject('/youBotArmJoint3'),
            self.sim.getObject('/youBotArmJoint4')
        ]
        return [self.sim.getJointPosition(joint) for joint in arm_joint_handles]

    def start_simulation(self):
        if not self.running:
            self.sim.startSimulation()
            self.running = True
            print("Simulation started.")
            self.base_drawing_handles = []
            self.manipulator_drawing_handles = []

            ###----------------------------------###
            ### Base Planner
            occupancy_grid = OccupancyGrid(self.sim)
            maze = occupancy_grid.grid
            start = occupancy_grid.start_position
            payload = occupancy_grid.payload_position
            goal = occupancy_grid.goal_position

            print("Planning base path")
            base_planner = AStar(maze, occupancy_grid.map_dims, occupancy_grid.cell_size, heuristic=1)
            start_payload_theta, start_payload = base_planner.astar_trajectory(start, payload)
            payload_goal_theta, payload_goal = base_planner.astar_trajectory(payload, goal)
            if start_payload_theta and payload_goal_theta:
                base_path_theta, base_path = start_payload_theta + payload_goal_theta[1:], start_payload + payload_goal[1:]
                occupancy_grid.visualize(path=base_path)
                self.visualize_basepath_in_sim(base_path_theta)

            ### Manipulator Planner
            if start_payload_theta:
                standoff_waypoint = np.array(start_payload_theta[-2])
                print("Standoff waypoint:", standoff_waypoint)

                cube_payload_handle = self.sim.getObject('/Cuboid_initial')
                cube_payload_position = np.array(self.sim.getObjectPosition(cube_payload_handle, -1))
                cube_payload_orientation = self.sim.getObjectOrientation(cube_payload_handle, -1)
                R = self.euler_to_rotation_matrix(cube_payload_orientation)


                print("Payload position:", cube_payload_position)

                num_points = 10
                z_value = 0.0963  
                points = [
                    np.append(standoff_waypoint[:2] + t * (cube_payload_position[:2] - standoff_waypoint[:2]), z_value)
                    for t in np.linspace(0, 1, num_points)
                ]
                print(f"Generated {len(points)} points along the line.")

                print("Planning manipulator path")
                manipulator_planner = RRT(self.sim)
                current_arm_config = self.get_current_arm_config()

                # R = np.array([
                #     [1, 0, 0],
                #     [0, 1, 0],
                #     [0, 0, 1],
                # ])
                Tsc_i = np.eye(4)
                Tsc_i[:3, :3] = R  # Set rotation part
                Tsc_i[:3, 3] = cube_payload_position  # Set position part
                # Tsc_i = np.array([
                #     [1, 0, 0, payload_position[0]],
                #     [0, 1, 0, payload_position[1]],
                #     [0, 0, 1, z_value],
                #     [0, 0, 0, 1]
                # ])

                Tce_s = np.array([
                    [np.cos(3*np.pi/4), 0, np.sin(3*np.pi/4), 0],
                    [0, 1, 0, 0],
                    [-np.sin(3*np.pi/4), 0, np.cos(3*np.pi/4), 0.15],
                    [0, 0, 0, 1]
                ])
                
                waypoint1 = Tsc_i @ Tce_s

                # print(Tsc_i)
                # print(Tce_s)
                # print(waypoint1)

                for i, point in enumerate(points):
                    print(f"Testing point {i+1}/{len(points)}: {point}")

                    # Compute orientation facing the payload
                    dx, dy = cube_payload_position[0] - point[0], cube_payload_position[1] - point[1]
                    theta = np.arctan2(dy, dx)

                    # Try IK with current base position
                    goal_config = manipulator_planner.inverse_kinematics(
                        waypoint1,
                        base_position=(point[0], point[1], point[2]),  # Use z from constant value
                        base_orientation=theta
                    )

                    if goal_config and len(goal_config) > 0:
                        print(f"IK angles: {goal_config}")
                        print(f"✅ Feasible IK solution found at point {point} with orientation {theta:.2f} radians.")

                        # Plan arm trajectory to the feasible configuration
                        start_time = time.time()
                        path = manipulator_planner.planning(current_arm_config, goal_config)
                        elapsed_time = time.time() - start_time
                        if path:
                            ee_positions = manipulator_planner.get_ee_positions(path)
                            self.visualize_manipulatorpath_in_sim(ee_positions)
                            print(f"Successfully planned trajectory with {path} points, {elapsed_time} sec.")
                            
                            # Run timing tests with ACTUAL configurations
                            self.run_timing_tests(
                                manipulator_planner,
                                start_config=[0.00, 0.64, 0.99, 1.32, 0.00],
                                goal_config=[0.14, -1.56, 0.50, -2.87, 3.14]
                            )
                            break
                        else:
                            print("Arm path planning failed.")
                            break
                    if i == len(points) - 1:
                        print("❌ No feasible IK solution found along the line.")
            # if start_payload_theta:
            #     manipulator_planner = RRT(self.sim)
            #     print("Planning manipulator path")

            #     cube_payload_handle = self.sim.getObject('/Cuboid_initial')
            #     cube_payload_position = self.sim.getObjectPosition(cube_payload_handle, -1)

            #     Tsc_i = np.array([
            #         [1, 0, 0, cube_payload_position[0]],
            #         [0, 1, 0, cube_payload_position[1]],
            #         [0, 0, 1, cube_payload_position[2]],
            #         [0, 0, 0, 1]
            #     ])

            #     # Tce_s = np.array([
            #     #     [np.cos(3*np.pi/4), 0, np.sin(3*np.pi/4), 0],
            #     #     [0, 1, 0, 0],
            #     #     [-np.sin(3*np.pi/4), 0, np.cos(3*np.pi/4), 0.15],
            #     #     [0, 0, 0, 1] 
            #     # ])

            #     Tce_s = np.array([
            #         [1, 0, 0, 0],
            #         [0, 1, 0, 0],
            #         [0, 0, 1, 0.15],
            #         [0, 0, 0, 1] 
            #     ])

            #     Tce_g = np.array([[np.cos(3*np.pi/4), 0, np.sin(3*np.pi/4), 0],
            #             [0, 1, 0, 0],
            #             [-np.sin(3*np.pi/4), 0, np.cos(3*np.pi/4), 0],
            #             [0, 0,  0, 1]])
                
            #     # Retrieve pose of Cuboid_goal dynamically
            #     cuboid_goal_handle = self.sim.getObject('/Cuboid_goal')
            #     cuboid_goal_position = self.sim.getObjectPosition(cuboid_goal_handle, -1)
            #     cuboid_goal_orientation = self.sim.getObjectOrientation(cuboid_goal_handle, -1)
            #     R = self.euler_to_rotation_matrix(cuboid_goal_orientation)
            #     Tsc_f = np.eye(4)
            #     Tsc_f[:3, :3] = R
            #     Tsc_f[:3, 3] = cuboid_goal_position
                
            #     waypoint1 = Tsc_i@Tce_s
            #     base_payloadstdoff_pose = start_payload_theta[-2] 
            #     print("Base payload standoff pose:", base_payloadstdoff_pose)

                # waypoint2 = Tsc_i@Tce_g
                # waypoint3 = Tsc_i@Tce_g
                # waypoint4 = Tsc_i@Tce_s
                # waypoint5 = Tsc_f@Tce_s
                # waypoint6 = Tsc_f@Tce_g
                # waypoint7 = Tsc_f@Tce_g
                # waypoint8 = Tsc_f@Tce_s

                # print(Tsc_i)
                # print(Tce_s)
            #     print("Waypoint1:","\n", waypoint1)
            #     current_arm_config = self.get_current_arm_config()
            #     goal_config = manipulator_planner.inverse_kinematics(waypoint1, base_position=(base_payloadstdoff_pose[0], base_payloadstdoff_pose[1], 0.0963),
            #     base_orientation=base_payloadstdoff_pose[2], initial_guess = None)
            #     print("Goal configuration:", goal_config)
            #     if len(goal_config):
            #         path = manipulator_planner.planning(current_arm_config, goal_config)
                    
            #         if path:
            #             ee_positions = manipulator_planner.get_ee_positions(path)
            #             self.visualize_manipulatorpath_in_sim(ee_positions)
            #             print(f"Successfully planned standoff trajectory with {len(path)} waypoints")
            #         else:
            #             print("Arm path planning failed")
            #     else:
            #         print("IK failed for standoff position")
            # else:
            #     print("Base path planning failed")
                ###----------------------------------###

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

    def visualize_basepath_in_sim(self, path_theta):
        """Visualize base path without clearing manipulator drawings"""
        for handle in self.base_drawing_handles:
            self.sim.removeDrawingObject(handle)
        self.base_drawing_handles = []

        floor_height = 0.001
        point_handle = self.sim.addDrawingObject(
            self.sim.drawing_points,  # objectType
            8,                        # size
            0,                        # duplicateTolerance
            -1,                       # parentObjectHandle
            len(path_theta),          # maxItemCount
            [0, 1, 0]                 # color (green)
        )
        self.base_drawing_handles.append(point_handle)

        line_handle = self.sim.addDrawingObject(
            self.sim.drawing_lines,   # objectType
            4,                        # size
            0,                        # duplicateTolerance
            -1,                       # parentObjectHandle
            len(path_theta)-1,        # maxItemCount
            [1, 0, 0]                 # color (red)
        )
        self.base_drawing_handles.append(line_handle)

        for i, waypoint in enumerate(path_theta):
            x, y, _ = waypoint
            self.sim.addDrawingObjectItem(point_handle, [x, y, floor_height])
            if i < len(path_theta) - 1:
                next_x, next_y, _ = path_theta[i+1]
                self.sim.addDrawingObjectItem(line_handle, 
                    [x, y, floor_height, next_x, next_y, floor_height])

    def visualize_manipulatorpath_in_sim(self, ee_positions):
        """Visualize manipulator path without clearing base drawings"""
        for handle in self.manipulator_drawing_handles:
            self.sim.removeDrawingObject(handle)
        self.manipulator_drawing_handles = []

        # Create drawing objects for regular path points and lines
        points_handle = self.sim.addDrawingObject(
            self.sim.drawing_points,
            8,  # Point size
            0,  # Duplicate tolerance
            -1,  # Parent handle
            len(ee_positions),
            [0, 0, 1]  # Blue color
        )
        lines_handle = self.sim.addDrawingObject(
            self.sim.drawing_lines,
            4,  # Line width
            0,  # Duplicate tolerance
            -1,  # Parent handle
            len(ee_positions)-1,
            [1, 0, 1]  # Magenta color
        )
        
        # Create special drawing objects for initial and final positions
        start_handle = self.sim.addDrawingObject(
            self.sim.drawing_points,
            16,  # Larger point size
            0,  # Duplicate tolerance
            -1,  # Parent handle
            1,  # Only one point
            [0, 1, 0]  # Green color for initial position
        )
        end_handle = self.sim.addDrawingObject(
            self.sim.drawing_points,
            16,  # Larger point size
            0,  # Duplicate tolerance
            -1,  # Parent handle
            1,  # Only one point
            [1, 0, 0]  # Red color for final position
        )

        # Add points and lines
        prev_pos = None
        for i, pos in enumerate(ee_positions):
            # Add regular point
            self.sim.addDrawingObjectItem(points_handle, pos)
            
            # Add line to previous point if exists
            if prev_pos is not None:
                self.sim.addDrawingObjectItem(lines_handle, prev_pos + pos)
            prev_pos = pos
        
        # Add special markers for start and end positions
        if len(ee_positions) > 0:
            self.sim.addDrawingObjectItem(start_handle, ee_positions[0])
            self.sim.addDrawingObjectItem(end_handle, ee_positions[-1])

        # Track all handles for cleanup
        self.manipulator_drawing_handles.extend([points_handle, lines_handle, start_handle, end_handle])
        print(f"Visualized manipulator path with {len(ee_positions)} points")

    # def visualize_manipulatorpath_in_sim(self, ee_poses):
    #     """Visualize manipulator path with end-effector positions and orientations"""
    #     # Clear previous drawings
    #     for handle in self.manipulator_drawing_handles:
    #         self.sim.removeDrawingObject(handle)
    #     self.manipulator_drawing_handles = []

    #     # Create drawing objects for path
    #     points_handle = self.sim.addDrawingObject(
    #         self.sim.drawing_points, 8, 0, -1, len(ee_poses), [0, 0, 1]
    #     )
    #     lines_handle = self.sim.addDrawingObject(
    #         self.sim.drawing_lines, 4, 0, -1, len(ee_poses)-1, [1, 0, 1]
    #     )
        
    #     # Special markers for start/end
    #     start_handle = self.sim.addDrawingObject(
    #         self.sim.drawing_points, 16, 0, -1, 1, [0, 1, 0]
    #     )
    #     end_handle = self.sim.addDrawingObject(
    #         self.sim.drawing_points, 16, 0, -1, 1, [1, 0, 0]
    #     )

    #     # Create single objects for orientation axes
    #     x_axis_handle = self.sim.addDrawingObject(
    #         self.sim.drawing_lines, 4, 0, -1, len(ee_poses), [1, 0, 0]
    #     )
        
    #     # Create a more prominent forward direction vector (Z-axis for typical grippers)
    #     forward_handle = self.sim.addDrawingObject(
    #         self.sim.drawing_lines, 6, 0, -1, len(ee_poses), [0, 0, 1]
    #     )
        
    #     # Add path points and orientation vectors
    #     prev_pos = None
    #     for i, pose in enumerate(ee_poses):
    #         # Convert pose to NumPy array if it's not already
    #         if not isinstance(pose, np.ndarray):
    #             pose = np.array(pose)
                
    #         pos = pose[:3, 3].flatten() if pose.ndim > 1 else pose[:3]
    #         pos_list = pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
            
    #         # Add position point
    #         self.sim.addDrawingObjectItem(points_handle, pos_list)
            
    #         # Add line connecting to previous point
    #         if prev_pos:
    #             self.sim.addDrawingObjectItem(lines_handle, prev_pos + pos_list)
    #         prev_pos = pos_list
            
    #         # Add orientation vectors (if pose contains rotation matrix)
    #         if pose.ndim > 1 and pose.shape[0] >= 3 and pose.shape[1] >= 3:
    #             # Forward direction (Z-axis for standard gripper orientation)
    #             forward_dir = pose[:3, 2]  # Z-axis column of rotation matrix
    #             forward_end = pos + forward_dir * 0.08  # Make forward vector longer
    #             self.sim.addDrawingObjectItem(
    #                 forward_handle, 
    #                 pos_list + forward_end.tolist()
    #             )
                
    #             # X-axis (typically points to the right of gripper)
    #             x_dir = pose[:3, 0]  # X-axis column
    #             x_end = pos + x_dir * 0.08
    #             self.sim.addDrawingObjectItem(
    #                 x_axis_handle,
    #                 pos_list + x_end.tolist()
    #             )

    #     # Add start/end markers
    #     if len(ee_poses) > 0:
    #         start_pos = ee_poses[0][:3, 3].flatten() if isinstance(ee_poses[0], np.ndarray) and ee_poses[0].ndim > 1 else ee_poses[0][:3]
    #         end_pos = ee_poses[-1][:3, 3].flatten() if isinstance(ee_poses[-1], np.ndarray) and ee_poses[-1].ndim > 1 else ee_poses[-1][:3]
            
    #         self.sim.addDrawingObjectItem(start_handle, start_pos.tolist() if isinstance(start_pos, np.ndarray) else list(start_pos))
    #         self.sim.addDrawingObjectItem(end_handle, end_pos.tolist() if isinstance(end_pos, np.ndarray) else list(end_pos))

    #     # Track all drawing objects
    #     self.manipulator_drawing_handles = [points_handle, lines_handle, start_handle, end_handle, x_axis_handle, forward_handle]
    #     print(f"Visualized path with {len(ee_poses)} points including orientation vectors")


    def run_timing_tests(self, manipulator_planner, start_config, goal_config):
        """
        Run timing tests with ACTUAL configurations from simulation.
        
        Args:
            start_config: Current arm configuration from get_current_arm_config()
            goal_config: Successful IK solution from planning phase
        """
        
        # Define test parameters
        iterations_list = [100, 200, 300, 400, 500]
        
        print("\n=== Running Timing Tests with Real Configurations ===")
        print(f"Start Config: {[f'{x:.2f}' for x in start_config]}")
        print(f"Goal Config: {[f'{x:.2f}' for x in goal_config]}")
        
        results = manipulator_planner.measure_time_for_iterations(
            start_config, goal_config, iterations_list
        )
        
        # Visualize results
        self.visualize_timing_results(results)

    def visualize_timing_results(self, results):
        """Visualize timing results with success/failure info and path length."""
        
        iterations = list(results.keys())
        times = [v['time'] for v in results.values()]
        successes = [v['success'] for v in results.values()]
        path_lengths = [v['path_length'] for v in results.values()]

        plt.figure(figsize=(12, 8))

        # Plot time vs iterations (Top row, full width)
        plt.subplot(2, 1, 1)
        plt.plot(iterations, times, 'bo-')
        plt.xlabel('Iteration Count')
        plt.ylabel('Time (seconds)')
        plt.title('Planning Time vs Iterations')
        plt.grid(True)

        # Plot success rate (Bottom row, left column)
        plt.subplot(2, 2, 3)
        plt.bar(iterations, [int(s) for s in successes], width=20)
        plt.xlabel('Iteration Count')
        plt.ylabel('Success (1=Yes, 0=No)')
        plt.title('Planning Success vs Iterations')
        plt.ylim(0, 1)
        plt.grid(True)

        # Plot path length (Bottom row, right column)
        plt.subplot(2, 2, 4)
        plt.plot(iterations, path_lengths, 'gs-')
        plt.xlabel('Iteration Count')
        plt.ylabel('Path Length (nodes)')
        plt.title('Path Length vs Iterations')
        plt.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def stop_simulation(self):
        """Stop simulation and clear all drawings"""
        for handle in self.base_drawing_handles:
            self.sim.removeDrawingObject(handle)
        self.base_drawing_handles = []
        for handle in self.manipulator_drawing_handles:
            self.sim.removeDrawingObject(handle)
        self.manipulator_drawing_handles = []

        self.sim.stopSimulation()
        self.running = False
        print("Simulation stopped. All drawings cleared.")

if __name__ == "__main__":
    sim_manager = SimulationManager()
    
    sim_manager.start_simulation()
    
    try:
        # sim_manager.run_timing_tests()  # Run timing tests after simulation starts
        while True:
            pass
    except KeyboardInterrupt:
        sim_manager.stop_simulation()
