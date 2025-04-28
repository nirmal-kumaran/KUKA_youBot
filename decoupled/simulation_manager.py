from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from occupancy_grid import OccupancyGrid
from base_planner import AStar
from manipulator_planner import RRT, RRTStar
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

            manipulator_start_theta = [[0.0006325064846133266, 0.6121231757197148, 0.9676419239432787, 1.301388960041053, 0.00041225662731125823], [-0.04736809952361489, 0.6175543075652783, 0.9190760519728081, 1.2471593592051335, -0.04823906506741384], [0.0043375677292135165, 0.5843348227692005, 0.8893492396429469, 1.1830011820704076, -0.08321097864498772], [0.0712204120928803, 0.5618927126412736, 0.9108123724654469, 1.13120965529174, -0.12657004351278447], [0.02791550306476124, 0.5269794401027031, 0.9238369543721187, 1.0511425007297734, -0.14460776180586135], [0.05574264591832589, 0.4906882552718341, 0.868025134436245, 1.0085437528789443, -0.19918786313561426], [0.12488473759154747, 0.4710963932847113, 0.8304828365675889, 1.0146785061376626, -0.2573984358533355], [0.12055211368547145, 0.5014708260983167, 0.7781610394702249, 0.9353740654335551, -0.2517541968035417], [0.07664216575445085, 0.527739197407198, 0.7592790824370304, 0.8760706137398943, -0.31098646383164086], [0.07323475882727762, 0.5579518961462623, 0.7066436251135488, 0.7969799838837304, -0.30393215208360885], [0.09771311962602469, 0.5741974785312239, 0.665388562320044, 0.7118988498698411, -0.3179361277474828], [0.09996183883443986, 0.6020900916384249, 0.7007283748655755, 0.6371896911195251, -0.26908344466824474], [0.1788680152392123, 0.6128085165911544, 0.7244364755661096, 0.5932263323451396, -0.3032020574501214], [0.1574794627050275, 0.6522402437214097, 0.7804035355000267, 0.5250743047392193, -0.2886879935267146], [0.15188376958974437, 0.681611650133623, 0.7235058759079978, 0.448789488876449, -0.2816671740830192], [0.10069140230324342, 0.6463159734084952, 0.71106523384541, 0.38950244350208674, -0.2320301575331887], [0.09630681126808642, 0.6769147917319193, 0.6538060728977837, 0.3137959455180747, -0.22616480765145716], [0.03915943460349937, 0.6788843732116616, 0.5973407211788273, 0.27837523701064576, -0.2739909046408286], [-0.03147736404891466, 0.6467504935103873, 0.6039550170286422, 0.24263278941626437, -0.32553297465995346], [-0.01277878279441879, 0.603067759616379, 0.5736959265002848, 0.2202766633020731, -0.405073988509271], [-0.014405903339162357, 0.6362432260031478, 0.5173188691779372, 0.1454124653409473, -0.39441002207724574], [-0.016033023883905925, 0.6694186923899165, 0.46094181185558974, 0.07054826737982148, -0.38374605564522046], [-0.017660144428649494, 0.7025941587766852, 0.40456475453324225, -0.004315930581304334, -0.3730820892131952], [-0.019287264973393062, 0.7357696251634539, 0.3481876972108947, -0.07918012854243013, -0.3624181227811699], [-0.05075730360585633, 0.7717973469424823, 0.2758817496195387, -0.029549192760862812, -0.35791700200449456], [-0.051469347736825294, 0.8040640280113261, 0.22137382144880247, -0.10619981476758736, -0.34733767317544506], [-0.05218139186779426, 0.83633070908017, 0.16686589327806622, -0.1828504367743119, -0.33675834434639557], [-0.057664281413385615, 0.8697390372627399, 0.16421532186947263, -0.2735935748056111, -0.31201003325822063], [-0.058220167012698665, 0.9018039600088842, 0.10832882558601759, -0.3494011079349131, -0.3019378351618052], [-0.058776052612011714, 0.9338688827550284, 0.05244232930256256, -0.4252086410642151, -0.2918656370653898], [-0.11439848546703704, 0.9238873763329366, 0.030843256794636353, -0.4381609597447338, -0.37042919095922516], [-0.1130504940103979, 0.956397460088017, -0.02448308586232383, -0.5137686907621718, -0.35763293401159185], [-0.11170250255375877, 0.9889075438430973, -0.079809428519284, -0.5893764217796098, -0.34483667706395854], [-0.18823356540793273, 1.0095160732588226, -0.1298470007399379, -0.624188832224839, -0.346456053174504], [-0.2097536120134198, 0.945025974429111, -0.18855392617001793, -0.6675341569593549, -0.3537082581659588], [-0.20461217447783475, 0.980438222488019, -0.2418180682686247, -0.7430127731756483, -0.3400903299808435], [-0.1994707369422497, 1.0158504705469271, -0.2950822103672315, -0.8184913893919417, -0.3264724017957282], [-0.19432929940666466, 1.0512627186058352, -0.3483463524658382, -0.893970005608235, -0.31285447361061286], [-0.1891878618710796, 1.0866749666647433, -0.401610494564445, -0.9694486218245284, -0.29923654542549755], [-0.2368630062233989, 1.0214969437452512, -0.4446155258839086, -0.9805193681424643, -0.2604175836095558], [-0.166534203049918, 1.0191066196305414, -0.5058286392190692, -0.9664798461721147, -0.22719120292382228], [-0.1623456127334579, 1.0581938065858942, -0.5551952093928972, -1.0433332138992262, -0.21664847342183666], [-0.1581570224169978, 1.097280993541247, -0.6045617795667253, -1.1201865816263377, -0.20610574391985104], [-0.1539684321005377, 1.1363681804966, -0.6539283497405534, -1.1970399493534492, -0.19556301441786542], [-0.1497798417840776, 1.1754553674519528, -0.7032949199143814, -1.2738933170805606, -0.1850202849158798], [-0.11659814988716313, 1.1526823834601123, -0.7912822046925484, -1.2500973056619036, -0.1765141647136657], [-0.11424866173856973, 1.1934067361586909, -0.8361139479532275, -1.3289831706351587, -0.1663582744842814], [-0.11189917358997634, 1.2341310888572694, -0.8809456912139065, -1.4078690356084138, -0.15620238425489708], [-0.10954968544138294, 1.274855441555848, -0.9257774344745856, -1.486754900581669, -0.14604649402551276], [-0.10720019729278954, 1.3155797942544265, -0.9706091777352647, -1.565640765554924, -0.13589060379612844], [-0.10485070914419614, 1.356304146953005, -1.0154409209959439, -1.6445266305281792, -0.12573471356674412], [-0.10250122099560274, 1.3970284996515836, -1.060272664256623, -1.7234124955014343, -0.1155788233373598], [-0.10015173284700934, 1.4377528523501621, -1.105104407517302, -1.8022983604746894, -0.10542293310797549], [-0.09780224469841595, 1.478477205048741, -1.149936150777981, -1.8811842254479445, -0.09526704287859117], [-0.09545275654982256, 1.5192015577473197, -1.1947678940386601, -1.9600700904211996, -0.08511115264920685], [-0.09310326840122918, 1.5599259104458982, -1.2395996372993392, -2.0389559553944547, -0.07495526241982253], [-0.09075378025263578, 1.6006502631444768, -1.2844313805600183, -2.11784182036771, -0.06479937219043823], [-0.08840429210404238, 1.6413746158430556, -1.3292631238206973, -2.196727685340965, -0.05464348196105392], [-0.086054803955449, 1.682098968541634, -1.3740948670813764, -2.27561355031422, -0.04448759173166961], [-0.08370531580685561, 1.7228233212402126, -1.4189266103420555, -2.354499415287475, -0.0343317015022853], [-0.08135582765826221, 1.7635476739387914, -1.4637583536027345, -2.43338528026073, -0.024175811272901], [-0.07900633950966883, 1.80427202663737, -1.5085900968634136, -2.512271145233985, -0.014019921043516702], [-0.07505341300939561, 1.8727892366880194, -1.5840178388993467, -2.6449936815602246, 0.0030669868626302943]]
            controller.follow_path(start_payload_theta, manipulator_start_theta)
            controller.follow_path(payload_goal_theta, manipulator_payload_theta)

            controller.set_arm_position_youbot([0,0, 0,0,0])

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
