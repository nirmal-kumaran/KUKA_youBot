import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class YouBotController:
    def __init__(self, sim, dt=0.01):
        """Initialize controller for the Kuka YouBot."""
        
        self.sim = sim
        self.dt = dt

        self.youbot_handle = sim.getObject('/youBot')

        self.wheel_joints = [
            '/rollingJoint_fl', '/rollingJoint_fr', 
            '/rollingJoint_rl', '/rollingJoint_rr'
        ]
        self.wheel_handles = [sim.getObject(joint) for joint in self.wheel_joints]
        
        # Get arm joint handles (not used in path following)
        self.arm_joints = [
            '/youBotArmJoint0', '/youBotArmJoint1', 
            '/youBotArmJoint2', '/youBotArmJoint3', 
            '/youBotArmJoint4'
        ]

        self.arm_handles = [sim.getObject(joint) for joint in self.arm_joints]
        
        self.stop()

        self.youbot_max_W = 0.2 #[rad/seconds] Maximum angular velocity (omega) of youbot
        self.youbot_max_V = 0.5 #[m/seconds] Maximum linear velocity (V) of youbot

        self.goal_dist_threshold = 0.01 #[m] The threshold distance at which robot is declared to be at goal 
        self.goal_orientation_threshold = 0.01 #[rad] The threshold orientation at which robot is declared to be at goal 

        # Robot parameters
        self.wheel_radius = 0.0475  # Wheel radius (meters)
        self.half_length = 0.235    # Half the robot's length (meters)
        self.half_width = 0.15      # Half the robot's width (meters)
          
    # def get_robot_pose(self):

    #     position = self.sim.getObjectPosition(self.youbot_handle, -1)
    #     angles = self.sim.getObjectOrientation(self.youbot_handle, -1)
        
    #     print(angles)
    #     yaw = self.normalize_angle(angles[2]) # Angle b/w robot's head and global x

    #     return [position[0], position[1], yaw]

    def set_arm_position_youbot(self, list):
        # Motion limits in deg/sec, then converted to rad/sec
        vel = 180
        accel = 40
        jerk = 80

        max_vel = [float(np.deg2rad(vel))] * 5
        max_accel = [float(np.deg2rad(accel))] * 5
        max_jerk = [float(np.deg2rad(jerk))] * 5

        # Get only 5 arm joint handles
        handles = self.arm_handles[:5]

        angles = [[0.00016827562038024269, 0.5435698870030836, 0.9175151824094341, 1.2705860225917798, -8.667589826005795e-05], [-0.2416629679792381, 0.5781678871079382, 0.6780731389862094, 1.0005658648209526, -0.24519328585918015], [-0.6870686414352338, 0.3955825872635978, 0.668461621839978, 1.0757342452472762, -0.3571474177891562], [-1.073065399032333, 0.5390415589086736, 0.5697368121824226, 1.3339876242726598, -0.2940234860428298], [-1.4204106175632962, 0.7473364482813235, 0.7697463869561888, 1.2261020513194152, -0.47928387407081635], [-1.758065783893215, 0.8554996762232587, 0.6201856902333708, 1.526465157936968, -0.3711020505017243], [-2.0957209502231335, 0.9636629041651935, 0.47062499351055276, 1.8268282645545206, -0.26292022693263223], [-2.433376116553051, 1.0718261321071287, 0.32106429678773474, 2.1271913711720725, -0.15473840336354017], [-2.771031282882971, 1.179989360049064, 0.171503600064917, 2.427554477789624, -0.04655657979444816]]
        for pos_deg in angles:
            # Convert to radians and ensure Python float type
            target_rad = pos_deg

            self.sim.moveToConfig({
                'joints': handles,
                'targetPos': target_rad,
                'maxVel': max_vel,
                'maxAccel': max_accel,
                'maxJerk': max_jerk
            })

        print("Finished all arm motions.")

    def get_robot_pose(self):
        """
        Get current robot position and orientation using custom quaternion conversion
        to avoid gimbal lock issues.
        """
        position = self.sim.getObjectPosition(self.youbot_handle, -1)
        orientation = self.sim.getObjectQuaternion(self.youbot_handle, -1)

        q = orientation

        yaw, pitch, roll  = R.from_quat(q).as_euler('yxz') 

        return [position[0], position[1], roll]
    

    def set_wheel_velocities(self, wheel_vels):
        """Set wheel velocity targets in simulation."""

        v_fl = wheel_vels[0]
        V_fr = wheel_vels[1]
        v_rl = wheel_vels[2]
        v_rr = wheel_vels[3]

        v = [v_fl, V_fr, v_rl, v_rr]
        for i, handle in enumerate(self.wheel_handles):
    
            self.sim.setJointTargetVelocity(handle, v[i])

    def compute_wheel_velocities(self, velocity_list):
        
        # Robot's Wheel orientation

        vx = -velocity_list[1]
        vy_w = velocity_list[0]
        omega = velocity_list[2]
        
        """Correctly convert chassis velocities to wheel velocities for YouBot."""
        l = self.half_length
        w = self.half_width
        r = self.wheel_radius
        
        # Correct wheel kinematics (fixed signs)

        # Rear wheel is front wheel
        v_rr = (vx - vy_w - (l + w) * omega) / r  # Front right
        v_rl = (vx + vy_w + (l + w) * omega) / r  # Front left
        v_fr = (vx + vy_w - (l + w) * omega) / r  # Rear right
        v_fl = (vx - vy_w + (l + w) * omega) / r  # Rear left

        return [v_fl, v_fr, v_rl, v_rr]

    def follow_path(self, path, pos_threshold=0.05, orient_threshold=0.1):
        
        path = path[1:]
        # print(path)
        # self.get_robot_pose()
        # path = [(2.25011, -2.17, 1.57)]
            # print("robot_stopped")
            # self.stop()

        for i, waypoint in enumerate(path):
            target_x, target_y, target_theta = waypoint

            print(f"Moving to waypoint {i+1}/{len(path)}: [{target_x:.2f}, {target_y:.2f}, {target_theta:.2f}]")
            
            waypoint_reached = False
            start_time = time.time()
            timeout = 30  # Timeout to prevent infinite loops

            while not waypoint_reached: # and (time.time() - start_time) < timeout

                self.gtg(waypoint)

                if self.at_goal(waypoint):
                    waypoint_reached = True
                    self.stop()
                    print(f"Waypoint {i+1} reached!")

        print("Path following complete!")

    def stop(self):
            """Stop the robot."""
            self.set_wheel_velocities([0, 0, 0, 0])

    def at_goal(self, goal_state):    
    
        robot_state = self.get_robot_pose()
        flag_dist_thershold = False
        flag_orientation_threshold = False
        
        #check distance to goal point 
        d = np.sqrt(((goal_state[0] - robot_state[0])**2) + ((goal_state[1] - robot_state[1])**2))
        # print(d)

        if d <= self.goal_dist_threshold:
            flag_dist_thershold = True
        
        #check orientation diff 
        delta_theta = goal_state[2] - robot_state[2]
        #restrict angle to (-pi,pi)
        delta_theta = ((delta_theta + np.pi)%(2.0*np.pi)) - np.pi

        # print(f"Robot Theta = {robot_state[2]}, Goal State = {goal_state[2]}")
        # print(f"error theta {delta_theta}")

        if np.abs(delta_theta) < self.goal_orientation_threshold:
            flag_orientation_threshold = True
            
        if flag_dist_thershold and flag_orientation_threshold:
            
            return True
        else:
            return False
        
    def gtg(self, goal_state):  
    
        #The Go to goal controller
        # print(f"Goal_poosition - {goal_state[0]}, {goal_state[1]}, {goal_state[2]}")
        robot_state = self.get_robot_pose()
        print(f"Robot - {robot_state}")
        print(f"Goal - {goal_state}")

        # print(f"Current_Actual_poosition - {robot_state[0]}, {robot_state[1]}, {robot_state[2]}")
        # print(f"Moving to waypoint {goal_state[0]}, y {goal_state[1]}, theta {goal_state[2]}")
        

        #check distance to goal point 
        d = np.sqrt(((goal_state[0] - robot_state[0])**2) + ((goal_state[1] - robot_state[1])**2))
        print(f"Error Distance {d}")

        c = 100
        if d > self.goal_dist_threshold:
            #Only linear motion
            Wa = 0.0

            

            #Get actuation in global frame
            if np.abs(goal_state[0] - robot_state[0]) > (self.goal_dist_threshold/2.0):
                Vx = c * np.copysign(self.youbot_max_V, (goal_state[0] - robot_state[0]))
            else:
                Vx = 0.0
            
            if np.abs(goal_state[1] - robot_state[1]) > (self.goal_dist_threshold/2.0):
                Vy = c * np.copysign(self.youbot_max_V, (goal_state[1] - robot_state[1]))
            else:
                Vy = 0.0                
            
            
            #Get current orientation and get actuation in local frame
            local_Vy = -Vx*np.sin(robot_state[2]) + Vy*np.cos(robot_state[2])
            local_Vz = -Vy*np.sin(robot_state[2]) - Vx*np.cos(robot_state[2])

            # print(f"Local_velocities {local_Vy}, {local_Vz}, {Wa}")
            
            #request robot to execute velocity

            loc_vel = [local_Vy, local_Vz, Wa]
            wheel_vel = self.compute_wheel_velocities(loc_vel)
            
            self.set_wheel_velocities(wheel_vel)
            return
        
        else:
            print("stage2")
            #check orientation diff 
            delta_theta = goal_state[2] - robot_state[2]
            #restrict angle to (-pi,pi)
            delta_theta = ((delta_theta + np.pi)%(2.0*np.pi)) - np.pi
            
            
            if np.abs(delta_theta) > self.goal_orientation_threshold:
                #Only angular motion 
                Vx = 0.0
                Vy = 0.0 
                Wa = c * np.copysign(self.youbot_max_W, delta_theta)

                # print(f"Angular Velocity {Wa}")
                
                #request robot to execute velocity
                

                local_Vz = 0
                local_Vy = 0
                W_x = Wa

                loc_vel = [local_Vy, local_Vz, W_x]
                wheel_vel = self.compute_wheel_velocities(loc_vel)
                self.set_wheel_velocities(wheel_vel)
                
                return
            
            else:
                local_Vz = 0
                local_Vy = 0
                W_x = 0

                loc_vel = [local_Vy, local_Vz, W_x]
                wheel_vel = self.compute_wheel_velocities(loc_vel)
                self.set_wheel_velocities(wheel_vel)
                return
            
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π] with consistent handling near ±π."""
        normalized = ((angle + np.pi) % (2 * np.pi)) - np.pi
        if abs(abs(normalized) - np.pi) < 0.1:
            return np.sign(normalized) * np.pi
        return normalized
    
    # def normalize_angle(angle):
    #     return np.arctan2(np.sin(angle), np.cos(angle))