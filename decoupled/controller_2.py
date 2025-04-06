import time
import numpy as np
from scipy.spatial.transform import Rotation as R


class YouBotController:
    def __init__(self, sim, dt=0.01):
        """Initialize controller for the Kuka YouBot."""
        
        self.sim = sim
        self.dt = dt
        
        # Robot parameters
        self.wheel_radius = 0.0475  # Wheel radius (meters)
        self.half_length = 0.235    # Half the robot's length (meters)
        self.half_width = 0.15      # Half the robot's width (meters)
        
        # Control gains
        self.pos_gain = 1.5         # Position control gain
        self.orient_gain = 0.5      # Orientation gain
        self.max_linear_vel = 2.0   # Maximum linear velocity (m/s)
        self.max_angular_vel = 1.0  # Maximum angular velocity (rad/s)
        
        # Get robot handle
        self.youbot_handle = sim.getObject('/youBot')
        
        
        # Get wheel handles in correct order
        self.wheel_joints = [
            '/rollingJoint_fr', '/rollingJoint_fl', 
            '/rollingJoint_rr', '/rollingJoint_rl'
        ]
        self.wheel_handles = [sim.getObject(joint) for joint in self.wheel_joints]
        self.stop()

        # Get arm joint handles (not used in path following)
        self.arm_joints = [
            '/youBotArmJoint0', '/youBotArmJoint1', 
            '/youBotArmJoint2', '/youBotArmJoint3', 
            '/youBotArmJoint4'
        ]
        self.arm_handles = [sim.getObject(joint) for joint in self.arm_joints]

        self.youbot_max_W = 0.2 #[rad/seconds] Maximum angular velocity (omega) of youbot
        self.youbot_max_V = 0.5 #[m/seconds] Maximum linear velocity (V) of youbot

        self.goal_dist_threshold = 0.3 #[m] The threshold distance at which robot is declared to be at goal 
        self.goal_orientation_threshold = 0.1 #[rad] The threshold orientation at which robot is declared to be at goal 
            
    def get_robot_pose(self):
        """
        Get current robot position and orientation using custom quaternion conversion
        to avoid gimbal lock issues.
        """
        position = self.sim.getObjectPosition(self.youbot_handle, -1)
        orientation = self.sim.getObjectQuaternion(self.youbot_handle, -1)
        
        print(orientation)
        yaw_new = 0.0
        # roll, yaw, pitch  = orientation
        
        # # Convert to quaternion using the standard formula
        # cy = np.cos(yaw * 0.5)
        # sy = np.sin(yaw * 0.5)
        # cp = np.cos(pitch * 0.5)
        # sp = np.sin(pitch * 0.5)
        # cr = np.cos(roll * 0.5)
        # sr = np.sin(roll * 0.5)
        
        # # Quaternion components (w, x, y, z)
        # w = cr * cp * cy + sr * sp * sy
        # x = sr * cp * cy - cr * sp * sy
        # y = cr * sp * cy + sr * cp * sy
        # z = cr * cp * sy - sr * sp * cy
        
        # # Convert back to Euler angles (yaw-pitch-roll sequence)
        # # This helps avoid gimbal lock by using quaternion intermediate representation
        # sinr_cosp = 2 * (w * x + y * z)
        # cosr_cosp = 1 - 2 * (x * x + y * y)
        # roll_new = np.arctan2(sinr_cosp, cosr_cosp)
        
        # sinp = 2 * (w * y - z * x)
        # pitch_new = np.arcsin(sinp)
        
        # siny_cosp = 2 * (w * z + x * y)
        # cosy_cosp = 1 - 2 * (y * y + z * z)
        # yaw_new = np.arctan2(siny_cosp, cosy_cosp)
        
        # print(f"This is {yaw_new}")
        # Return position and the yaw (heading angle)
        return [position[0], position[1], yaw_new]
    
    def compute_wheel_velocities(self, vx, vy, omega):
        """Correctly convert chassis velocities to wheel velocities for YouBot."""
        l = self.half_length
        w = self.half_width
        r = self.wheel_radius
        
        # Correct wheel kinematics (fixed signs)
        v_fr = (vx - vy - (l + w) * omega) / r  # Front right
        v_fl = (vx + vy + (l + w) * omega) / r  # Front left
        v_rr = (vx + vy - (l + w) * omega) / r  # Rear right
        v_rl = (vx - vy + (l + w) * omega) / r  # Rear left
        
        return [v_fr, v_fl, v_rr, v_rl]
    
    def set_wheel_velocities(self, wheel_vels):
        """Set wheel velocity targets in simulation."""

        Vx = wheel_vels[0]
        Vy = wheel_vels[1]
        Wa = wheel_vels[2]

        v = [-Vx -Vy -Wa, -Vx +Vy -Wa, -Vx -Vy +Wa, -Vx +Vy +Wa,]
        for i, handle in enumerate(self.wheel_handles):
    
            self.sim.setJointTargetVelocity(handle, v[i])
    
    # def follow_path(self, path, pos_threshold=0.05, orient_threshold=0.1):
    #     """Follow the path with corrected error transformation and control."""
    #     print(f"Starting path following with {len(path)} waypoints")
        
    #     for i, waypoint in enumerate(path):
    #         target_x, target_y, target_theta = waypoint
    #         print(f"Moving to waypoint {i+1}/{len(path)}: [{target_x:.2f}, {target_y:.2f}, {target_theta:.2f}]")
            
    #         waypoint_reached = False
    #         start_time = time.time()
    #         timeout = 30  # Timeout to prevent infinite loops
            
    #         while not waypoint_reached and (time.time() - start_time) < timeout:
    #             curr_x, curr_y, curr_theta = self.get_robot_pose()
                
    #             # Position error in world frame
    #             pos_error_world = [target_x - curr_x, target_y - curr_y]
                
    #             # Orientation error (shortest path)
    #             orient_error = self.normalize_angle(target_theta - curr_theta)
                
    #             # Transform position error to robot frame (corrected)
    #             cos_theta = np.cos(curr_theta)
    #             sin_theta = np.sin(curr_theta)
    #             pos_error_robot = [
    #                 cos_theta * pos_error_world[0] + sin_theta * pos_error_world[1],
    #                 -sin_theta * pos_error_world[0] + cos_theta * pos_error_world[1]
    #             ]
                
    #             # Position control terms
    #             vx = self.pos_gain * pos_error_robot[0]
    #             vy = self.pos_gain * pos_error_robot[1]
                
    #             # Limit linear velocities
    #             v_mag = np.sqrt(vx**2 + vy**2)
    #             if v_mag > self.max_linear_vel:
    #                 scale = self.max_linear_vel / v_mag
    #                 vx *= scale
    #                 vy *= scale
                
    #             # Orientation control with reduced gain for large errors
    #             if abs(orient_error) > 1.5:
    #                 omega = 0.5 * self.orient_gain * orient_error
    #             else:
    #                 omega = self.orient_gain * orient_error
    #             omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)
                
    #             # Compute and set wheel velocities
    #             wheel_vels = self.compute_wheel_velocities(vx, vy, omega)
    #             self.set_wheel_velocities(wheel_vels)
                
    #             # Check if waypoint is reached
    #             dist_error = np.sqrt(pos_error_world[0]**2 + pos_error_world[1]**2)
    #             if dist_error < pos_threshold and abs(orient_error) < orient_threshold:
    #                 waypoint_reached = True
    #                 print(f"Waypoint {i+1} reached!")
                
    #             time.sleep(self.dt)
            
    #         if not waypoint_reached:
    #             print(f"Timeout for waypoint {i+1}. Proceeding.")
        
    #     self.stop()
    #     print("Path following complete!")

    def follow_path(self, path, pos_threshold=0.05, orient_threshold=0.1):

        # path = path[1:]

        path = [(2.25011, -1.17, 1.1207)]
        # print("robot_stopped")
        # self.stop()

        for i, waypoint in enumerate(path):
            target_x, target_y, target_theta = waypoint

            # print(target_theta)
            print(f"Moving to waypoint {i+1}/{len(path)}: [{target_x:.2f}, {target_y:.2f}, {target_theta:.2f}]")
            
            waypoint_reached = False
            start_time = time.time()
            timeout = 30  # Timeout to prevent infinite loops

            while not waypoint_reached: # and (time.time() - start_time) < timeout

                self.gtg(waypoint)
                print(waypoint_reached)

                if self.at_goal(waypoint):
                    waypoint_reached = True
                    print(f"Waypoint {i+1} reached!")

        print("Path following complete!")
            
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π] with consistent handling near ±π."""
        normalized = ((angle + np.pi) % (2 * np.pi)) - np.pi
        if abs(abs(normalized) - np.pi) < 0.1:
            return np.sign(normalized) * np.pi
        return normalized
    
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
        print(f"error_distance - {d}")
        if d <= self.goal_dist_threshold:
            flag_dist_thershold = True
        
        #check orientation diff 
        delta_theta = goal_state[2] - robot_state[2]
        #restrict angle to (-pi,pi)
        delta_theta = ((delta_theta + np.pi)%(2.0*np.pi)) - np.pi
        
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

        print(f"Current_Actual_poosition - {robot_state[0]}, {robot_state[1]}, {robot_state[2]}")
        print(f"Moving to waypoint {goal_state[0]}, y {goal_state[1]}, theta {goal_state[2]}")
        

        #check distance to goal point 
        d = np.sqrt(((goal_state[0] - robot_state[0])**2) + ((goal_state[1] - robot_state[1])**2))
        print(d)
        
        if d > self.goal_dist_threshold:
            #Only linear motion
            Wa = 0.0

            #Get actuation in global frame
            if np.abs(goal_state[0] - robot_state[0]) > (self.goal_dist_threshold/2.0):
                Vx = np.copysign(self.youbot_max_V, goal_state[0] - robot_state[0])
            else:
                Vx = 0.0
            
            if np.abs(goal_state[1] - robot_state[1]) > (self.goal_dist_threshold/2.0):
                Vy = np.copysign(self.youbot_max_V, goal_state[1] - robot_state[1])
            else:
                Vy = 0.0                
            
            #Get current orientation and get actuation in local frame
            local_Vx = Vx*np.cos(-robot_state[2]) - Vy*np.sin(-robot_state[2])
            local_Vy = Vx*np.sin(-robot_state[2]) + Vy*np.cos(-robot_state[2])
            
            #request robot to execute velocity

            wheel_vel = [local_Vx, local_Vy, Wa]
            self.set_wheel_velocities(wheel_vel)
            
            return
        else:
            #check orientation diff 
            delta_theta = goal_state[2] - robot_state[2]
            #restrict angle to (-pi,pi)
            delta_theta = ((delta_theta + np.pi)%(2.0*np.pi)) - np.pi
            
            if np.abs(delta_theta) > self.goal_orientation_threshold:
                #Only angular motion 
                Vx = 0.0
                Vy = 0.0 
                Wa = np.copysign(self.youbot_max_W, delta_theta)
                
                #request robot to execute velocity
                self.compute_wheel_velocities(Vx, Vy, Wa)
                
                return
            
            else:
                #request robot to stop
                self.compute_wheel_velocities(0.0, 0.0, 0.0)
                
                return
    
    # move_arm remains unchanged if not used in path following