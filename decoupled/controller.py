import time
import numpy as np

class YouBotController:
    def __init__(self, sim, dt=0.01):
        """Initialize controller for the Kuka YouBot with proper orientation handling."""
        self.sim = sim
        self.dt = dt
        
        # Robot parameters
        self.wheel_radius = 0.0475  # Wheel radius (meters)
        self.half_length = 0.235    # Half the robot's length (meters)
        self.half_width = 0.15      # Half the robot's width (meters)
        
        # Control gains
        self.pos_gain = 3.0         # Position control gain
        self.orient_gain = 1.0      # Orientation gain
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
        
        # Get arm joint handles
        self.arm_joints = [
            '/youBotArmJoint0', '/youBotArmJoint1', 
            '/youBotArmJoint2', '/youBotArmJoint3', 
            '/youBotArmJoint4'
        ]
        self.arm_handles = [sim.getObject(joint) for joint in self.arm_joints]
    
    def get_robot_pose(self):
        """
        Get current robot position and orientation using custom quaternion conversion
        to avoid gimbal lock issues.
        """
        position = self.sim.getObjectPosition(self.youbot_handle, -1)
        orientation = self.sim.getObjectOrientation(self.youbot_handle, -1)
        
        # Manual conversion from Euler angles to quaternion
        # Extract the angles
        roll, yaw, pitch  = orientation
        
        # Convert to quaternion using the standard formula
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        # Quaternion components (w, x, y, z)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        # Convert back to Euler angles (yaw-pitch-roll sequence)
        # This helps avoid gimbal lock by using quaternion intermediate representation
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll_new = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        pitch_new = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw_new = np.arctan2(siny_cosp, cosy_cosp)
        
        # Return position and the yaw (heading angle)
        return [position[0], position[1], yaw_new]

    
    def compute_wheel_velocities(self, vx, vy, omega):
        """Convert chassis velocities to wheel velocities."""
        l = self.half_length
        w = self.half_width
        r = self.wheel_radius
        
        # Corrected wheel kinematics for YouBot
        v_fr = (-vx - vy - (l + w) * omega) / r  # Front right
        v_fl = (-vx + vy - (l + w) * omega) / r  # Front left
        v_rr = (-vx + vy + (l + w) * omega) / r  # Rear right
        v_rl = (-vx - vy + (l + w) * omega) / r  # Rear left
        
        return [v_fr, v_fl, v_rr, v_rl]
    
    def set_wheel_velocities(self, wheel_vels):
        """Set wheel velocity targets in simulation."""
        for i, handle in enumerate(self.wheel_handles):
            vel = np.clip(wheel_vels[i], -self.max_linear_vel/self.wheel_radius, 
                           self.max_linear_vel/self.wheel_radius)
            self.sim.setJointTargetVelocity(handle, vel)
    
    def follow_path(self, path, pos_threshold=0.05, orient_threshold=0.1):
        """Follow the A* path with orientation control."""
        print(f"Starting path following with {len(path)} waypoints")
        
        for i, waypoint in enumerate(path):
            target_x, target_y, target_theta = waypoint
            print(f"Moving to waypoint {i+1}/{len(path)}: [{target_x:.2f}, {target_y:.2f}, {target_theta:.2f}]")
            
            # Control loop for this waypoint
            waypoint_reached = False
            start_time = time.time()
            timeout = 30  # Timeout to prevent infinite loops
            
            while not waypoint_reached and (time.time() - start_time) < timeout:
                # Get current robot pose using quaternions to avoid gimbal lock
                curr_x, curr_y, curr_theta = self.get_robot_pose()
                
                # Calculate errors in world frame
                pos_error_world = [target_x - curr_x, target_y - curr_y]
                
                # Calculate orientation error - normalize for shortest path
                orient_error = self.normalize_angle(target_theta - curr_theta)
                
                # Transform position errors to robot frame
                cos_theta = np.cos(curr_theta)
                sin_theta = np.sin(curr_theta)
                pos_error_robot = [
                    -cos_theta * pos_error_world[0] - sin_theta * pos_error_world[1],
                    sin_theta * pos_error_world[0] - cos_theta * pos_error_world[1]
                ]
                
                # Calculate control terms for position
                vx = self.pos_gain * pos_error_robot[0]
                vy = self.pos_gain * pos_error_robot[1]
                
                # Limit linear velocities
                v_mag = np.sqrt(vx**2 + vy**2)
                if v_mag > self.max_linear_vel:
                    scale = self.max_linear_vel / v_mag
                    vx *= scale
                    vy *= scale
                
                # Use reduced gain for large orientation errors
                if abs(orient_error) > 1.5:
                    omega = 0.5 * self.orient_gain * orient_error
                else:
                    omega = self.orient_gain * orient_error
                    
                omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)
                
                # Compute and set wheel velocities
                wheel_vels = self.compute_wheel_velocities(vx, vy, omega)
                self.set_wheel_velocities(wheel_vels)
                
                # Debug output
                dist_error = np.sqrt(pos_error_world[0]**2 + pos_error_world[1]**2)
                print(f"Position error: {dist_error:.3f}m, Orientation error: {orient_error:.3f}rad, " +
                      f"Velocities: vx={vx:.2f}, vy={vy:.2f}, omega={omega:.2f}")
                
                # Check if waypoint is reached
                if (dist_error < pos_threshold and abs(orient_error) < orient_threshold):
                    waypoint_reached = True
                    print(f"Waypoint {i+1} reached!")
                
                time.sleep(self.dt)
            
            if not waypoint_reached:
                print(f"Timeout reached for waypoint {i+1}. Moving to next waypoint.")
        
        # Stop the robot after completing the path
        self.set_wheel_velocities([0, 0, 0, 0])
        print("Path following complete!")

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] with special handling for angles near ±π."""
        normalized = ((angle + np.pi) % (2 * np.pi)) - np.pi
        
        # If the angle is very close to ±π, use a consistent value to avoid oscillation
        if abs(abs(normalized) - np.pi) < 0.1:
            return np.sign(normalized) * np.pi
            
        return normalized
    
    def stop(self):
        """Stop the robot."""
        self.set_wheel_velocities([0, 0, 0, 0])
    
    def move_arm(self, joint_angles, speed=0.5):
        """Move the arm to specified joint angles."""
        for i, angle in enumerate(joint_angles):
            self.sim.setJointTargetPosition(self.arm_handles[i], angle)
