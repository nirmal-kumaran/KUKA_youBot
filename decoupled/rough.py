import numpy as np
import modern_robotics as mr

def inverse_kinematics(target_pose):
        """
        Solve inverse kinematics to compute joint angles for a given end-effector pose.

        Args:
            target_pose: Target SE(3) pose of the end-effector (4x4 matrix).

        Returns:
            Joint angles [J0, J1, J2, J3, J4] or None if no solution is found.
        """
        # Define home configuration of the end-effector
        M = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])
        
        # Define screw axes in body frame
        B_list_arm = np.array([[0,0,0,0,0],
                        [0,-1,-1,-1,0],
                        [1,0,0,0,1],
                        [0,-0.5076,-0.3526,-0.2176,0],
                        [0.033,0,0,0,0],
                        [0,0,0,0,0]])

        # Initial guess for joint angles
        initial_guess = [0.0, -np.pi/4, np.pi/6, -np.pi/6, np.pi/12]

        # Use Modern Robotics' IK solver
        joint_angles, success = mr.IKinBody(B_list_arm, M, target_pose, initial_guess, eomg=1e-3, ev=1e-3)
        
        if success:
            return joint_angles.tolist()
        # else:
        #     raise Exception("Failed to find a valid joint configuration.")
        
def normalize_joint_angles(joint_angles):
    """
    Normalize joint angles to the range [-π, π].

    Args:
        joint_angles: List or array of joint angles in radians.

    Returns:
        Normalized joint angles in the range [-π, π].
    """
    return [(angle + np.pi) % (2 * np.pi) - np.pi for angle in joint_angles]


ik = inverse_kinematics(np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0.1],
                    [0, 0, 1, 0.8],
                    [0, 0, 0, 1]]))
print(ik)



# Example usage
original_angles = [7.534508202457563, -8.844009968710582, 18.174803767379615, -9.330793798669033, -7.534508202457564]
normalized_angles = normalize_joint_angles(original_angles)
print(normalized_angles)
# Define inputs
B_list_arm = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, -1, -1, 0],
    [1, 0, 0, 0, 1],
    [0, -0.5076, -0.3526, -0.2176, 0],
    [0.033, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
M = np.array([
    [1, 0, 0, 0.033],
    [0, 1, 0,     0],
    [0, 0, 1, 0.6546],
    [0, 0, 0,     1]
])
thetalist = np.array(normalized_angles)

# Compute Forward Kinematics
T_ee = mr.FKinBody(M, B_list_arm, thetalist)

print("Computed End-Effector Pose:")
print(T_ee)
