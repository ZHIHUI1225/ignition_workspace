import numpy as np
from geometry_msgs.msg import PoseStamped

def get_wall_pose_from_topic(topic_pose: PoseStamped):
    """
    Extracts the (x, z) position from the OptiTrack topic pose.
    Returns as numpy array [x, z].
    """
    return np.array([topic_pose.pose.position.x, topic_pose.pose.position.z])

# Camera frame positions (pixels)
CAMERA_WALL0 = np.array([135, 335.5])
CAMERA_WALL1 = np.array([651.5, 176])

# Example: OptiTrack frame positions (meters, from topic)
# These should be filled in with actual topic data at runtime
# OPTITRACK_WALL0 = get_wall_pose_from_topic(wall0_pose_msg)
# OPTITRACK_WALL1 = get_wall_pose_from_topic(wall1_pose_msg)


def compute_affine_transform(optitrack_pts, camera_pts):
    """
    Computes affine transform (rotation, scale, translation) from optitrack_pts to camera_pts.
    optitrack_pts: Nx2 numpy array (e.g. [[x0, z0], [x1, z1]])
    camera_pts: Nx2 numpy array (e.g. [[x0, y0], [x1, y1]])
    Returns: 2x2 matrix (rotation+scale), 2x1 translation vector
    """
    # Solve for A, b in: camera_pts = A @ optitrack_pts.T + b
    # Use two points for a unique solution
    A = np.zeros((2,2))
    b = np.zeros((2,))
    # Build system
    X = optitrack_pts.T
    Y = camera_pts.T
    # A = (Y1-Y0)/(X1-X0) for each axis
    delta_X = X[:,1] - X[:,0]
    delta_Y = Y[:,1] - Y[:,0]
    scale = np.linalg.norm(delta_Y) / np.linalg.norm(delta_X)
    theta = np.arctan2(delta_Y[1], delta_Y[0]) - np.arctan2(delta_X[1], delta_X[0])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = scale * R
    b = Y[:,0] - A @ X[:,0]
    return A, b


def optitrack_to_camera(optitrack_pos, A, b):
    """
    Transforms a position from OptiTrack frame to camera frame using affine transform.
    optitrack_pos: [x, z] numpy array
    A: 2x2 matrix
    b: 2x1 vector
    Returns: [x, y] in camera frame
    """
    return (A @ optitrack_pos) + b


def camera_to_optitrack(camera_pos, A, b):
    """
    Transforms a position from camera frame to OptiTrack frame using inverse affine transform.
    camera_pos: [x, y] numpy array
    A: 2x2 matrix
    b: 2x1 vector
    Returns: [x, z] in OptiTrack frame
    """
    return np.linalg.inv(A) @ (camera_pos - b)

def print_robot_positions_in_camera(optitrack_wall0_pose, optitrack_wall1_pose, robot_poses):
    """
    Print the positions of robot0, robot1, robot2 in camera frame.
    optitrack_wall0_pose, optitrack_wall1_pose: PoseStamped messages for wall0 and wall1
    robot_poses: dict with keys 'robot0', 'robot1', 'robot2', values are PoseStamped messages
    """
    optitrack_pts = np.array([
        get_wall_pose_from_topic(optitrack_wall0_pose),
        get_wall_pose_from_topic(optitrack_wall1_pose)
    ])
    camera_pts = np.array([CAMERA_WALL0, CAMERA_WALL1])
    A, b = compute_affine_transform(optitrack_pts, camera_pts)
    for robot in ['robot0', 'robot1', 'robot2']:
        if robot in robot_poses and robot_poses[robot] is not None:
            optitrack_pos = get_wall_pose_from_topic(robot_poses[robot])
            camera_pos = optitrack_to_camera(optitrack_pos, A, b)
            print(f"{robot} position in camera frame: {camera_pos}")
        else:
            print(f"{robot} pose not available.")

# Example usage:
# optitrack_pts = np.array([[wall0_x, wall0_z], [wall1_x, wall1_z]])
# camera_pts = np.array([CAMERA_WALL0, CAMERA_WALL1])
# A, b = compute_affine_transform(optitrack_pts, camera_pts)
# camera_xy = optitrack_to_camera(np.array([wall0_x, wall0_z]), A, b)
# optitrack_xz = camera_to_optitrack(np.array([135, 335.5]), A, b)
print_robot_positions_in_camera(optitrack_wall0_pose, optitrack_wall1_pose, {
    'robot0': robot0_pose,
    'robot1': robot1_pose,
    'robot2': robot2_pose
})