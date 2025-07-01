import numpy as np

def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Convert degrees to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Rotation matrix around x-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation matrix around y-axis (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation matrix around z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

def create_transformation_matrix(roll, pitch, yaw, x_mm, y_mm, z_mm):
    R = rpy_to_rotation_matrix(roll, pitch, yaw)
    
    # Convert mm to meters
    x = x_mm / 1000.0
    y = y_mm / 1000.0
    z = z_mm / 1000.0

    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T

# Input values ; get values for xarm ufactory
roll = -175.6      # degrees
pitch = -37   # degrees
yaw = -0.6      # degrees
x_mm = 179.6
y_mm = 89.1
z_mm = 314.6

T = create_transformation_matrix(roll, pitch, yaw, x_mm, y_mm, z_mm)
print("Homogeneous Transformation Matrix:\n", T)