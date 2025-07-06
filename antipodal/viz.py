
# use this to get grasps : force_closure) omen@phoenix:~/home2/fc_evaluation/DA2/scripts$ nano posetest_last_pls.py 




import sys
sys.path.append('DA2_tools')
import random
import numpy as np
import h5py
import open3d as o3d
from DA2_tools import create_panda_marker # create_robotiq_marker
from scipy.spatial.transform import Rotation as R

def compute_camera_wrt_base(roll, pitch, yaw, x_mm, y_mm, z_mm):
    # Convert degrees to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx
    R_base = Rz @ Ry @ Rx  

    # Position from mm to meters
    x = x_mm / 1000.0
    y = y_mm / 1000.0
    z = z_mm / 1000.0

    # Construct T_eef_wrt_base
    T_eef_wrt_base = np.eye(4)
    T_eef_wrt_base[:3, :3] = R_base
    T_eef_wrt_base[:3, 3] = [x, y, z]

    # Camera relative to end-effector
    R_cam = R.from_euler('xyz', [0, 0, -np.pi / 2], degrees=False).as_matrix()
    t_cam = [0.08, 0, 0.04]
    T_cam_wrt_eef = np.eye(4)
    T_cam_wrt_eef[:3, :3] = R_cam
    T_cam_wrt_eef[:3, 3] = t_cam

    # Final transformation
    T_cam_wrt_base = T_eef_wrt_base @ T_cam_wrt_eef
    return T_cam_wrt_base


# Example usage
T_cam_wrt_base = compute_camera_wrt_base(
    x_mm=-25.4 , y_mm=328 , z_mm=277.3,
    roll=-175.7 , pitch=-62, yaw=-7.5

)
theta = np.pi  # 180 degrees
rotation_matrix_z = np.array([
        [np.cos(theta), -np.sin(theta), 0,0],
        [np.sin(theta),  np.cos(theta), 0,0],
        [0,              0,             1,0],
        [0,0,0,1]
    ])
T_cam_wrt_base = T_cam_wrt_base @rotation_matrix_z #coz image is expected to rotaed by 180 deg

print("T_cam_wrt_base is", T_cam_wrt_base)


basepath= '/home/pavan/Desktop/RA_L/Anygrap_Xarm/antipodal/yellow_cylinder/experiment_dir/registered_meshes/' #'/home/pavan/Desktop/RA_L/Anygrap_Xarm/real_life/yellow_cylinder/experiment_dir/registered_meshes/'

# Load the object mesh
obj_path = f'{basepath}0.obj'
object_mesh = o3d.io.read_triangle_mesh(obj_path)
object_mesh.compute_vertex_normals()
object_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
object_mesh.transform(T_cam_wrt_base)
pcd= o3d.io.read_point_cloud(f'{basepath}scene_complete.ply')

### rotate about z axis by 180
theta = np.pi  # 180 degrees
rotation_matrix_z = np.array([
        [np.cos(theta), -np.sin(theta), 0,0],
        [np.sin(theta),  np.cos(theta), 0,0],
        [0,              0,             1,0],
        [0,0,0,1]
    ])

pcd.transform(T_cam_wrt_base)
# Load grasp transforms
grasp_path = f"{basepath}0_decomposed_1.h5"


# with h5py.File(grasp_path, 'r') as f:
#     def print_structure(name, obj):
#         if isinstance(obj, h5py.Group):
#             print(f"Group: {name}")
#         elif isinstance(obj, h5py.Dataset):
#             print(f"Dataset: {name} — shape: {obj.shape}, dtype: {obj.dtype}")

#     f.visititems(print_structure)

grasps_data = h5py.File(grasp_path, 'r')
grasps = grasps_data['grasps/transforms'][:].reshape(-1, 4, 4)
end_points = grasps_data['grasps/end_points'][:]  # shape: (103, 2, 3)
print(f"Loaded {len(grasps)} grasp transforms")

# Create Open3D geometries
geometries = [object_mesh]


# Visualize only the first N grasps to avoid performance issues

selected_index=[]
for i, grasp in enumerate(grasps):
    # Create a gripper mesh
    gripper_trimesh = create_panda_marker()

    # Convert to Open3D if it's a Trimesh mesh
    if hasattr(gripper_trimesh, 'dump'):
        gripper_trimesh = gripper_trimesh.dump().sum()
    gripper_o3d = o3d.geometry.TriangleMesh()
    gripper_o3d.vertices = o3d.utility.Vector3dVector(gripper_trimesh.vertices)
    gripper_o3d.triangles = o3d.utility.Vector3iVector(gripper_trimesh.faces)
    gripper_o3d.compute_vertex_normals()
    gripper_o3d.paint_uniform_color([0.1, 0.7, 0.1])  # Green

    # Apply the transformation
    gripper_o3d.transform(T_cam_wrt_base@ grasp)
### filtering
    theta = -np.pi / 2  # 90 degrees in radians
    Rz_90 = np.array([
        [np.cos(theta), -np.sin(theta), 0,0],
        [np.sin(theta),  np.cos(theta), 0,0],
        [0,              0,             1,0],[0,0,0,1]
    ])
    grasp_worldT = T_cam_wrt_base@ grasp 

    ##
    R_mat_XG = grasp_worldT[:3, :3]  #xram gripper
    T_mat_XG = grasp_worldT[:3, 3] 


    theta = -np.pi / 2  # 90 degrees in radians
    Rz_90 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    R_mat_XG = R_mat_XG @Rz_90

    ##grasp
    z_axis = R_mat_XG[:, 2]  # Local z-axis (gripper heading)
    t_moved = T_mat_XG - 0.08 * z_axis #113 worked

    T = np.eye(4)
    T[:3, :3] = R_mat_XG  
    T[:3, 3] =t_moved
    ##
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    axis_frame.transform(T)
    
    approach_vector = T[:3, 2]  # z-axis in world frame
    # World z-axis
    world_z = np.array([0, 0, -1])
    # Compute angle between approach and world Z-axis
    cos_theta = np.dot(approach_vector, world_z) / (np.linalg.norm(approach_vector))
    angle_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    print(i , "   ",angle_deg)
    # Filter: keep only top-down grasps
    if  t_moved[2]<0.145:
        continue  # Skip if not top-down
###    
    geometries.append(axis_frame)
    geometries.append(gripper_o3d)
    selected_index.append(i)
    for point in end_points[i]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(point)
        sphere.transform(T_cam_wrt_base)
        sphere.paint_uniform_color([1, 0, 0])
        geometries.append(sphere)



# Show everything in Open3D visualizer
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
o3d.visualization.draw_geometries(geometries+[axes])
random.shuffle(selected_index)
print("seleced index of grasp ",selected_index)