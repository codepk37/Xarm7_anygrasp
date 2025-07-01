import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import time
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from scipy.spatial.transform import Rotation as R
from makegripper_points import  plot_gripper_pro_max
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--object', type=str, required=False, help='Object name for grasp detection')  # New argument for 003_cracker_box
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))


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
    x_mm=179.6, y_mm=89.1, z_mm=314.6,
    roll=-175.6, pitch=-37, yaw=-0.6

)

print("T_cam_wrt_base is", T_cam_wrt_base)



def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'rgb.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))

    fx, fy = 384.722, 384.357
    cx, cy = 323.244, 243.432
    scale = 1000.0

    # set workspace to filter output grasps
    xmin, xmax = -0.25, 0.2
    ymin, ymax = -0.3, 0.2
    zmin, zmax = 0.0, 2
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]


    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < zmax)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    # print(points.min(axis=0), points.max(axis=0))
    # print("Point cloud bounds:", points.min(axis=0), points.max(axis=0))

    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)

    print(f"Min Bound (x, y, z): {min_bounds}")
    print(f"Max Bound (x, y, z): {max_bounds}")
    mask = (
    (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &  # x values
    (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &  # y values
    (points[:, 2] >= zmin) & (points[:, 2] <= zmax)    # z values
    )

    # Apply the mask to filter the points and colors
    points = points[mask]
    colors = colors[mask]

    pcd_global = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # pcd_global.points = o3d.utility.Vector3dVector(points)
    # pcd_global.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd_global,axis])

### rotate about z axis by 180
    theta = np.pi  # 180 degrees
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    # Rotate points directly (around origin)
    points = (rotation_matrix @ points.T).T  # Transpose, multiply, then transpose back
    points = points.astype(np.float32)
    colors = colors.astype(np.float32)
    pcd_global.points = o3d.utility.Vector3dVector(points)
    pcd_global.colors = o3d.utility.Vector3dVector(colors)
###
    

    #visualize point cloud
    # o3d.visualization.draw_geometries([pcd_global,axis])


    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=True, collision_detection=True)

    # now transform pointcloud and grasp with T_cam_wrt_base
    # T_cam_wrt_base
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud.transform(T_cam_wrt_base)



    print("hi before mask")
    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
    else:
         print(len(gg),"tpatl grasp")
    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20] #10


    pcds=[]
    gripper_points = []
    print("Top grasp scores:", gg_pick.scores)
    print('Highest grasp score:', gg_pick[0].score)



    gripper_output_subfolder = f"./{data_dir}"
    os.makedirs(gripper_output_subfolder, exist_ok=True)

    translations =[]
    rotations =[]
    widths =[]
    heights =[]
    scores =[]
  
    for graspi in gg_pick:
        t_grasp = graspi.translation 
        R_grasp = graspi.rotation_matrix
        width = graspi.width
        depth = graspi.height
        score = graspi.score

        # Convert to 4x4 transformation matrix
        T_grasp = np.eye(4)
        T_grasp[:3, :3] = R_grasp
        T_grasp[:3, 3] = t_grasp

        T_grasp_in_base = T_cam_wrt_base @ T_grasp  #camera -> world frame

        R_new = T_grasp_in_base[:3, :3]
        t_new = T_grasp_in_base[:3, 3]

        gripper_pcd,gripper_point = plot_gripper_pro_max(t_new, R_new, width=width, depth=depth, score=score,color=[0,1,0])
        
        pcds.append(gripper_pcd)
        gripper_points.append(gripper_point)
        translations.append(t_new)
        rotations.append(R_new)
        widths.append(width)
        heights.append(depth)
        scores.append(score)





    dots = []
    for p in gripper_points:
        # Create a sphere (representing a dot)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)  # radius can be adjusted
        sphere.translate(p[0])  # Translate to the given point
        sphere.paint_uniform_color([1, 0, 0])  # Color red
        dots.append(sphere)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere.translate(p[1])  # Translate to the given point
        sphere.paint_uniform_color([1, 0, 0])  # Color red
        dots.append(sphere)

    # points = np.random.rand(1000, 3)
    gripper_data = {
    'translations': np.array(translations),
    'rotations': np.array(rotations),
    'widths': np.array(widths),
    'heights': np.array(heights),
    'scores': np.array(scores),
    'gripper_points': np.array(gripper_points)
    }
    np.save(f"{gripper_output_subfolder}/grasps.npy", gripper_data)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cloud,axis]+pcds+dots)
    cloud_filename =  f"./{gripper_output_subfolder}/cloud.ply"
    o3d.io.write_point_cloud(cloud_filename, cloud)

if __name__ == '__main__':
    
    demo('./pavanoutput6')




