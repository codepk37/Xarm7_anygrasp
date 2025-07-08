import open3d as o3d
import numpy as np
import cv2
import pyrealsense2 as rs

# Load saved data
depth_path = "./orange_cylinder/depth.npy"
rgb_path = "./orange_cylinder/rgb.png"

depth = np.load(depth_path)  # depth in mm
rgb = cv2.imread(rgb_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert to RGB for Open3D

# Convert depth to uint16 for Open3D
depth_uint16 = depth.astype(np.uint16)

# Get intrinsics from RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()
print("intrinsics were",intrinsics)
pipeline.stop()

# Create Open3D camera intrinsics
o3d_intr = o3d.camera.PinholeCameraIntrinsic(
    width=intrinsics.width,
    height=intrinsics.height,
    fx=intrinsics.fx,
    fy=intrinsics.fy,
    cx=intrinsics.ppx,
    cy=intrinsics.ppy
)

# Create Open3D images
depth_o3d = o3d.geometry.Image(depth_uint16)
rgb_o3d = o3d.geometry.Image(rgb)

# Create RGBD image
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    rgb_o3d,
    depth_o3d,
    depth_scale=1000.0,   # mm to meters
    depth_trunc=3.0,      # max range (3 meters)
    convert_rgb_to_intensity=False
)

# Generate point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, o3d_intr
)

# Flip to Open3D coordinate system
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Visualize
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd,axis], window_name="Point Cloud")
