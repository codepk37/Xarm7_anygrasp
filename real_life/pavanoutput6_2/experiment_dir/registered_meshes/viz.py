import open3d as o3d
import numpy as np
import os

# === Set paths ===
base_dir = "./"
ply_path = os.path.join(base_dir, "scene_complete.ply")

# === Load and trim point cloud ===
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)

# Compute Euclidean distance from origin and mask
distances = np.linalg.norm(points, axis=1)
mask = distances <= 3.0

# Apply mask to points
pcd.points = o3d.utility.Vector3dVector(points[mask])

# Apply mask to colors if available
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])

# === Load all .obj meshes in the directory ===
mesh_objs = []
for file in os.listdir(base_dir):
    if file.endswith(".obj"):
        mesh_path = os.path.join(base_dir, file)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        mesh_objs.append(mesh)

# === Visualize trimmed point cloud + meshes ===
geometries = [pcd] + mesh_objs
o3d.visualization.draw_geometries(geometries)
