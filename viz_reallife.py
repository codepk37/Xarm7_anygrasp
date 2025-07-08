import open3d as o3d
import numpy as np
import os

# === Set paths ===
base_dir = "real_life/green_bottle/experiment_dir/registered_meshes"
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


# import open3d as o3d
# import numpy as np
# import os
# import glob

# # === Set root directory ===
# root_dir = "real_life"

# # === Pattern to match all pavanoutput6* folders ===
# folder_pattern = os.path.join(root_dir, "pavanoutput6*")

# # === Iterate through each matched folder ===
# for folder_path in glob.glob(folder_pattern):
#     registered_meshes_dir = os.path.join(folder_path, "experiment_dir", "registered_meshes")
#     print(f"\n=== Processing: {registered_meshes_dir} ===")

#     # Path to scene_complete.ply
#     ply_path = os.path.join(registered_meshes_dir, "scene_complete.ply")
#     if not os.path.exists(ply_path):
#         print(" -> PLY file not found, skipping.")
#         continue

#     # Load and trim point cloud
#     pcd = o3d.io.read_point_cloud(ply_path)
#     # points = np.asarray(pcd.points)
#     # distances = np.linalg.norm(points, axis=1)
#     # mask = distances <= 3.0
#     # pcd.points = o3d.utility.Vector3dVector(points[mask])

#     # if pcd.has_colors():
#     #     colors = np.asarray(pcd.colors)
#     #     pcd.colors = o3d.utility.Vector3dVector(colors[mask])

#     # Load .obj meshes
#     mesh_objs = []
#     if os.path.exists(registered_meshes_dir):
#         for file in os.listdir(registered_meshes_dir):
#             if file.endswith(".obj"):
#                 mesh_path = os.path.join(registered_meshes_dir, file)
#                 mesh = o3d.io.read_triangle_mesh(mesh_path)
#                 mesh.compute_vertex_normals()
#                 mesh_objs.append(mesh)
#     else:
#         print(" -> registered_meshes directory missing, skipping.")
#         continue

#     # Visualize
#     geometries = [pcd] + mesh_objs
#     o3d.visualization.draw_geometries(geometries)
