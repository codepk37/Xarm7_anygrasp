import numpy as np
import open3d as o3d

def create_mesh_box(width, height, depth=0.0299, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box
def plot_gripper_pro_max(center, R, width, depth, score=1, color=[0,1,0]):
    depth = depth /2 # half depth
    x, y, z = center
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base =  0.02
    
    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score # red for high score
        color_g = 0
        color_b = 1 - score # blue for low score
    
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    # print("left_point ", depth , -width/2 , -(height/2))
    # print("right_point ", depth , width/2, -(height/2))
    left_point = np.array([depth, -width/2 -finger_width, -(height/2)])
    right_point = np.array([depth,  width/2+finger_width, -(height/2)])
    
    gripper_points= np.vstack([left_point, right_point])
    gripper_points = np.dot(R, gripper_points.T).T + center

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    # print("bottom_points ", bottom_points)

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    # print("tail_points ", tail_points)

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center



    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)

    # print("vertices ", np.asarray(gripper.vertices))
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper ,gripper_points


if __name__=="__main__":

    rotation_matrix = np.array([[ 1.43918172e-01, -4.37861681e-01, -8.87448430e-01],
                                [-2.77105689e-01,  8.43075991e-01, -4.60907042e-01],
                                [ 9.49999988e-01,  3.12249899e-01, -1.36488767e-08]])

    # Translation vector (3,)
    translation = np.array([-0.01600729,  0.07222326,  0.4161298])
    # Call the plot_gripper_pro_max function and visualize

    # we just want width , height is always same 0.0299
    gripper,gripper_points = plot_gripper_pro_max(translation, rotation_matrix, width=0.0185, depth=0.0299, score=0.05, color=[1,1,0])

    # Visualize the gripper mesh
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Create red spheres (dots) at the given points
    dots = []
    for point in gripper_points:
        # Create a sphere (representing a dot)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)  # radius can be adjusted
        sphere.translate(point)  # Translate to the given point
        sphere.paint_uniform_color([1, 0, 0])  # Color red
        dots.append(sphere)




    # Create a line set to represent the vector direction from one point to the other
    lines = [[0, 1]]  # Connect point 0 to point 1
    colors = [[0, 1, 0]]  # Green color for the vector

    # Create a LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(gripper_points)  # Points for the line
    line_set.lines = o3d.utility.Vector2iVector(lines)    # Define lines connecting the points
    line_set.colors = o3d.utility.Vector3dVector(colors)  # Color for the line



    pcd = o3d.io.read_point_cloud("/media/pavan/STORAGE/linux_storage/anygrasp_sdk/grasp_detection/example_data/curr_pipeline_pcd/037_scissors.pcd")
    # points_np = np.load(pcd_file_path) forr .npy

    points = np.asarray(pcd.points, dtype=np.float32)  # Ensure dtype is float32
    colors = np.asarray(pcd.colors, dtype=np.float32)  # Ensure dtype is float32

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the gripper, coordinate frame, and vectors (lines)
    o3d.visualization.draw_geometries([gripper,cloud, coordinate_frame] + dots + [line_set])