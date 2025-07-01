import pyrealsense2 as rs
import numpy as np
import cv2
import os

output_dir = './pavanoutput6'
os.makedirs(output_dir, exist_ok=True)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB + Depth streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

try:
    print("Capturing frame...")
    for _ in range(100):  # Skip initial unstable frames
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        raise RuntimeError("Could not retrieve frames")

    # Convert to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Save RGB
    rgb_path = os.path.join(output_dir, 'rgb.png')
    cv2.imwrite(rgb_path, color_image)

    # Normalize depth for visualization (clip to valid range for display)
    depth_vis = np.clip(depth_image, 0, 3000)  # assuming 0–3m range
    depth_vis = cv2.convertScaleAbs(depth_vis, alpha=255.0 / 3000.0)  # normalize to 0–255

    # Apply a colormap
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    # Save for visualization
    cv2.imwrite(os.path.join(output_dir, 'depth.png'), depth_image) #depth_colormap

    # Save original depth as .npy
    depth_npy_path = os.path.join(output_dir, 'depth.npy')
    np.save(depth_npy_path, depth_image)

    print("Saved rgb.png, depth.png, and depth.npy to ./pavanoutput")

finally:
    pipeline.stop()
