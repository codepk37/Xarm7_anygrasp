Step 1) 
conda activate pybullet
run to save images [pavanvideo_stream640x480](pavanvideo_stream640x480.py)

----
Step 2)
conda activate /media/pavan/STORAGE/linux_storage/anygrasp_sdk/py3-mink

Run to get grasp in base frame (internally grasp in cam->endeff frame ,  endeff frame -> base frame)

python real_life.py --checkpoint_path log/checkpoint_detection.tar

NOTE: if u change end-effector position, update this after checking "ufactory studio" 
T_cam_wrt_base = compute_camera_wrt_base(
    roll=-175.6, pitch=-37, yaw=-0.6,
    x_mm=179.6, y_mm=89.1, z_mm=314.6
)

-----

step 3)
conda activate pybullet
see index wise grasp :[real_like](real_like.ipynb)

example prints:
Grasp Index: 0
  Score: 0.3296
  Translation (x, y, z): [556.19852157  96.84616325 121.85274518]
  Rotation (roll, pitch, yaw) [deg]: [-101.89943203   84.98887378  140.48428298]
  Width: 0.0658  | Depth: 0.0300

-----
step 4)
conda activate pybullet
Feed above obtained grasp translation and roll,pitch,yaw in below api 
[xarm_api](xarm_api.ipynb)

end-effector will move to grasp position
