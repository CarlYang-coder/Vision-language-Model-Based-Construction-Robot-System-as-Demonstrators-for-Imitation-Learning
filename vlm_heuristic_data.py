import argparse
import os
import pickle
import gc
import numpy as np
import cv2

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import torch
from omni.isaac.core import World
from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController
from omni.isaac.universal_robots.tasks import TimberAssembly
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.isaac.core.utils.prims as prims
from omni.isaac.core.articulations import Articulation
from omni.isaac.core import SimulationContext
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.sensor import Camera, get_all_camera_objects
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles


# ========== Your custom modules ==========
from sam_point_sampling import PointSampling
from visual_question_answering import VQA
import keypoint_utils

# ========== CLI ==========
parser = argparse.ArgumentParser(description="Isaac Sim run with per-frame logging.")
parser.add_argument("--sim_name", type=str, default="Simulation_1",
                    help="Folder name under ~/Downloads/Data_Collecting")
parser.add_argument("--mask_idx", type=int, default=5,
                    help="Mask index to use for keypoint sampling")
args = parser.parse_args()
SIM_NAME = args.sim_name
MASK_IDX = args.mask_idx

# ========== Save paths ==========
base_dir  = os.path.abspath(os.path.expanduser("~/Downloads/Data_Collecting"))
sim_dir   = os.path.abspath(os.path.join(base_dir, SIM_NAME))
frame_dir = os.path.join(sim_dir, "frame")
os.makedirs(frame_dir, exist_ok=True)

# Single pkl record container: frame_idx -> dict
records = {}

# ====== Video output (camera) ======
video_path = os.path.join(sim_dir, "run.mp4")
video_writer = None        # Lazily initialized after the first frame (to get resolution)
VIDEO_FPS   = 30

# ========== World / Task ==========
my_world = World(stage_units_in_meters=1.0)
my_task = TimberAssembly(target_position=np.array([-0.2, 0.652155, 0.0515/2]))
my_world.add_task(my_task)
my_world.reset()

task_params = my_task.get_params()
robot_name = task_params["robot_name"]["value"]
my_ur10 = my_world.scene.get_object(robot_name)
my_controller = PickPlaceController(name="pick_place_controller",
                                    gripper=my_ur10.gripper,
                                    robot_articulation=my_ur10)
articulation_controller = my_ur10.get_articulation_controller()

# Load scene
stage = stage_utils.add_reference_to_stage(
    usd_path="/home/carl/Downloads/test_6_modified.usd",
    prim_path="/World/factory",
)

# Object
timber_path = "/World/factory/Wood_block"
_ = prims.get_prim_at_path(timber_path)
timber_articulation = Articulation(timber_path)

# Camera
cameras = get_all_camera_objects("/World/factory")
camera0 = cameras[0]
camera0.initialize()
camera0.set_resolution([512, 512])
camera0.add_distance_to_image_plane_to_frame()

# Vision / geometry utilities
point_sampler = PointSampling()

# ========== Motion and target ==========
placing_position = np.array([0.00067, 0.63435, 0.01], dtype=float)  # Initial placeholder; will be overwritten after frame 200
rotation_angles  = np.array([0.0, np.pi/2, np.pi/4], dtype=float)   # Initial placeholder; will be overwritten after frame 200
Z_GROUND_TRUTH   = -0.00216499                                      # Fixed z

# Only update once
computed_place_once = False

# ========== Main loop ==========
i = 0
reset_needed = False
simulation_context = SimulationContext()

while simulation_app.is_running() and not my_controller.is_done():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            camera0.initialize()
            camera0.set_resolution([512, 512])
            reset_needed = False

        observations = my_world.get_observations()
        t = simulation_context.current_time

        # First go through a via-point (no pressing down): unchanged
        if t < 10:
            actions = my_controller.forward(
                picking_position=timber_articulation.get_local_pose()[0],
                placing_position=np.array([0.3, 0.97194, 0.64551]),
                current_joint_positions=observations[robot_name]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0.02]),
                end_effector_orientation=euler_angles_to_quat(np.array([0, np.pi/2, 0])),
            )
        else:
            # After that, always use the placing_position updated at frame 200 (z fixed)
            actions = my_controller.forward(
                picking_position=timber_articulation.get_local_pose()[0],
                placing_position=placing_position.reshape(3,),
                current_joint_positions=observations[robot_name]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0.04]),  # Hover without pressing down
                end_effector_orientation=euler_angles_to_quat(rotation_angles),
            )

        if my_controller.is_done():
            print("done picking and placing")

        articulation_controller.apply_action(actions)

    # Capture RGB / depth
    image = camera0.get_rgb()
    depth = np.array(camera0.get_current_frame()['distance_to_image_plane'])

    if image is not None and image.size:
        # Save PNG for each frame
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.png"), image_bgr)

        # Initialize VideoWriter once (on first frame, using its resolution)
        if video_writer is None:
            h, w = image_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # If mp4v doesn't work, try "XVID"
            video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (w, h))
            if not video_writer.isOpened():
                print("[WARN] Failed to open VideoWriter. Check encoder / path permissions:", video_path)
                video_writer = None

        # Write current frame to video
        if video_writer is not None:
            video_writer.write(image_bgr)

        # === Only at frame 200: SAM + VQA, update (x,y), keep z fixed ===
        if (i == 200) and (not computed_place_once):
            print("[i=200] Initial keypoint sampling.")
            point_sampler.setup_image(image=image)
            point_sampler.generate_masks()

            # Save mask visualizations and cutouts
            for idx in range(len(point_sampler.masks)):
                point_sampler.visualize_masks(
                    mask_idx=idx,
                    save_path=os.path.join(sim_dir, f"timber_sam_output_{idx}.png")
                )
                point_sampler.cutout_segmentation(
                    mask_idx=idx,
                    save_path=os.path.join(sim_dir, f"timber_sam_cutout_{idx}.png")
                )

            # Use CLI-selected mask_idx
            selected_points_2D, mask_points_2D = point_sampler.sample_keypoints(
                mask_idx=MASK_IDX, num_points=8
            )

            # Project to 3D (mainly for visualization; final xy uses VQA-selected two points)
            mask_points_depths = depth[mask_points_2D[:, 1], mask_points_2D[:, 0]].squeeze()
            mask_points_3D = camera0.get_world_points_from_image_coords(
                points_2d=mask_points_2D,
                depth=mask_points_depths,
            )
            selected_points_3D = keypoint_utils.fps(mask_points_3D, 8)
            selected_points_2D_proj = camera0.get_image_coords_from_world_points(
                points_3d=selected_points_3D,
            ).astype(int)

            # Save marked image (and record absolute path)
            marked_path = os.path.abspath(os.path.join(sim_dir, f"timber_sam_cutout_marked_3D_mask{MASK_IDX}.png"))
            point_sampler.save_annotated_image(
                selected_points_2D_proj,
                save_path=marked_path
            )
            print("[DEBUG] VQA will read:", marked_path, "exists?", os.path.exists(marked_path))

            # Use VQA to select two numeric labels that correspond to pixel points
            print("[i=200] Vision-Language model is selecting the pair of numbers.")
            # first_number, second_number = VQA()
            first_number, second_number = VQA(marked_path)
            pts_px = np.array([
                point_sampler.get_point_by_label(first_number)[::-1],  # (x1,y1)
                point_sampler.get_point_by_label(second_number)[::-1], # (x2,y2)
            ], dtype=int)

            # Clamp to image bounds and get depth
            H, W = depth.shape[:2]
            pts_px[:, 0] = np.clip(pts_px[:, 0], 0, W - 1)
            pts_px[:, 1] = np.clip(pts_px[:, 1], 0, H - 1)
            d = depth[pts_px[:, 1], pts_px[:, 0]].astype(float)
            valid = np.isfinite(d) & (d > 0.0)

            if valid.sum() < 2:
                print("[WARN @ i=200] Invalid depth for the two VQA-selected points; keeping initial placing_position.")
            else:
                # Back-project to world coordinates
                pts_world = camera0.get_world_points_from_image_coords(
                    points_2d=pts_px[valid], depth=d[valid]
                )  # (2,3)

                # Update xy as the mean of the two points; set z to ground truth
                xy_mean = pts_world.mean(axis=0)[:2]
                placing_position = np.array([xy_mean[0], xy_mean[1], Z_GROUND_TRUTH], dtype=float)

                # Compute yaw from direction between the two points (projected on XY plane),
                # pitch = π/2, roll = 0
                dir_vec = pts_world[0] - pts_world[1]
                dir_vec[2] = 0.0
                n = np.linalg.norm(dir_vec[:2])
                if n > 1e-6:
                    dir_vec /= n
                    xh, yh = dir_vec[:2]
                    yaw = np.arctan2(yh, xh) - np.pi / 2.0
                else:
                    yaw = 0.0
                rotation_angles = np.array([0.0, np.pi/2, yaw], dtype=float)

                print("[UPDATE @ i=200] placing_position =", placing_position,
                      "rotation_angles =", rotation_angles)

                # Immediately move to the new pose once (still hovering with end_effector_offset=0.04)
                actions = my_controller.forward(
                    picking_position=timber_articulation.get_world_pose()[0],
                    placing_position=placing_position.reshape(3,),
                    current_joint_positions=observations[robot_name]["joint_positions"],
                    end_effector_offset=np.array([0, 0, 0.04]),
                    end_effector_orientation=euler_angles_to_quat(rotation_angles),
                )
                articulation_controller.apply_action(actions)

            computed_place_once = True

        # ===== Per-frame logging =====
        # Joint positions
        joint_pos = None
        try:
            jp = my_ur10.get_joint_positions()
            if jp is not None:
                joint_pos = np.asarray(jp, dtype=float)
        except Exception as e:
            print("[WARN] my_ur10.get_joint_positions() failed:", e)

        if joint_pos is None:
            try:
                joint_pos = np.asarray(observations[robot_name]["joint_positions"], dtype=float)
            except Exception as e:
                print("[WARN] Could not fetch joint_positions from observations:", e)

        if joint_pos is None:
            joint_pos = np.full((6,), np.nan, dtype=float)

        # Current end-effector pose (position + orientation)
        try:
            ee_obj = getattr(my_ur10, "end_effector", None) or my_ur10.get_end_effector()
        except Exception:
            ee_obj = my_ur10.gripper  # Fallback

        ee_pos, ee_quat = ee_obj.get_world_pose()          # ee_pos: (x,y,z), ee_quat: quaternion
        ee_pos = np.asarray(ee_pos, dtype=float)
        ee_rpy = quat_to_euler_angles(np.asarray(ee_quat, dtype=float))  # [roll, pitch, yaw] (radians)

        # Next-step target end-effector pose (using current placing_position + rotation_angles)
        # Only need {x, y, z, yaw}
        target_next_xyz = np.asarray(placing_position, dtype=float)
        target_next_yaw = float(rotation_angles[2])

        # Write current frame record
        records[int(i)] = {
            "joint_position": joint_pos.tolist(),
            "ee_pose": {
                "x": float(ee_pos[0]),
                "y": float(ee_pos[1]),
                "z": float(ee_pos[2]),
                "roll":  float(ee_rpy[0]),
                "pitch": float(ee_rpy[1]),
                "yaw":   float(ee_rpy[2]),
            },
            "ee_pose_next": {
                "x": float(target_next_xyz[0]),
                "y": float(target_next_xyz[1]),
                "z": float(target_next_xyz[2]),
                "yaw": float(target_next_yaw),
            },
        }

    i += 1

# ===== Save pkl =====
os.makedirs(sim_dir, exist_ok=True)
with open(os.path.join(sim_dir, "data.pkl"), "wb") as f:
    pickle.dump(records, f)

# ===== Save final frame =====
final_image = camera0.get_rgb()
if final_image is not None and final_image.size:
    final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(sim_dir, "final_frame.png"), final_bgr)

# Release video resources
try:
    if video_writer is not None:
        video_writer.release()
except Exception as e:
    print("[WARN] Failed to release VideoWriter:", e)

simulation_app.close()
gc.collect()

