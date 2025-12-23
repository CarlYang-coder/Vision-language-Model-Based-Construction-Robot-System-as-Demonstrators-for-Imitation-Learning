#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import gc
import re
import glob
import numpy as np
import cv2
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

from omni.isaac.core import World
from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController
from omni.isaac.universal_robots.tasks import TimberAssembly
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core import SimulationContext
import omni.isaac.core.utils.prims as prims
from omni.isaac.core.articulations import Articulation
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.sensor import Camera, get_all_camera_objects

# ========= Basic parameters =========
SIM_NAME   = "demo_1_img"    # Used only for output directory naming
USD_PATH   = "/home/carl/Downloads/test_6_modified.usd"
DATA_ROOT  = os.path.expanduser("~/Downloads/Data_Collecting")
H5_PATH    = os.path.join(DATA_ROOT, "bc_dataset_from616.hdf5")

# ----- Trigger frame: use the camera RGB image at frame 180 for prediction -----
TARGET_IMG_FRAME = 180      # The frame index where prediction is triggered
PRED_USE_ALL_STEPS = False  # True = predict at every frame; False = predict only once at TARGET_IMG_FRAME

# ----- Fixed Z coordinate for predicted end-effector target -----
Z_FIXED    = -0.00216499

# ----- Output (video + final frame) -----
sim_dir   = os.path.join(DATA_ROOT, SIM_NAME)
os.makedirs(sim_dir, exist_ok=True)
VIDEO_FPS   = 30
video_path  = os.path.join(sim_dir, "camera_view.mp4")
last_frame_path = os.path.join(sim_dir, "final_frame.png")
video_writer = None

# ========= Action de-normalization (must match dataset settings) =========
with h5py.File(H5_PATH, "r") as f:
    a_scale = np.asarray(f["data"].attrs["action_scale"], dtype=np.float32)  # (3,)
    a_bias  = np.asarray(f["data"].attrs["action_bias"],  dtype=np.float32)  # (3,)
if a_scale.shape != (3,) or a_bias.shape != (3,):
    raise RuntimeError(f"HDF5 action_scale/bias should have shape (3,), got {a_scale.shape}")

def denorm_xyyaw(x_norm):  # (3,) -> (3,) world space
    return (x_norm - a_bias) / a_scale

# ========= TinyCNN (must match training configuration) =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TinyCNN(nn.Module):
    def __init__(self, in_ch=3, out_dim=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.gap   = nn.AdaptiveAvgPool2d((1,1))
        self.fc1   = nn.Linear(64, 64)
        self.fc2   = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # normalized output [-1,1]
        return x

# ----- Select or auto-search checkpoint -----
RUN_DIR = os.path.expanduser('~/Downloads/robomimic_runs/simple_img2xyyaw_tiny/tinycnn_img180_to_xyyaw_right')
CKPT    = ""  # e.g. "/path/to/best.pt"; empty -> auto search RUN_DIR

def _epoch_num(p):
    m = re.search(r'epoch[_\-](\d+)', os.path.basename(p))
    return int(m.group(1)) if m else -1

def find_ckpt():
    if CKPT and os.path.isfile(CKPT):
        print(f"[CKPT] Using explicit: {CKPT}")
        return CKPT
    cands = []
    for pat in ("**/*best*.pt", "**/*best*.pth", "**/epoch_*.pt", "**/epoch-*.pt", "**/*.pt", "**/*.pth"):
        cands += glob.glob(os.path.join(RUN_DIR, pat), recursive=True)
    if not cands:
        raise FileNotFoundError(f"Checkpoint not found. Check RUN_DIR={RUN_DIR} or set CKPT manually.")

    bests = [p for p in cands if 'best' in os.path.basename(p).lower()]
    if bests:
        bests.sort(key=os.path.getmtime)
        return bests[-1]

    epochs = [p for p in cands if _epoch_num(p) >= 0]
    if epochs:
        epochs.sort(key=_epoch_num)
        return epochs[-1]

    cands.sort(key=os.path.getmtime)
    return cands[-1]

def load_model(ckpt_path):
    model = TinyCNN(in_ch=3, out_dim=3).to(DEVICE)
    obj = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(obj, dict):
        state = None
        for k in ("state_dict", "model", "model_state", "weights"):
            if k in obj and isinstance(obj[k], dict):
                state = obj[k]
                break
        if state is None:
            state = obj

        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state[k] = v

        model.load_state_dict(new_state, strict=False)

    elif isinstance(obj, nn.Module):
        model = obj.to(DEVICE)
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(obj)}")

    model.eval()
    print(f"[CKPT] Loaded: {ckpt_path}")
    return model

def preprocess_img(img_uint8, out_hw=(224,224)):
    img = cv2.resize(img_uint8, out_hw, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None, ...]
    return torch.from_numpy(img).to(DEVICE)

# ========= World + task setup =========
my_world = World(stage_units_in_meters=1.0)
my_task = TimberAssembly(target_position=np.array([-0.2, 0.652155, 0.0515/2]))
my_world.add_task(my_task)
my_world.reset()

task_params = my_task.get_params()
robot_name = task_params["robot_name"]["value"]

my_ur10 = my_world.scene.get_object(robot_name)
my_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=my_ur10.gripper,
    robot_articulation=my_ur10
)
articulation_controller = my_ur10.get_articulation_controller()

# Load USD scene
stage_utils.add_reference_to_stage(
    usd_path=USD_PATH,
    prim_path="/World/factory",
)

# Timber block
timber_path = "/World/factory/Wood_block"
timber_prim = prims.get_prim_at_path(timber_path)
print(f"[Info] Timber prim: {timber_prim}")
timber_articulation = Articulation(timber_path)

# Camera
cameras = get_all_camera_objects("/World/factory") or get_all_camera_objects("/World")
if not cameras:
    cam_prim_path = "/World/PolicyCam"
    camera0 = Camera(
        prim_path=cam_prim_path,
        name="PolicyCam",
        position=np.array([1.2, 0.0, 1.0]),
        frequency=VIDEO_FPS
    )
else:
    camera0 = cameras[0]

camera0.initialize()
camera0.set_resolution([512, 512])
camera0.add_distance_to_image_plane_to_frame()

# ========= Initial target (will be replaced after prediction) =========
placing_position = np.array([0.0, 0.0, Z_FIXED], dtype=float)
rotation_angles  = np.array([0.0, np.pi/2, np.pi/4], dtype=float)

# ========= Load model checkpoint =========
ckpt_path = find_ckpt()
model = load_model(ckpt_path)

# ========= Main simulation loop =========
i = 0
reset_needed = False
simulation_context = SimulationContext()
pred_applied_once = False

print("[Info] Starting simulation loop...")
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

        # ===== Prediction logic: trigger prediction at frame TARGET_IMG_FRAME =====
        need_predict = False
        if PRED_USE_ALL_STEPS:
            need_predict = True
        else:
            if (i == TARGET_IMG_FRAME) and (not pred_applied_once):
                need_predict = True

        if need_predict:
            img_rgb = camera0.get_rgb()
            if img_rgb is not None and img_rgb.size:
                x = preprocess_img(img_rgb, out_hw=(224,224))
                with torch.no_grad():
                    pred_norm = model(x).detach().cpu().numpy().reshape(3)  # [-1,1]
                xy_yaw = denorm_xyyaw(pred_norm)  # world coordinates {x,y,yaw}
                xw, yw, yaw = float(xy_yaw[0]), float(xy_yaw[1]), float(xy_yaw[2])
                placing_position = np.array([xw, yw, Z_FIXED], dtype=float)
                rotation_angles  = np.array([0.0, np.pi/2, yaw], dtype=float)
                pred_applied_once = True
                print(f"[PRED @ frame={i}] norm={pred_norm}  -> world(x,y,yaw)={xy_yaw}")

        # ===== Controller logic: t < 10 goes via a waypoint; then goes to predicted target =====
        if t < 10:
            actions = my_controller.forward(
                picking_position=timber_articulation.get_local_pose()[0],
                placing_position=np.array([0.3, 0.97194, 0.64551]),
                current_joint_positions=observations[robot_name]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0.02]),
                end_effector_orientation=euler_angles_to_quat(np.array([0, np.pi/2, 0])),
            )
        else:
            actions = my_controller.forward(
                picking_position=timber_articulation.get_local_pose()[0],
                placing_position=placing_position.reshape(3,),
                current_joint_positions=observations[robot_name]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0.04]),
                end_effector_orientation=euler_angles_to_quat(rotation_angles),
            )

        if my_controller.is_done():
            print("Completed pick and place task.")
        articulation_controller.apply_action(actions)

    # ===== Camera recording =====
    img_rgb = camera0.get_rgb()
    if img_rgb is not None and img_rgb.size:
        if video_writer is None:
            h, w = img_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (w, h))
            if not video_writer.isOpened():
                print(f"[WARN] Cannot open VideoWriter: {video_path}")
                video_writer = None
        if video_writer is not None:
            video_writer.write(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Debug print current state
    picking_position_world = timber_articulation.get_world_pose()[0]
    print(
        f"[Frame {i:04d}] t={simulation_context.current_time:.3f}s\n"
        f"  picking_position (world): {picking_position_world}\n"
        f"  placing_position (world): {placing_position}\n"
        f"  rotation_angles [roll, pitch, yaw]: {rotation_angles}"
    )
    i += 1

# ===== Save last frame and release resources =====
last = camera0.get_rgb()
if last is not None and last.size:
    cv2.imwrite(last_frame_path, cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    print(f"[OK] Saved final frame: {last_frame_path}")

try:
    if video_writer is not None:
        video_writer.release()
        print(f"[OK] Saved video: {video_path}")
except Exception as e:
    print(f"[WARN] Failed to release VideoWriter: {e}")

simulation_app.close()
gc.collect()
print("[Done]")

