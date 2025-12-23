#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === Isaac Sim startup ===
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# ====== Standard library / dependencies ======
import os, re, glob, json, gc
import numpy as np
import cv2, h5py
import torch, torch.nn as nn, torch.nn.functional as F

# Isaac Sim / UR tasks and sensors
from omni.isaac.core import World
from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController
from omni.isaac.universal_robots.tasks import TimberAssembly
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core import SimulationContext
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prims
from omni.isaac.core.articulations import Articulation
from omni.isaac.sensor import Camera, get_all_camera_objects

# ========= Paths & key parameters (edit as needed) =========
SIM_NAME   = "sim_eval_clipe_7_BL"  # Used for output directory naming
USD_PATH   = "/home/carl/Downloads/test_6_modified.usd"  # USD scene file
DATA_ROOT  = os.path.expanduser("~/Downloads/Data_Collecting")
DS         = os.path.join(DATA_ROOT, "bc_dataset_from616.hdf5")

# Use the same run directory as the evaluation script (decoupled CLIP model)
RUN_DIR    = os.path.expanduser(
    "~/Downloads/robomimic_runs/simple_img2xyyaw_tiny/decoupled_tinycnn_clip_xy_from_img_yaw_from_text"
)
CKPT       = ""  # Leave empty for auto-search, or set absolute path to e.g. models/model_best.pth

# Camera trigger: which frame index to run a prediction (e.g., first frame or 180th frame)
TARGET_IMG_FRAME   = 180
PRED_USE_ALL_STEPS = False  # True: predict at every step; False: predict once at TARGET_IMG_FRAME

# Fixed Z for the target (tuned for your controller)
Z_FIXED = -0.00216499

# Outputs (save video and final frame)
SIM_DIR         = os.path.join(DATA_ROOT, SIM_NAME)
os.makedirs(SIM_DIR, exist_ok=True)
VIDEO_FPS       = 30
VIDEO_PATH      = os.path.join(SIM_DIR, "camera_view.mp4")
LAST_FRAME_PATH = os.path.join(SIM_DIR, "final_frame.png")
video_writer    = None

# ========= Text instructions (two variants consistent with the dataset) =========
INSTR_UL_BR = "Place the timber along the diagonal of the upper left corner and the bottom right corner."
INSTR_BL_UR = "Place the timber along the diagonal of the bottom left corner and the upper right corner."
TEXT_INSTR  = INSTR_BL_UR

# ========= Device =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= Model definition (must match evaluation script) =========
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.a(self.b(self.c(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=s, padding=1, groups=in_ch, bias=False)
        self.db = nn.BatchNorm2d(in_ch)
        self.da = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pb = nn.BatchNorm2d(out_ch)
        self.pa = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.da(self.db(self.dw(x)))
        x = self.pa(self.pb(self.pw(x)))
        return x


class TinyCNN_Feature(nn.Module):
    def __init__(self, feat_dim=128, width=32):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 2, width * 4
        self.stem = ConvBNAct(3, c1, k=3, s=2, p=1)     # 160->80
        self.b2   = DepthwiseSeparable(c1, c2, s=2)     # 80->40
        self.b3   = DepthwiseSeparable(c2, c3, s=1)     # 40->40
        self.b4   = DepthwiseSeparable(c3, c4, s=2)     # 40->20
        self.b5   = DepthwiseSeparable(c4, c4, s=1)     # 20->20
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, feat_dim),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.pool(x)
        return self.proj(x)  # [B, feat_dim]


# Frozen CLIP text encoder (same as in evaluation)
from transformers import CLIPTextModel, CLIPTokenizer

class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.tokenizer    = CLIPTokenizer.from_pretrained(name)
        self.text_encoder = CLIPTextModel.from_pretrained(
            name,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, batch_text):
        if isinstance(batch_text, str):
            batch_text = [batch_text]
        inputs = self.tokenizer(
            batch_text, return_tensors='pt', padding=True, truncation=True, max_length=77
        ).to(self.device)
        out = self.text_encoder(**inputs).last_hidden_state    # [B,L,512]
        eos_pos = inputs.attention_mask.sum(dim=1) - 1
        sent = out[torch.arange(out.size(0), device=self.device), eos_pos]  # [B,512]
        return sent


class MultiModalDecoupledCLIP(nn.Module):
    """
    Image -> xy, Text -> yaw (cos,sin)
    Output matches evaluation: [x_norm, y_norm, yaw_norm]
    """
    def __init__(self, v_backbone: TinyCNN_Feature, t_encoder: FrozenCLIPTextEncoder,
                 v_dim=128, t_in_dim=512, t_dim=128):
        super().__init__()
        self.v_backbone = v_backbone
        self.t_encoder  = t_encoder
        self.text_proj  = nn.Sequential(
            nn.Linear(t_in_dim, t_dim),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )
        self.xy_head    = nn.Sequential(
            nn.Linear(v_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 2),
        )
        self.yaw_head   = nn.Sequential(
            nn.Linear(t_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 2),  # [cos, sin]
        )

    def forward(self, images, texts):
        v_feat = self.v_backbone(images)
        with torch.no_grad():
            t_raw = self.t_encoder(texts)
        t_feat = self.text_proj(t_raw)
        pred_xy = self.xy_head(v_feat)     # [B,2]
        pred_cs = self.yaw_head(t_feat)    # [B,2]
        pred_cs = pred_cs / (pred_cs.norm(dim=1, keepdim=True) + 1e-8)
        yaw = torch.atan2(pred_cs[:, 1], pred_cs[:, 0]) / torch.pi  # [-1,1]
        pred = torch.cat([pred_xy, yaw.unsqueeze(1)], dim=1)
        return pred


# ========= Checkpoint search/load (same as evaluation) =========
def _epoch_num(p):
    m = re.search(r'epoch[_\-](\d+)', os.path.basename(p))
    return int(m.group(1)) if m else -1


def find_ckpt():
    if CKPT and os.path.isfile(CKPT):
        print(f"[CKPT] Using explicit path: {CKPT}")
        return CKPT
    pats = ("**/*best*.pt", "**/*best*.pth", "**/epoch_*.pt",
            "**/epoch-*.pt", "**/*.pt", "**/*.pth")
    cands = []
    for pat in pats:
        cands += glob.glob(os.path.join(RUN_DIR, pat), recursive=True)
    if not cands:
        raise FileNotFoundError(
            f"Checkpoint not found. Check RUN_DIR={RUN_DIR} or set CKPT manually."
        )
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


def _get_state_and_keys(obj):
    st = obj.get("model", obj) if isinstance(obj, dict) else obj
    if "state_dict" in st and isinstance(st["state_dict"], dict):
        st = st["state_dict"]
    keys = list(st.keys()) if isinstance(st, dict) else []
    return st, keys


def build_model_from_ckpt_meta(ckpt_obj):
    st, keys = _get_state_and_keys(ckpt_obj)
    clip_name = ckpt_obj.get('clip_name', "openai/clip-vit-base-patch32")
    v_dim     = int(ckpt_obj.get('v_dim', 128)) if isinstance(ckpt_obj, dict) and 'v_dim' in ckpt_obj else 128
    t_dim     = int(ckpt_obj.get('t_dim', 128)) if isinstance(ckpt_obj, dict) and 't_dim' in ckpt_obj else 128
    v_backbone = TinyCNN_Feature(feat_dim=v_dim, width=32).to(DEVICE)
    t_encoder  = FrozenCLIPTextEncoder(name=clip_name, device=DEVICE)
    model = MultiModalDecoupledCLIP(
        v_backbone=v_backbone,
        t_encoder=t_encoder,
        v_dim=v_dim,
        t_in_dim=512,
        t_dim=t_dim
    ).to(DEVICE)
    return model


def load_state_flexible(model, obj):
    st, _ = _get_state_and_keys(obj)
    new_state = {}
    for k, v in st.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[CKPT] missing keys: {list(missing)[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[CKPT] unexpected keys: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")


def load_model(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model_from_ckpt_meta(ckpt)
    load_state_flexible(model, ckpt)
    model.eval()
    return model, ckpt


# ========= Image preprocessing (same as evaluation) =========
def preprocess_img(img_uint8, size_hw=(160, 160)):
    img = cv2.resize(img_uint8, size_hw, interpolation=cv2.INTER_AREA).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return torch.from_numpy(img).to(DEVICE)


# ========= Read/use action_scale / action_bias =========
def get_action_scale_bias(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        a_scale = np.asarray(f["data"].attrs["action_scale"], dtype=np.float32)
        a_bias  = np.asarray(f["data"].attrs["action_bias"],  dtype=np.float32)
    if a_scale.shape != (3,) or a_bias.shape != (3,):
        raise RuntimeError(
            f"action_scale/bias should have shape (3,), got {a_scale.shape}, {a_bias.shape}"
        )
    return a_scale, a_bias


def denorm_actions(norm_xyz: np.ndarray, a_scale: np.ndarray, a_bias: np.ndarray):
    return (norm_xyz - a_bias) / a_scale


# ========= Load scale/bias & model =========
a_scale, a_bias = get_action_scale_bias(DS)
ckpt_path = find_ckpt()
print(f"[Info] Using checkpoint: {ckpt_path}")
model, ckpt = load_model(ckpt_path)
IMG_SIZE = int(ckpt.get("img_size", 160)) if isinstance(ckpt, dict) else 160

# ========= Build world / task / camera =========
my_world = World(stage_units_in_meters=1.0)
my_task  = TimberAssembly(target_position=np.array([-0.2, 0.652155, 0.0515 / 2]))
my_world.add_task(my_task)
my_world.reset()

task_params = my_task.get_params()
robot_name  = task_params["robot_name"]["value"]
my_ur10 = my_world.scene.get_object(robot_name)
my_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=my_ur10.gripper,
    robot_articulation=my_ur10
)
articulation_controller = my_ur10.get_articulation_controller()

# Load additional scene
stage_utils.add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/factory")

# Timber object
timber_path = "/World/factory/Wood_block"
timber_prim = prims.get_prim_at_path(timber_path)
print(f"[Info] Timber prim: {timber_prim}")
timber_articulation = Articulation(timber_path)

# Camera (prefer existing camera; otherwise, create one)
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

# ========= Initial target (yaw will be overridden after prediction) =========
placing_position = np.array([0.0, 0.0, Z_FIXED], dtype=float)
rotation_angles  = np.array([0.0, np.pi / 2, np.pi / 4], dtype=float)

# ========= Main simulation loop =========
i = 0
reset_needed = False
simulation_context = SimulationContext()
pred_applied_once = False

print("[Info] Starting simulation loop...")
print(f"[Info] Using instruction: \"{TEXT_INSTR}\"")

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

        # ======= Prediction trigger: single evaluation-style prediction at TARGET_IMG_FRAME =======
        need_predict = PRED_USE_ALL_STEPS or (i == TARGET_IMG_FRAME and not pred_applied_once)

        if need_predict:
            img_rgb = camera0.get_rgb()
            if img_rgb is not None and img_rgb.size:
                x_img = preprocess_img(img_rgb, size_hw=(IMG_SIZE, IMG_SIZE))
                with torch.no_grad():
                    pred_norm = model(x_img, [TEXT_INSTR]).cpu().numpy().reshape(3)  # [-1,1]
                xy_yaw = denorm_actions(pred_norm, a_scale, a_bias)  # world coordinates
                xw, yw, yaw = float(xy_yaw[0]), float(xy_yaw[1]), float(xy_yaw[2])
                # roll/pitch fixed; yaw predicted by model (same as evaluation)
                placing_position = np.array([xw, yw, Z_FIXED], dtype=float)
                rotation_angles  = np.array([0.0, np.pi / 2, yaw], dtype=float)
                pred_applied_once = True
                print(f"[PRED @ frame={i}] norm={pred_norm}  -> world(x,y,yaw)={xy_yaw}")

        # ======= Controller: for t < 10 go via a waypoint; then go to the predicted target =======
        if t < 10:
            actions = my_controller.forward(
                picking_position=timber_articulation.get_local_pose()[0],
                placing_position=np.array([0.3, 0.97194, 0.64551]),
                current_joint_positions=observations[robot_name]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0.02]),
                end_effector_orientation=euler_angles_to_quat(np.array([0, np.pi / 2, 0])),
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
            print("Pick-and-place sequence is complete.")
        articulation_controller.apply_action(actions)

    # ====== Camera: write video frames ======
    img_rgb = camera0.get_rgb()
    if img_rgb is not None and img_rgb.size:
        if video_writer is None:
            h, w = img_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (w, h))
            if not video_writer.isOpened():
                print(f"[WARN] Cannot open VideoWriter: {VIDEO_PATH}")
                video_writer = None
        if video_writer is not None:
            video_writer.write(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Debug print current target
    picking_position_world = timber_articulation.get_world_pose()[0]
    print(
        f"[Frame {i:04d}] t={simulation_context.current_time:.3f}s\n"
        f"  picking_position (world): {picking_position_world}\n"
        f"  placing_position (world): {placing_position}\n"
        f"  rotation_angles [roll, pitch, yaw]: {rotation_angles}"
    )
    i += 1

# ====== Save final frame + release resources ======
last = camera0.get_rgb()
if last is not None and last.size:
    cv2.imwrite(LAST_FRAME_PATH, cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    print(f"[OK] Saved final frame: {LAST_FRAME_PATH}")

try:
    if video_writer is not None:
        video_writer.release()
        print(f"[OK] Saved video: {VIDEO_PATH}")
except Exception as e:
    print(f"[WARN] Failed to release VideoWriter: {e}")

simulation_app.close()
gc.collect()
print("[Done]")

