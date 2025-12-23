#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, re, pickle, json, h5py, numpy as np
from tqdm import tqdm

BASE = os.path.expanduser('~/Downloads/Data_Collecting')
OUT  = os.path.join(BASE, 'bc_dataset_from616.hdf5')
START_FRAME = 201

# ---------- Load pkl ----------
def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

def extract_with_keys(obj):
    """
    Compatible with both dict{frame_idx: {...}} and list[{...}]

    Returns:
        keys:           list[int], sorted by frame index
        q:              (N, 6) joint_position
        ee:             (N, 6) end-effector pose [x,y,z,roll,pitch,yaw] (from ee_pose)
        ee_next_xyyaw:  (N, 3) target [x,y,yaw] (from ee_pose_next), or None if missing
        tgt:            target_position_next (fallback if present), otherwise None
    """
    def _ee_xyz_from_pose(d):      # ee_pose -> [x,y,z,roll,pitch,yaw]
        return [d["x"], d["y"], d["z"], d["roll"], d["pitch"], d["yaw"]]

    def _ee_next_xyyaw(d):        # ee_pose_next -> [x,y,yaw]
        return [d["x"], d["y"], d["yaw"]]

    if isinstance(obj, dict):
        keys = sorted(int(k) for k in obj.keys())
        q   = np.array([obj[k]["joint_position"] for k in keys], dtype=np.float32)
        ee  = np.array([_ee_xyz_from_pose(obj[k]["ee_pose"]) for k in keys], dtype=np.float32)

        has_ee_next = all(("ee_pose_next" in obj[k]) for k in keys)
        ee_next_xyyaw = (
            np.array([_ee_next_xyyaw(obj[k]["ee_pose_next"]) for k in keys],
                     dtype=np.float32) if has_ee_next else None
        )

        tgt = None
        if "target_position_next" in obj[keys[0]]:
            tgt = np.array(obj[keys[0]]["target_position_next"], dtype=np.float32)
        return keys, q, ee, ee_next_xyyaw, tgt

    elif isinstance(obj, (list, tuple)):
        keys = list(range(len(obj)))
        q   = np.array([obj[i]["joint_position"] for i in keys], dtype=np.float32)
        ee  = np.array([_ee_xyz_from_pose(obj[i]["ee_pose"]) for i in keys], dtype=np.float32)

        has_ee_next = len(obj) > 0 and all(("ee_pose_next" in fr) for fr in obj)
        ee_next_xyyaw = (
            np.array([_ee_next_xyyaw(obj[i]["ee_pose_next"]) for i in keys],
                     dtype=np.float32) if has_ee_next else None
        )

        tgt = None
        if len(obj) > 0 and "target_position_next" in obj[0]:
            tgt = np.array(obj[0]["target_position_next"], dtype=np.float32)
        return keys, q, ee, ee_next_xyyaw, tgt

    else:
        raise TypeError("Unsupported pickle structure")


def valid_transition_indices(keys):
    """Keep only adjacent transitions where both frame indices are >= START_FRAME."""
    return [i for i in range(len(keys)-1) if (keys[i] >= START_FRAME and keys[i+1] >= START_FRAME)]

# ---------- Infer Simulation coordinates from directory name or pkl as fallback ----------
_num = r'([+-]?\d*(?:\.\d+)?(?:[eE][+-]?\d+)?)'

PATTERNS = [
    re.compile(rf'\[\s*{_num}\s*[,xX; ]\s*{_num}\s*\]'),   # match “[x,y]” anywhere (handles “Simulation_[x,y]_...”)
    re.compile(rf'Simulation[^\[]*\[\s*{_num}\s*[,xX; ]\s*{_num}\s*\]'),
    re.compile(rf'Simulation.*?_x\s*{_num}.*?_y\s*{_num}', re.IGNORECASE),
    re.compile(rf'Simulation.*?{_num}[_\-xX ]+{_num}$'),
]

def parse_sim_xy_from_dir(dirname):
    """
    Prefer parsing 2D coordinates from the directory name.
    Returns np.array([x,y], dtype=float) or None.
    """
    base = os.path.basename(dirname)
    for pat in PATTERNS:
        m = pat.search(base)
        if not m:
            continue
        # Use the last captured pair
        xs = [float(g) for g in m.groups()[-2:]]
        return np.array(xs, dtype=np.float32)
    return None

# ---------- Pass 1: scan to get action range (after cropping) ----------
demo_dirs = sorted(glob.glob(os.path.join(BASE, 'Simulation*')))
act_mins, act_maxs = [], []
kept_total, demos_with_data = 0, 0

for d in tqdm(demo_dirs, desc='Scanning'):
    pkl_path = os.path.join(d, 'data.pkl')
    if not os.path.exists(pkl_path):
        continue
    obj = load_pkl(pkl_path)
    keys, q, ee, ee_next_xyyaw, _ = extract_with_keys(obj)
    idx = valid_transition_indices(keys)
    if not idx or ee_next_xyyaw is None:
        continue

    # action_raw: for step i, the target (x, y, yaw), aligned with obs[i]
    action_raw = ee_next_xyyaw[idx]           # shape (M, 3)

    act_mins.append(action_raw.min(axis=0))
    act_maxs.append(action_raw.max(axis=0))
    kept_total += len(idx)
    demos_with_data += 1

if kept_total == 0:
    raise RuntimeError(f"No transitions remain after cropping to frame >= {START_FRAME}.")

act_min = np.min(np.stack(act_mins, 0), axis=0)
act_max = np.max(np.stack(act_maxs, 0), axis=0)
rng = np.maximum(act_max - act_min, 1e-8)
act_scale = 2.0 / rng
act_bias  = -1.0 - act_min * act_scale   # map [min,max] -> [-1,1]

print(f"Demos kept: {demos_with_data}/{len(demo_dirs)} | Transitions kept: {kept_total}")
print("Action min:", act_min)
print("Action max:", act_max)

# ---------- Pass 2: write HDF5 (including Simulation_xy) ----------
if os.path.exists(OUT):
    os.remove(OUT)
h5 = h5py.File(OUT, 'w')
data_grp = h5.create_group('data')

def write_demo(g, q, ee, ee_next_xyyaw, keys, act_scale, act_bias, sim_xy):
    """
    action = {x,y,yaw} absolute target from ee_pose_next, linearly scaled to [-1,1]
    """
    idx = valid_transition_indices(keys)
    if not idx or ee_next_xyyaw is None:
        g.attrs['num_samples'] = 0
        return 0

    q0, q1   = q[:-1][idx],  q[1:][idx]
    ee0, ee1 = ee[:-1][idx], ee[1:][idx]

    action_raw = ee_next_xyyaw[idx]                # (N,3)
    actions    = action_raw * act_scale + act_bias # normalized to [-1,1]
    N = actions.shape[0]

    g.attrs['num_samples'] = int(N)
    g.attrs['start_frame'] = START_FRAME
    g.create_dataset('actions', data=actions, compression='gzip')

    obs = g.create_group('obs')
    obs.create_dataset('qpos',   data=q0,  compression='gzip')
    obs.create_dataset('ee_pos', data=ee0, compression='gzip')

    # Simulation_xy (keep the same logic)
    if sim_xy is None:
        sim_mat = np.full((N, 2), np.nan, dtype=np.float32)
    else:
        sim_mat = np.repeat(sim_xy.reshape(1, 2), N, axis=0)
    obs.create_dataset('Simulation_xy', data=sim_mat, compression='gzip')

    nxt = g.create_group('next_obs')
    nxt.create_dataset('qpos',   data=q1,  compression='gzip')
    nxt.create_dataset('ee_pos', data=ee1, compression='gzip')
    nxt.create_dataset('Simulation_xy', data=sim_mat.copy(), compression='gzip')

    dones = np.zeros((N,), dtype=np.uint8); dones[-1] = 1
    g.create_dataset('dones',   data=dones)
    g.create_dataset('rewards', data=np.zeros((N,), dtype=np.float32))
    return N

total_samples, demo_idx = 0, 0
for d in tqdm(demo_dirs, desc='Writing'):
    pkl_path = os.path.join(d, 'data.pkl')
    if not os.path.exists(pkl_path):
        continue
    obj = load_pkl(pkl_path)
    keys, q, ee, ee_next_xyyaw, tgt0 = extract_with_keys(obj)

    # Parse Simulation coordinates: prefer directory name, otherwise use target_position_next[:2]
    sim_xy = parse_sim_xy_from_dir(d)
    if sim_xy is None and tgt0 is not None and tgt0.size >= 2:
        sim_xy = np.asarray(tgt0[:2], dtype=np.float32)

    grp_name = f'demo_{demo_idx}'
    grp = data_grp.create_group(grp_name)
    n = write_demo(grp, q, ee, ee_next_xyyaw, keys, act_scale, act_bias, sim_xy)
    if n > 0:
        # Add mapping from HDF5 demo to source Simulation_* directory
        grp.attrs['sim_dir']  = d
        grp.attrs['sim_name'] = os.path.basename(d)
        print(f"[Map] {grp_name}  <--  {grp.attrs['sim_name']}   (samples={n})")
        total_samples += n
        demo_idx += 1
    else:
        del data_grp[grp_name]  # drop empty

data_grp.attrs['total'] = int(total_samples)
# Record the observation keys (Simulation_xy may contain NaN in some demos)
data_grp.attrs['env_args'] = json.dumps({
    "env_name": "CustomIsaacSim",
    "env_type": "custom",
    "env_kwargs": {
        "obs_keys": ["qpos", "ee_pos", "Simulation_xy"],
        "action":   "ee_pose_next_{x,y,yaw}_norm[-1,1]",
        "action_dim": 3,
        "start_frame": START_FRAME
    }
})
# Stash normalization for deployment
data_grp.attrs['action_min']   = act_min
data_grp.attrs['action_max']   = act_max
data_grp.attrs['action_scale'] = act_scale
data_grp.attrs['action_bias']  = act_bias

h5.close()
print(f"✅ Wrote {demo_idx} demos, {total_samples} cropped transitions to {OUT}")

# ----- Quick sanity check: print shapes from the first demo -----
import h5py, numpy as np

if os.path.exists(OUT):
    with h5py.File(OUT, "r") as f:
        data = f.get("data")
        if data is None or len(data.keys()) == 0:
            print("[CHECK] data group empty.")
        else:
            # Take the first demo (or the lexicographically first one if demo_0 does not exist)
            demo_names = sorted(list(data.keys()))
            demo0 = demo_names[0]
            g0 = data[demo0]

            act_shape = g0["actions"].shape
            ee_shape  = g0["obs"]["ee_pos"].shape
            q_shape   = g0["obs"]["qpos"].shape

            print(f"[CHECK] {demo0}: actions shape = {act_shape} (should be (N, 3))")
            print(f"[CHECK] {demo0}: obs/ee_pos shape = {ee_shape} (3 or 6 depending on your choice)")
            print(f"[CHECK] {demo0}: obs/qpos shape = {q_shape}")

            # Optional: assert shapes are correct
            assert act_shape[1] == 3, f"actions second dim should be 3, got {act_shape[1]}"
            # If you expect ee_pos to be 3D, enable the next line; if 6D, change 3->6.
            # assert ee_shape[1] == 3, f"ee_pos second dim should be 3, got {ee_shape[1]}"

            # Optional: print first few rows to inspect value ranges
            A = g0["actions"][:5]
            print("[CHECK] actions sample (first 5 rows):\n", np.array(A))
else:
    print(f"[CHECK] Output not found: {OUT}")

