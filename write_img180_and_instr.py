#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, h5py, json, math
import numpy as np

# ==== Paths & flags ====
H5_PATH   = os.path.expanduser('~/Downloads/Data_Collecting/bc_dataset_from616.hdf5')
OVERWRITE = True  # True = overwrite existing attrs['text']; False = skip if already present

# ==== Text templates (two diagonal patterns) ====
TEXT_A = "Place the timber along the diagonal of the upper left corner and the bottom right corner."
TEXT_B = "Place the timber along the diagonal of the bottom left corner and the upper right corner."

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def get_action_scale_bias(f):
    a_scale = np.asarray(f['data'].attrs['action_scale'], dtype=np.float32)
    a_bias  = np.asarray(f['data'].attrs['action_bias'], dtype=np.float32)
    if a_scale.shape != (3,) or a_bias.shape != (3,):
        raise RuntimeError("action_scale / action_bias must both have shape (3,)")
    return a_scale, a_bias

def yaw_world_from_norm(yaw_norm, a_scale, a_bias):
    # Actions were normalized jointly for (x, y, yaw), so reconstruct a 3D vector then de-normalize
    v = np.array([0.0, 0.0, yaw_norm], dtype=np.float32)
    w = (v - a_bias) / a_scale
    return wrap_pi(float(w[2]))

def kmeans_2_on_unit(vecs, iters=20):
    """
    Simple k-means with k=2 on the unit circle.

    vecs: (N, 2) = [cos(theta), sin(theta)]
    Returns:
        labels:  (N,) cluster assignment
        centers: (2, 2) cluster centers on unit circle
    """
    N = vecs.shape[0]
    # Use two fixed directions for initialization (-135° and -45°) for robustness
    init = np.array([
        [math.cos(-3*np.pi/4), math.sin(-3*np.pi/4)],  # -135°
        [math.cos(-np.pi/4),   math.sin(-np.pi/4)],    #  -45°
    ], dtype=np.float32)

    centers = init.copy()
    for _ in range(iters):
        sims = vecs @ centers.T                           # (N,2)
        labels = sims.argmax(axis=1)                      # (N,)
        new_centers = np.zeros_like(centers)
        for k in range(2):
            m = (labels == k)
            if m.any():
                c = vecs[m].mean(axis=0)
                n = np.linalg.norm(c) + 1e-8
                new_centers[k] = c / n
            else:
                new_centers[k] = centers[k]
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return labels, centers

def main():
    with h5py.File(H5_PATH, 'r+') as f:
        if 'data' not in f:
            raise RuntimeError("HDF5 is missing /data group.")

        a_scale, a_bias = get_action_scale_bias(f)

        # Collect the first-step yaw_world per demo
        demos, yaws = [], []
        for demo in f['data'].keys():
            g = f['data'][demo]
            if not isinstance(g, h5py.Group):
                continue
            if 'actions' not in g or len(g['actions']) == 0:
                continue
            yaw_norm = float(g['actions'][0][2])
            yaw_w = yaw_world_from_norm(yaw_norm, a_scale, a_bias)
            demos.append(demo)
            yaws.append(yaw_w)

        if not demos:
            raise RuntimeError("No usable demos (missing actions).")

        yaws = np.array(yaws, dtype=np.float32)
        vecs = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)  # (N,2) on the unit circle

        # k=2 means on unit circle
        labels, centers = kmeans_2_on_unit(vecs)

        # Key rule: the cluster whose center is closer to -pi/4 is mapped to TEXT_B (BL -> UR)
        center_angles = np.arctan2(centers[:,1], centers[:,0])  # (-π, π]
        d_to_neg_pi4 = np.abs((center_angles - (-np.pi/4) + np.pi) % (2*np.pi) - np.pi)
        b_idx = int(d_to_neg_pi4.argmin())  # cluster near -pi/4 -> TEXT_B
        a_idx = 1 - b_idx                   # the other cluster      -> TEXT_A

        cnt_new, cnt_skip = 0, 0
        for demo, lab in zip(demos, labels):
            g = f['data'][demo]
            if (not OVERWRITE) and ('text' in g.attrs):
                cnt_skip += 1
                continue
            # Mapping rule: lab == b_idx -> TEXT_B; otherwise TEXT_A
            g.attrs['text'] = TEXT_B if lab == b_idx else TEXT_A
            cnt_new += 1

        meta = {
            "total_demos": len(demos),
            "written": cnt_new,
            "skipped_existing": cnt_skip,
            "center_angles_deg": (center_angles * 180 / np.pi).tolist(),
            "cluster_sizes": [int((labels==0).sum()), int((labels==1).sum())],
            "map_near_-pi/4_to": "TEXT_B",  # record mapping rule so it is not confusing later
            "text_A": TEXT_A,
            "text_B": TEXT_B,
            "overwrite": OVERWRITE,
        }
        print(json.dumps(meta, indent=2))

        # Optional: re-count the distribution of written texts
        dist = {}
        for demo in f['data'].keys():
            g = f['data'][demo]
            if not isinstance(g, h5py.Group):
                continue
            t = g.attrs.get('text', None)
            if t is None:
                continue
            if isinstance(t, bytes):
                t = t.decode('utf-8')
            dist[t] = dist.get(t, 0) + 1
        print("\n[TEXT DISTRIBUTION]")
        for k, v in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {repr(k)}: {v}")

if __name__ == "__main__":
    main()

