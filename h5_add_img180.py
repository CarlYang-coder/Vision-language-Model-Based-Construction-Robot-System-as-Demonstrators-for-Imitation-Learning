#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, h5py, numpy as np

# ==== Config ====
H5_PATH    = os.path.expanduser('~/Downloads/Data_Collecting/bc_dataset_from616.hdf5')
FRAME_NAME = 'frame_0180.png'         # Which frame to use
TARGET_HW  = (512, 512)               # Input size used in the training script
OVERWRITE  = False                    # True = overwrite even if dataset already exists

# ==== Utility: safe image reader (only needs numpy; cv2 optional) ====
def imread_rgb(path, target_hw=None):
    try:
        import imageio.v2 as iio
        img = iio.imread(path)
    except Exception:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise FileNotFoundError(f"cv2.imread failed: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if target_hw is not None and (img.shape[0] != target_hw[0] or img.shape[1] != target_hw[1]):
        try:
            import cv2
            img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image
            img = np.array(
                Image.fromarray(img).resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            )
    return img.astype(np.uint8)

def main():
    if not os.path.isfile(H5_PATH):
        raise FileNotFoundError(f"HDF5 not found: {H5_PATH}")

    wrote, skipped, missing = 0, 0, 0
    with h5py.File(H5_PATH, 'r+') as f:
        if 'data' not in f:
            raise RuntimeError("HDF5 is missing the /data group")
        for demo in sorted(
            f['data'].keys(),
            key=lambda s: int(s.split('_')[-1]) if s.startswith('demo_') else s
        ):
            g = f['data'][demo]
            if not isinstance(g, h5py.Group):
                continue

            N = int(g.attrs.get('num_samples', 0))
            if N <= 0:
                continue

            # Locate simulation directory
            sim_dir = g.attrs.get('sim_dir', None)
            if isinstance(sim_dir, bytes):
                sim_dir = sim_dir.decode('utf-8')
            if not sim_dir or not os.path.isdir(sim_dir):
                # Fallback: guess a directory by name
                sim_name = g.attrs.get('sim_name', demo)
                if isinstance(sim_name, bytes):
                    sim_name = sim_name.decode('utf-8')
                base = os.path.dirname(H5_PATH)
                guess = os.path.join(base, sim_name)
                sim_dir = guess if os.path.isdir(guess) else None

            if not sim_dir:
                print(f"[MISS] {demo}: cannot locate sim_dir (not in attrs either), skip")
                missing += 1
                continue

            img_path = os.path.join(sim_dir, 'frame', FRAME_NAME)
            if not os.path.isfile(img_path):
                print(f"[MISS] {demo}: frame image not found {img_path}")
                missing += 1
                continue

            # Existing dataset handling
            if 'img180' in g and not OVERWRITE:
                print(f"[Skip] {demo}: img180 already exists (N={g['img180'].shape[0]}), skip")
                skipped += 1
                continue
            if 'img180' in g and OVERWRITE:
                del g['img180']

            # Read image & replicate to (N, H, W, 3)
            try:
                img = imread_rgb(img_path, TARGET_HW)   # (H,W,3) uint8
            except Exception as e:
                print(f"[ERR ] {demo}: failed to read image {img_path} -> {e}")
                missing += 1
                continue

            img_stack = np.repeat(img[None, ...], N, axis=0)  # (N,H,W,3)
            g.create_dataset('img180', data=img_stack, compression='gzip', dtype='uint8')

            print(
                f"[OK] {demo}: wrote img180 shape=(N={N},{img.shape[0]},{img.shape[1]},3) "
                f"from {img_path}"
            )
            wrote += 1

    print(f"\nDone. wrote={wrote}, skipped={skipped}, missing={missing}.")
    print("✅ Reminder: in the training script, enable image-only observation "
          "(rgb=['img180'], low_dim=[])")

if __name__ == '__main__':
    main()

