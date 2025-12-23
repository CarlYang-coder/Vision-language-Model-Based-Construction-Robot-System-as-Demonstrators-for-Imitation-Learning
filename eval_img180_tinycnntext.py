#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, re, h5py, numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F

# ========= Basic paths =========
DS = os.path.expanduser('~/Downloads/Data_Collecting/bc_dataset_from616.hdf5')
OUT_DIR = os.path.expanduser('~/Downloads/Data_Collecting/pred_img180_xyyaw_from616')
os.makedirs(OUT_DIR, exist_ok=True)

# Select run dir / checkpoint file (default: your current multimodal experiment)
RUN_DIR = os.path.expanduser(
    '~/Downloads/robomimic_runs/simple_img2xyyaw_tiny/decoupled_tinycnn_clip_xy_from_img_yaw_from_text'
)
CKPT = ""   # empty -> auto search

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========= Text (fallback to default if missing) =========
TEXT_DEFAULT = "move to target"

# ========= Debug options =========
DO_AB_TEST = True   # For one image, run an A/B text experiment with "turn left"/"turn right" (only once, print only)

# ========= Modules (names consistent with training) =========
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.a(self.b(self.c(x)))


class DepthwiseSeparable(nn.Module):
    # Note: naming db/pb matches training script
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=s, padding=1, groups=in_ch, bias=False
        )
        self.db = nn.BatchNorm2d(in_ch)
        self.da = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pb = nn.BatchNorm2d(out_ch)
        self.pa = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.da(self.db(self.dw(x)))
        x = self.pa(self.pb(self.pw(x)))
        return x


class TinyCNN_Feature(nn.Module):
    # Same structure as training: b2/b3/b4/b5
    def __init__(self, feat_dim=128, width=32):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 2, width * 4  # 32,64,64,128
        self.stem = ConvBNAct(3, c1, k=3, s=2, p=1)        # 160->80
        self.b2   = DepthwiseSeparable(c1, c2, s=2)        # 80->40
        self.b3   = DepthwiseSeparable(c2, c3, s=1)        # 40->40
        self.b4   = DepthwiseSeparable(c3, c4, s=2)        # 40->20
        self.b5   = DepthwiseSeparable(c4, c4, s=1)        # 20->20
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
        feat = self.proj(x)  # [B, feat_dim]
        return feat


# ---- Frozen CLIP text encoder (force safetensors) ----
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.tokenizer    = CLIPTokenizer.from_pretrained(name)
        self.text_encoder = CLIPTextModel.from_pretrained(
            name,
            use_safetensors=True,       # important: use .safetensors only, avoid torch.load restriction
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


# ---- Two head structures ----
class MultiModalConcatMLP(nn.Module):
    # Old: concatenation-based model (single head)
    def __init__(
        self,
        v_backbone: TinyCNN_Feature,
        t_encoder: FrozenCLIPTextEncoder,
        v_dim=128,
        t_in_dim=512,
        t_dim=128,
        out_dim=3
    ):
        super().__init__()
        self.v_backbone = v_backbone
        self.t_encoder  = t_encoder
        self.text_proj  = nn.Sequential(
            nn.Linear(t_in_dim, t_dim),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
        fused_dim = v_dim + t_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )

    def forward(self, images, texts):
        v_feat = self.v_backbone(images)
        with torch.no_grad():
            t_raw = self.t_encoder(texts)
        t_feat = self.text_proj(t_raw)
        fused  = torch.cat([v_feat, t_feat], dim=1)
        pred   = self.head(fused)  # [B,3] = [x,y,yaw_norm]
        return pred, v_feat, t_feat


class MultiModalDecoupledCLIP(nn.Module):
    # New: decoupled model (xy_head / yaw_head), convert cos/sin back to yaw_norm during inference
    def __init__(
        self,
        v_backbone: TinyCNN_Feature,
        t_encoder: FrozenCLIPTextEncoder,
        v_dim=128,
        t_in_dim=512,
        t_dim=128
    ):
        super().__init__()
        self.v_backbone = v_backbone
        self.t_encoder  = t_encoder
        self.text_proj  = nn.Sequential(
            nn.Linear(t_in_dim, t_dim),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
        self.xy_head    = nn.Sequential(
            nn.Linear(v_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )
        self.yaw_head   = nn.Sequential(
            nn.Linear(t_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)  # [cos,sin]
        )

    def forward(self, images, texts):
        v_feat = self.v_backbone(images)       # [B, v_dim]
        with torch.no_grad():
            t_raw = self.t_encoder(texts)      # [B,512]
        t_feat = self.text_proj(t_raw)         # [B, t_dim]
        pred_xy = self.xy_head(v_feat)         # [B,2]
        pred_cs = self.yaw_head(t_feat)        # [B,2] (cos, sin)
        # Normalize and convert to yaw_norm ∈ [-1,1]
        pred_cs = pred_cs / (pred_cs.norm(dim=1, keepdim=True) + 1e-8)
        yaw = torch.atan2(pred_cs[:, 1], pred_cs[:, 0]) / torch.pi   # [-1,1]
        pred = torch.cat([pred_xy, yaw.unsqueeze(1)], dim=1)         # [B,3]
        return pred, pred_xy, pred_cs


# ========= Image-only TinyCNN (for older checkpoints) =========
class TinyCNN_ImageOnly(nn.Module):
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
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1   = nn.Linear(64, 64)
        self.fc2   = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


# ========= Checkpoint search / load =========
def _epoch_num(p):
    m = re.search(r'epoch[_\-](\d+)', os.path.basename(p))
    return int(m.group(1)) if m else -1


def find_ckpt():
    if CKPT and os.path.isfile(CKPT):
        print(f"[CKPT] Use explicit: {CKPT}")
        return CKPT
    pats = ("**/*best*.pt", "**/*best*.pth", "**/epoch_*.pt", "**/epoch-*.pt", "**/*.pt", "**/*.pth")
    cands = []
    for pat in pats:
        cands += glob.glob(os.path.join(RUN_DIR, pat), recursive=True)
    if not cands:
        raise FileNotFoundError(f"Checkpoint not found: check RUN_DIR={RUN_DIR} or set CKPT manually")
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
    v_dim     = int(ckpt_obj.get('v_dim', 128)) if 'v_dim' in ckpt_obj else 128
    t_dim     = int(ckpt_obj.get('t_dim', 128)) if 't_dim' in ckpt_obj else 128
    v_backbone = TinyCNN_Feature(feat_dim=v_dim, width=32).to(DEVICE)
    t_encoder  = FrozenCLIPTextEncoder(name=clip_name, device=DEVICE)

    if any(k.startswith('xy_head.') for k in keys) or any(k.startswith('yaw_head.') for k in keys):
        print("[CKPT] Detected decoupled structure (xy_head / yaw_head)")
        model = MultiModalDecoupledCLIP(
            v_backbone=v_backbone,
            t_encoder=t_encoder,
            v_dim=v_dim,
            t_in_dim=512,
            t_dim=t_dim
        ).to(DEVICE)
    elif any(k.startswith('head.') for k in keys):
        print("[CKPT] Detected concat structure (head)")
        model = MultiModalConcatMLP(
            v_backbone=v_backbone,
            t_encoder=t_encoder,
            v_dim=v_dim,
            t_in_dim=512,
            t_dim=t_dim,
            out_dim=3
        ).to(DEVICE)
    else:
        print("[CKPT] No multimodal heads detected, falling back to old image-only TinyCNN")
        model = TinyCNN_ImageOnly(in_ch=3, out_dim=3).to(DEVICE)
    return model


def load_state_flexible(model, obj):
    st, _ = _get_state_and_keys(obj)
    # Remove DP prefix
    new_state = {}
    for k, v in st.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v
    # Flexible load: print missing/unexpected keys but don't raise
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[CKPT] missing keys: {list(missing)[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[CKPT] unexpected keys: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")


def load_model(ckpt_path):
    # Compatible with older torch (no weights_only argument)
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model_from_ckpt_meta(ckpt)
    load_state_flexible(model, ckpt)
    model.eval()
    return model, ckpt


# ========= Preprocessing =========
def preprocess_img(img_uint8, size=(160, 160)):
    img = cv2.resize(img_uint8, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return torch.from_numpy(img).to(DEVICE)


# ========= Read action scale/bias for de-normalization =========
def get_action_scale_bias(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        a_scale = np.asarray(f["data"].attrs["action_scale"], dtype=np.float32)
        a_bias  = np.asarray(f["data"].attrs["action_bias"],  dtype=np.float32)
    if a_scale.shape != (3,) or a_bias.shape != (3,):
        raise RuntimeError("action_scale/bias shape must be (3,)")
    return a_scale, a_bias


def denorm_actions(norm_xyz: np.ndarray, a_scale: np.ndarray, a_bias: np.ndarray):
    return (norm_xyz - a_bias) / a_scale


# ========= Main pipeline =========
def main():
    # 0) De-normalization parameters
    a_scale, a_bias = get_action_scale_bias(DS)

    # 1) Read test list
    with h5py.File(DS, 'r') as f:
        if 'mask' not in f or 'test' not in f['mask']:
            raise RuntimeError("HDF5 is missing /mask/test.")
        test_demos = [
            n.decode('utf-8') if isinstance(n, bytes) else n
            for n in f['mask']['test'][...]
        ]
    print(f"[Info] TEST demos: {len(test_demos)}")

    # 2) Load model
    ckpt_path = find_ckpt()
    print(f"[Info] Using checkpoint: {ckpt_path}")
    model, ckpt = load_model(ckpt_path)

    # Flags for model type
    is_decoupled = isinstance(model, MultiModalDecoupledCLIP)
    is_concat    = isinstance(model, MultiModalConcatMLP)

    # Stats containers
    yaw_pred_norm_all, yaw_true_norm_all = [], []
    yaw_pred_world_all, yaw_true_world_all = [], []

    MAX_PRINT = 50
    saved, skipped, printed = 0, 0, 0
    did_ab_once = False

    # 3) Inference and saving
    for demo in test_demos:
        with h5py.File(DS, 'r') as f:
            g = f['data'][demo]
            if "img180" not in g or "actions" not in g:
                print(f"[Skip] {demo}: missing img180 or actions")
                skipped += 1
                continue
            imgs = np.asarray(g["img180"][...], dtype=np.uint8)    # (N,H,W,3)
            acts = np.asarray(g["actions"][...], dtype=np.float32) # (N,3)

            # Text (for multimodal only; follow training logic: attrs['text'] has highest priority)
            text_str = TEXT_DEFAULT
            if is_decoupled or is_concat:
                s0 = None
                # 1) Preferred: group attribute 'text' (consistent with training)
                if 'text' in g.attrs:
                    v = g.attrs['text']
                    if isinstance(v, bytes):
                        s0 = v.decode('utf-8')
                    else:
                        s0 = str(v)
                # 2) Fallback: dataset-style 'instr'
                if not s0:
                    if 'instr' in g:
                        s0 = g['instr'][0]
                    elif 'obs' in g and 'instr' in g['obs']:
                        s0 = g['obs']['instr'][0]
                    if isinstance(s0, bytes):
                        s0 = s0.decode('utf-8')
                if isinstance(s0, str) and len(s0.strip()) > 0:
                    text_str = s0.strip()

        N = imgs.shape[0]
        if N == 0:
            print(f"[Skip] {demo}: N=0")
            skipped += 1
            continue

        # --- Important: actually used text for this demo --- #
        print(f"[{demo}] text_used = {text_str!r}")

        # Predict first frame
        x = preprocess_img(
            imgs[0],
            size=(int(ckpt.get("img_size", 160)), int(ckpt.get("img_size", 160)))
        )
        with torch.no_grad():
            if is_decoupled:
                # Optional: inspect text-branch feature norm (to verify text is changing)
                raw = model.t_encoder([text_str])
                t_feat = model.text_proj(raw)
                print(f"[{demo}] ||t_feat|| = {t_feat.norm().item():.6f}")

                pred1, pred_xy, pred_cs = model(x, [text_str])   # pred1: [1,3]
                pred1 = pred1
            elif is_concat:
                pred1, *_ = model(x, [text_str])                 # [1,3]
            else:
                pred1 = model(x)                                 # [1,3]
            pred1 = pred1.detach().cpu().numpy().reshape(3)      # normalized space [-1,1]

            # Single A/B text experiment (only once)
            if DO_AB_TEST and not did_ab_once and (is_decoupled or is_concat):
                for tt in ["turn left", "turn right"]:
                    if is_decoupled:
                        p_all, _, _ = model(x, [tt])
                    else:
                        p_all, *_ = model(x, [tt])
                    y_pred = float(p_all[0, 2].cpu().numpy())
                    print(
                        f"[A/B] text={tt!r} -> "
                        f"yaw_norm={y_pred:.6f}  yaw_world={y_pred*np.pi:.6f} rad"
                    )
                did_ab_once = True

        # Save the whole trajectory (repeat frame 0 prediction for all N steps)
        preds = np.clip(np.repeat(pred1[None, :], N, axis=0), -1.0, 1.0)
        out_path = os.path.join(OUT_DIR, f"{demo}.npy")
        np.save(out_path, preds)
        saved += 1

        # ====== Print / stats: first-step prediction vs GT (normalized & de-normalized) ======
        true1 = acts[0]  # normalized GT
        yaw_pred_norm_all.append(pred1[2])
        yaw_true_norm_all.append(true1[2])

        pred1_world = denorm_actions(pred1, a_scale, a_bias)
        true1_world = denorm_actions(true1, a_scale, a_bias)

        def wrap_pi(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        pred1_world[2] = wrap_pi(pred1_world[2])
        true1_world[2] = wrap_pi(true1_world[2])

        yaw_pred_world_all.append(pred1_world[2])
        yaw_true_world_all.append(true1_world[2])

        if printed < MAX_PRINT:
            print(f"[{demo}] step0 | pred_norm={pred1}  true_norm={true1}")
            print(f"           -> pred_world={pred1_world}  true_world={true1_world}")
            printed += 1

        print(f"[OK] {demo}: saved {preds.shape} -> {out_path}")

    # 4) Evaluation (still in normalized {x,y,yaw} space)
    print("\n=== Computing test loss (normalized {x,y,yaw}) ===")
    all_pred, all_true, per_demo = [], [], {}
    for demo in test_demos:
        pp = os.path.join(OUT_DIR, f"{demo}.npy")
        if not os.path.isfile(pp):
            continue
        P = np.load(pp).astype(np.float32)
        with h5py.File(DS, 'r') as f:
            g = f['data'][demo]
            if "actions" not in g:
                continue
            T = np.asarray(g["actions"][...], dtype=np.float32)

        if P.shape[0] == 0 or T.shape[0] == 0:
            continue

        # Evaluate only the first step
        P = P[:1]   # (1,3)
        T = T[:1]   # (1,3)

        diff = P - T
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        per_demo[demo] = {"MSE": mse, "MAE": mae, "num_steps": 1}
        all_pred.append(P)
        all_true.append(T)

    if not all_pred:
        raise RuntimeError("No evaluable samples found in test split.")

    P = np.concatenate(all_pred, axis=0)
    T = np.concatenate(all_true, axis=0)
    diff = P - T
    mse_dim = np.mean(diff**2, axis=0)
    mae_dim = np.mean(np.abs(diff), axis=0)
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    print(f"[TEST] MSE={mse:.6f}  MAE={mae:.6f}")

    # ====== Extra: yaw statistics (normalized & world) ======
    yaw_pred_norm_all = np.asarray(yaw_pred_norm_all, dtype=np.float32)
    yaw_true_norm_all = np.asarray(yaw_true_norm_all, dtype=np.float32)
    yaw_pred_world_all = np.asarray(yaw_pred_world_all, dtype=np.float32)
    yaw_true_world_all = np.asarray(yaw_true_world_all, dtype=np.float32)

    def stats(name, arr):
        return (
            f"{name}: mean={float(arr.mean()):.6f}, std={float(arr.std()):.6f}, "
            f"min={float(arr.min()):.6f}, max={float(arr.max()):.6f}, n={arr.size}"
        )

    print("\n=== Yaw stats (normalized) ===")
    print(stats("pred_yaw_norm", yaw_pred_norm_all))
    print(stats("true_yaw_norm", yaw_true_norm_all))

    print("\n=== Yaw stats (world, radians) ===")
    print(stats("pred_yaw_world", yaw_pred_world_all))
    print(stats("true_yaw_world", yaw_true_world_all))

    thr = np.deg2rad(5.0)  # 5 degrees
    target = -np.pi / 2
    near = np.abs((yaw_pred_world_all - target + np.pi) % (2 * np.pi) - np.pi) < thr
    ratio = float(near.mean())
    print(f"\n[Check] Fraction of predicted yaw in -pi/2±5°: {ratio*100:.2f}%")

    # 5) Save metrics
    summary = {
        "dataset": DS,
        "checkpoint": ckpt_path,
        "is_decoupled": bool(is_decoupled),
        "is_concat": bool(is_concat),
        "num_samples": int(P.shape[0]),
        "MSE_overall": mse,
        "MAE_overall": mae,
        "MSE_per_dim": mse_dim.tolist(),
        "MAE_per_dim": mae_dim.tolist(),
        "yaw_pred_norm_stats": {
            "mean": float(yaw_pred_norm_all.mean()),
            "std": float(yaw_pred_norm_all.std()),
            "min": float(yaw_pred_norm_all.min()),
            "max": float(yaw_pred_norm_all.max()),
            "n": int(yaw_pred_norm_all.size),
        },
        "yaw_true_norm_stats": {
            "mean": float(yaw_true_norm_all.mean()),
            "std": float(yaw_true_norm_all.std()),
            "min": float(yaw_true_norm_all.min()),
            "max": float(yaw_true_norm_all.max()),
            "n": int(yaw_true_norm_all.size),
        },
        "yaw_pred_world_stats": {
            "mean": float(yaw_pred_world_all.mean()),
            "std": float(yaw_pred_world_all.std()),
            "min": float(yaw_pred_world_all.min()),
            "max": float(yaw_pred_world_all.max()),
            "n": int(yaw_pred_world_all.size),
        },
        "yaw_true_world_stats": {
            "mean": float(yaw_true_world_all.mean()),
            "std": float(yaw_true_world_all.std()),
            "min": float(yaw_true_world_all.min()),
            "max": float(yaw_true_world_all.max()),
            "n": int(yaw_true_world_all.size),
        },
        "pct_pred_yaw_near_-pi/2_deg5": ratio,
        "out_dir": OUT_DIR,
        "run_dir": RUN_DIR,
    }
    with open(os.path.join(OUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"[TEST] MSE={mse:.6f}  MAE={mae:.6f}\n")
    print(f"[OK] wrote {os.path.join(OUT_DIR, 'test_metrics.txt')}")


if __name__ == "__main__":
    main()

