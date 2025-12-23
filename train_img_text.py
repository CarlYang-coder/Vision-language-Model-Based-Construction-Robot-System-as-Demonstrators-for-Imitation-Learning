import os, json, random, math
import h5py
import numpy as np
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, ImageOps

# ================== Path & Hyperparameters ==================
H5_PATH   = os.path.expanduser('~/Downloads/Data_Collecting/bc_dataset_from616.hdf5')
RUNS_DIR  = os.path.expanduser('~/Downloads/robomimic_runs/simple_img2xyyaw_tiny')
EXP_NAME  = 'decoupled_tinycnn_clip_xy_from_img_yaw_from_text'

BATCH_SZ  = 64
EPOCHS    = 50
LR        = 1e-3
WEIGHT_DECAY = 5e-4
IMG_SIZE  = 160
SEED      = 201


TEXT_DEFAULT = "move to target"                  # If demo has no text attr
CLIP_NAME    = "openai/clip-vit-base-patch32"
LAMBDA_YAW   = 3.0
LAMBDA_UNIT  = 0.1

# —— Data Augmentation —— #
USE_HFLIP    = False
USE_COLORJIT = False

os.makedirs(os.path.join(RUNS_DIR, EXP_NAME, 'models'), exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ================== H5 Scan & Mask ==================
def list_demos(f: h5py.File) -> List[str]:
    return [k for k in f['data'].keys() if isinstance(f['data'][k], h5py.Group)]

def has_img_and_actions(f: h5py.File, demo: str) -> Tuple[bool, str]:
    g = f['data'][demo]
    if 'img180' not in g: return False, 'missing img180'
    if 'actions' not in g: return False, 'missing actions'
    if len(g['img180']) == 0: return False, 'img180 empty'
    if len(g['actions']) == 0: return False, 'actions empty'
    if g['actions'].shape[-1] != 3: return False, f'actions dim={g["actions"].shape[-1]} != 3'
    if g['img180'].ndim != 4 or g['img180'].shape[-1] != 3: return False, f'img180 shape={g["img180"].shape} not (N,H,W,3)'
    return True, ''

def ensure_masks(f: h5py.File) -> None:
    rng = np.random.default_rng(SEED)
    demos_all = list_demos(f)

    usable = []
    for d in demos_all:
        ok, _ = has_img_and_actions(f, d)
        if ok:
            n = int(f['data'][d].attrs.get('num_samples', 0))
            if n > 0: usable.append(d)

    print(f"[SCAN] demos total={len(demos_all)} | usable={len(usable)}")

    need_build = ('mask' not in f)
    if not need_build:
        m = f['mask']
        splits_present = all(k in m for k in ('train','valid','test'))
        sizes = {k: (len(m[k]) if k in m else 0) for k in ('train','valid','test')}
        if (not splits_present) or (sizes['train']==0 and sizes['valid']==0 and sizes['test']==0):
            need_build = True

    if not need_build:
        print("[MASK] existing masks found, will use them as-is.")
        return

    if len(usable) == 0:
        raise RuntimeError("No usable demo.")

    usable_sorted = sorted(usable)
    rng.shuffle(usable_sorted)
    N = len(usable_sorted)
    n_train = (N * 8) // 10
    n_val   = (N * 1) // 10
    n_test  = N - n_train - n_val
    train = usable_sorted[:n_train]
    valid = usable_sorted[n_train:n_train+n_val]
    test  = usable_sorted[n_train+n_val:]

    print(f"[MASK] rebuild splits: total={N} -> train={len(train)}, valid={len(valid)}, test={len(test)}")

    dt = h5py.string_dtype(encoding='utf-8')
    m = f.require_group('mask')
    for key in ('train','valid','test'):
        if key in m: del m[key]
    m.create_dataset('train', data=np.array(train, dtype=dt))
    m.create_dataset('valid', data=np.array(valid, dtype=dt))
    m.create_dataset('test',  data=np.array(test,  dtype=dt))

def read_mask(f: h5py.File, split: str) -> List[str]:
    arr = f['mask'][split][...]
    return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in arr]

# ================== Dataset: Image + Text + Action ==================
class ImgPerDemoWithTextDataset(Dataset):
    def __init__(self, h5_path: str, demos: List[str], train: bool,
                 demo2text: Dict[str, str] | None = None,
                 default_text: str = TEXT_DEFAULT):
        self.h5_path = h5_path
        self.demos = demos
        self.train = train
        self.demo2text = demo2text or {}
        self.default_text = default_text

        aug = []
        if USE_COLORJIT and train:
            aug.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02))
        self.tf = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            *aug,
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def _text_for_demo(self, f, demo: str) -> str:
        if demo in self.demo2text:
            return self.demo2text[demo]
        g = f['data'][demo]
        if 'text' in g.attrs:  # Prior attrs['text']
            v = g.attrs['text']
            if isinstance(v, bytes): v = v.decode('utf-8')
            return str(v)
        return self.default_text

    def __len__(self): return len(self.demos)

    def __getitem__(self, idx):
        demo = self.demos[idx]
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            g = f['data'][demo]
            img = g['img180'][0]                     # (H,W,3) uint8
            y   = g['actions'][0].astype(np.float32) # (3,)
            t   = self._text_for_demo(f, demo)

        img = Image.fromarray(img)
        if self.train and USE_HFLIP and (random.random() < 0.5):
            img = ImageOps.mirror(img)
            y[0] = -y[0]
            y[2] = -y[2]

        x = self.tf(img)
        y = torch.from_numpy(y)
        return x, t, y, demo

# ================== Vision (TinyCNN → Vector) ==================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.ReLU(inplace=True)
    def forward(self, x): return self.a(self.b(self.c(x)))

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=s, padding=1, groups=in_ch, bias=False)
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
    def __init__(self, feat_dim=128, width=32):
        super().__init__()
        c1, c2, c3, c4 = width, width*2, width*2, width*4
        self.stem = ConvBNAct(3, c1, k=3, s=2, p=1)
        self.b2   = DepthwiseSeparable(c1, c2, s=2)
        self.b3   = DepthwiseSeparable(c2, c3, s=1)
        self.b4   = DepthwiseSeparable(c3, c4, s=2)
        self.b5   = DepthwiseSeparable(c4, c4, s=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(c4, feat_dim), nn.ReLU(True), nn.Dropout(0.1))
    def forward(self, x):
        x = self.stem(x); x = self.b2(x); x = self.b3(x); x = self.b4(x); x = self.b5(x)
        x = self.pool(x)
        feat = self.proj(x)
        return feat

# ================== Text encoder (Frozen CLIP) ==================
from transformers import CLIPTextModel, CLIPTokenizer

class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, name=CLIP_NAME, device='cuda'):
        super().__init__()
        self.tokenizer    = CLIPTokenizer.from_pretrained(name)
        self.text_encoder = CLIPTextModel.from_pretrained(
            name, use_safetensors=True, low_cpu_mem_usage=True, dtype=torch.float32
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, batch_text):
        if isinstance(batch_text, str): batch_text = [batch_text]
        inputs = self.tokenizer(batch_text, return_tensors='pt',
                                padding=True, truncation=True, max_length=77).to(self.device)
        out = self.text_encoder(**inputs).last_hidden_state
        eos_pos = inputs.attention_mask.sum(dim=1) - 1
        sent = out[torch.arange(out.size(0), device=self.device), eos_pos]  # [B,512]
        return sent

# ================== Multi-modal, decoupled heads ==================
class MultiModalDecoupled(nn.Module):
    def __init__(self, v_backbone: TinyCNN_Feature, t_encoder: FrozenCLIPTextEncoder,
                 v_dim=128, t_in_dim=512, t_dim=128):
        super().__init__()
        self.v_backbone = v_backbone
        self.t_encoder  = t_encoder
        self.text_proj  = nn.Sequential(nn.Linear(t_in_dim, t_dim), nn.ReLU(True), nn.Dropout(0.1))
        self.xy_head    = nn.Sequential(nn.Linear(v_dim, 128), nn.ReLU(True), nn.Linear(128, 2))
        self.yaw_head   = nn.Sequential(nn.Linear(t_dim, 128), nn.ReLU(True), nn.Linear(128, 2))  # (cos,sin)

    def forward(self, images, texts):
        v_feat = self.v_backbone(images)
        with torch.no_grad():
            t_raw = self.t_encoder(texts)
        t_feat = self.text_proj(t_raw)
        pred_xy = self.xy_head(v_feat)
        pred_cs = self.yaw_head(t_feat)
        return pred_xy, pred_cs

# ================== yaw head ==================
def yaw_to_cossin(yaw_norm: torch.Tensor):
    theta = yaw_norm * torch.pi
    return torch.cos(theta), torch.sin(theta)

def yaw_loss(pred_cs: torch.Tensor, gt_yaw_norm: torch.Tensor, lambda_unit=LAMBDA_UNIT):
    gt_c, gt_s = yaw_to_cossin(gt_yaw_norm)
    gt = torch.stack([gt_c, gt_s], dim=1)
    l2   = ((pred_cs - gt) ** 2).mean()
    unit = ((pred_cs.norm(dim=1) - 1.0) ** 2).mean()
    return l2 + lambda_unit * unit

@torch.no_grad()
def yaw_ang_mae_deg(pred_cs: torch.Tensor, gt_yaw_norm: torch.Tensor):
    pred_cs = pred_cs / (pred_cs.norm(dim=1, keepdim=True) + 1e-8)
    pred_theta = torch.atan2(pred_cs[:,1], pred_cs[:,0])
    gt_theta   = gt_yaw_norm * torch.pi
    d = torch.remainder(pred_theta - gt_theta + np.pi, 2*np.pi) - np.pi
    return (d.abs().mean() * 180.0 / np.pi).item()

@torch.no_grad()
def evaluate(model: MultiModalDecoupled, loader: DataLoader, device: str):
    model.eval()
    n = 0
    xy_mse_sum = 0.0
    yaw_mae_deg_sum = 0.0
    for x, t, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_xy = y[:, :2]
        y_yawn = y[:, 2]
        pred_xy, pred_cs = model(x, t)
        xy_mse_sum      += ((pred_xy - y_xy)**2).mean(dim=1).sum().item()
        yaw_mae_deg_sum += yaw_ang_mae_deg(pred_cs, y_yawn) * y.size(0)
        n += y.size(0)
    return xy_mse_sum / max(1, n), yaw_mae_deg_sum / max(1, n)

def print_text_distribution(h5_path, demos):
    dist = {}
    with h5py.File(h5_path, 'r') as f:
        for d in demos:
            g = f['data'][d]
            t = g.attrs.get('text', None)
            if t is None: t = TEXT_DEFAULT
            if isinstance(t, bytes): t = t.decode('utf-8')
            dist[t] = dist.get(t, 0) + 1
    print("[INFO] train text distribution (top 10):")
    for k, v in sorted(dist.items(), key=lambda x: -x[1])[:10]:
        print(f"  {repr(k)}: {v}")

def main():
    set_seed(SEED)

    if not os.path.isfile(H5_PATH):
        raise FileNotFoundError(f"HDF5 doesn't exist: {H5_PATH}")

    with h5py.File(H5_PATH, 'r+') as f:
        if 'data' not in f:
            raise RuntimeError("HDF5 lacks /data group.")
        print(f"[INFO] scanning H5 for img180 & actions ...")
        ensure_masks(f)

    with h5py.File(H5_PATH, 'r') as f:
        train_demos = [d for d in read_mask(f, 'train') if has_img_and_actions(f, d)[0]]
        valid_demos = [d for d in read_mask(f, 'valid') if has_img_and_actions(f, d)[0]]
        test_demos  = [d for d in read_mask(f, 'test')  if has_img_and_actions(f, d)[0]]

    print(f"[INFO] demos -> train={len(train_demos)}, valid={len(valid_demos)}, test={len(test_demos)}")
    if len(train_demos) == 0:
        raise RuntimeError("train is empty")

    print_text_distribution(H5_PATH, train_demos)

    ds_train = ImgPerDemoWithTextDataset(H5_PATH, train_demos, train=True)
    ds_valid = ImgPerDemoWithTextDataset(H5_PATH, valid_demos, train=False)
    ds_test  = ImgPerDemoWithTextDataset(H5_PATH, test_demos,  train=False)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SZ, shuffle=True,  num_workers=2, pin_memory=True, drop_last=False)
    dl_valid = DataLoader(ds_valid, batch_size=BATCH_SZ, shuffle=False, num_workers=2, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SZ, shuffle=False, num_workers=2, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    v_backbone = TinyCNN_Feature(feat_dim=128, width=32).to(device)
    t_encoder  = FrozenCLIPTextEncoder(name=CLIP_NAME, device=device)
    model = MultiModalDecoupled(v_backbone=v_backbone, t_encoder=t_encoder,
                                v_dim=128, t_in_dim=512, t_dim=128).to(device)

    # Different LR for different parts if desired
    opt = torch.optim.AdamW([
        {"params": v_backbone.parameters(),          "lr": LR},
        {"params": model.text_proj.parameters(),     "lr": LR*2.0},
        {"params": model.xy_head.parameters(),       "lr": LR},
        {"params": model.yaw_head.parameters(),      "lr": LR*2.0},
    ], weight_decay=WEIGHT_DECAY)

    best_val = float('inf')
    no_improve, PATIENCE = 15, 15
    save_dir = os.path.join(RUNS_DIR, EXP_NAME, 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_best.pth')

    print("\n[TRAIN] start")
    for epoch in range(1, EPOCHS+1):
        model.train()
        n = 0
        xy_mse_sum = 0.0
        yaw_l_sum  = 0.0

        for x, t, y, _ in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_xy   = y[:, :2]
            y_yawn = y[:, 2]

            opt.zero_grad(set_to_none=True)
            pred_xy, pred_cs = model(x, t)

            loss_xy  = ((pred_xy - y_xy) ** 2).mean()
            loss_yaw = yaw_loss(pred_cs, y_yawn, lambda_unit=LAMBDA_UNIT)
            loss = loss_xy + LAMBDA_YAW * loss_yaw

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            xy_mse_sum += loss_xy.item()  * y.size(0)
            yaw_l_sum  += loss_yaw.item() * y.size(0)
            n += y.size(0)

        train_xy_mse = xy_mse_sum / max(1, n)
        train_yaw_l  = yaw_l_sum  / max(1, n)
        val_xy_mse, val_yaw_mae_deg = evaluate(model, dl_valid, device)

        print(f"[Epoch {epoch:03d}] "
              f"train XY MSE={train_xy_mse:.6f} | train yaw L={train_yaw_l:.6f} || "
              f"valid XY MSE={val_xy_mse:.6f}, yaw MAE={val_yaw_mae_deg:.2f}°")

        # Simple A/B probing on text (using a dummy image)
        with torch.no_grad():
            _probe = torch.zeros((1,3,IMG_SIZE,IMG_SIZE), dtype=torch.float32, device=device)
            from transformers import logging as hf_logging
            hf_logging.set_verbosity_error()
            # Two instruction variants; can be aligned with write_text_attrs.py if needed.
            for txt in [
                "Place the timber along the diagonal of the bottom left corner and the upper right corner.",
                "Place the timber along the diagonal of the upper left corner and the bottom right corner.",
            ]:
                pred_xy, pred_cs = model(_probe, [txt])
                pred_cs = pred_cs / (pred_cs.norm(dim=1, keepdim=True) + 1e-8)
                yaw = torch.atan2(pred_cs[:,1], pred_cs[:,0])  # [-π,π]
                print(f"[A/B] text={txt!r} -> yaw_world={float(yaw.item()):.6f} rad")

        # Early stop
        score = val_xy_mse + (val_yaw_mae_deg * np.pi / 180.0)
        if score < best_val - 1e-6:
            best_val = score
            no_improve = 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_score': best_val,
                'clip_name': CLIP_NAME,
                'img_size': IMG_SIZE,
                'normalize': {'mean':[0.5,0.5,0.5], 'std':[0.5,0.5,0.5]},
                'decoupled': True,
                'use_hflip': USE_HFLIP,
                'use_colorjit': USE_COLORJIT,
                'lambda_yaw': LAMBDA_YAW,
                'lambda_unit': LAMBDA_UNIT,
            }, save_path)
            print(f"  ↳ Saved best to {save_path}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("  ↳ Early stopping.")
                break

    print("\n[TEST] evaluating best checkpoint...")
    if os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    test_xy_mse, test_yaw_mae_deg = evaluate(model, dl_test, device)
    print(f"[TEST] XY MSE={test_xy_mse:.6f}  yaw MAE={test_yaw_mae_deg:.2f}°")

    out_dir = os.path.join(RUNS_DIR, EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"TEST XY MSE={test_xy_mse:.6f}, yaw MAE={test_yaw_mae_deg:.2f} deg\n")
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({
            "test_xy_mse": float(test_xy_mse),
            "test_yaw_mae_deg": float(test_yaw_mae_deg),
            "epochs_trained": epoch,
            "best_val_score": float(best_val),
            "clip_name": CLIP_NAME,
            "img_size": IMG_SIZE,
            "use_hflip": USE_HFLIP,
            "use_colorjit": USE_COLORJIT
        }, f, indent=2)
    print(f"[OK] results saved to {out_dir}")

if __name__ == "__main__":
    main()

