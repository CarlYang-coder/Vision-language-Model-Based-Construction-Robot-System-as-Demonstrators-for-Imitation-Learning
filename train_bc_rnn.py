#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, glob
import numpy as np
import h5py
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train

# Target threshold (for early stopping / LR scheduler)
THRESH = 0.0067

# ===== Dataset path (fixed) =====
DS_PATH = os.path.expanduser('~/Downloads/Data_Collecting/bc_dataset_from616.hdf5')

config = config_factory(algo_name="bc")  # BC + optional RNN

with config.unlocked():
    # ========= 1) Dataset & splits =========
    config.train.data = DS_PATH
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.experiment.validate = True

    # Pure supervised learning; no environment rollouts
    config.experiment.rollout.enabled = False
    config.experiment.render = False
    config.experiment.render_video = False

    # ========= 2) Observation modalities =========
    # Assume your HDF5 structure:
    #   /data/<demo>/obs/qpos          -> joint positions
    #   /data/<demo>/obs/ee_pos        -> end-effector pose (your 6D definition, e.g. x,y,z,roll,pitch,yaw)
    #   /data/<demo>/obs/Simulation_xy -> (x,y) used in simulation
    #
    # Actions:
    #   /data/<demo>/actions           -> Δee_pose (6D), either in physical space (with action_scale/bias)
    #                                      or already normalized
    config.observation.modalities.obs.low_dim = ["qpos", "ee_pos", "Simulation_xy"]

    # ========= 3) RNN settings (BC-RNN) =========
    # Enable RNN (if supported by the robomimic version)
    try:
        config.algo.rnn.enabled = True
    except Exception:
        pass

    # Set sequence length to 60 (to match your later evaluation / simulation)
    for setter in [
        ("train", "seq_length"),
        ("algo", "rnn", "sequence_length"),
        ("algo", "rnn", "horizon"),
    ]:
        try:
            node = config
            for k in setter[:-1]:
                node = getattr(node, k)
            setattr(node, setter[-1], 60)
        except Exception:
            pass

    # ========= 4) Training hyperparameters =========
    config.train.batch_size = 64
    config.train.num_epochs = 100          # Can be large; early stopping will stop earlier if triggered
    config.train.num_data_workers = 0
    config.train.seed = 0

    # Learning rate
    try:
        config.train.lr = 1e-3
    except Exception:
        pass

    # ========= 5) Optimizer & LR scheduling =========
    # New robomimic layout: config.algo.optim_params.policy
    try:
        opt = config.algo.optim_params.policy
        opt.name = "AdamW"                # or "Adam"
        opt.lr = config.train.lr
        opt.weight_decay = 5e-4
        opt.betas = [0.9, 0.999]
        opt.eps = 1e-8
        opt.grad_clip = 1.0               # global gradient clipping

        opt.lr_scheduler = "ReduceLROnPlateau"
        opt.lr_scheduler_params = {
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "threshold": THRESH / 2,
            "min_lr": 1e-6,
        }
    except Exception:
        pass

    # Legacy layout: config.train.optim_params.policy (for backward compatibility)
    try:
        opt = config.train.optim_params.policy
        opt.name = "AdamW"
        opt.lr = config.train.lr
        opt.weight_decay = 5e-4
        opt.betas = [0.9, 0.999]
        opt.eps = 1e-8
        opt.grad_clip = 1.0
        opt.lr_scheduler = "ReduceLROnPlateau"
        opt.lr_scheduler_params = {
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "threshold": THRESH / 2,
            "min_lr": 1e-6,
        }
    except Exception:
        pass

    # ========= 6) Early stopping (if supported) =========
    try:
        config.experiment.early_stop.enabled = True
        config.experiment.early_stop.metric = "valid/loss"
        config.experiment.early_stop.mode = "min"
        config.experiment.early_stop.threshold = THRESH
        config.experiment.early_stop.patience = 5
    except Exception:
        pass

    # ========= 7) Logging & save paths =========
    # All outputs will be under:
    #   ~/Downloads/robomimic_runs/bc_rnn_from616_qpos_ee_simxy_to_delta_ee/<timestamp>/
    config.experiment.name = "bc_rnn_from616_qpos_ee_simxy_to_delta_ee"
    config.experiment.log_dir = os.path.abspath(os.path.expanduser("~/Downloads/robomimic_runs"))

    # Ensure models are saved per epoch under logs/.../models/
    try:
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = 1
    except Exception:
        pass

# ========= 8) Start training =========
device = "cuda" if torch.cuda.is_available() else "cpu"
train(config, device=device)
print("Training finished.")

