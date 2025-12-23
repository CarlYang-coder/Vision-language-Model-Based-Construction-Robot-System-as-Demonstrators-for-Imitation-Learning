#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 Isaac Sim 环境中离线修改 USD：
- 先启动 SimulationApp（headless），确保 pxr 可用
- 打开输入 USD，修改指定 Prim 的位置/旋转
- 保存为新的 USD 文件（不覆盖原文件）
用法示例：
  /home/carl/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh usd_edit_test.py \
    --in /home/carl/Downloads/test_6.usd \
    --prim /World/factory/Wood_block \
    --x 0.1 --y 0.2 --z 0.05 \
    --rx 0 --ry 0 --rz 0 \
    --out /home/carl/Downloads/test_6_modified.usd
"""

import argparse, os, sys

# 1) 先启动 Kit/Isaac 环境，使 pxr/omni.usd 可用
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# 2) 现在再导入 pxr / omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf
import omni.usd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="输入 USD 文件路径")
    ap.add_argument("--out", dest="out_path", required=True, help="输出 USD 文件路径（新文件）")
    ap.add_argument("--prim", dest="prim_path", required=True, help="需要修改的 Prim 路径（如 /World/factory/Timber_truss_assembly_3）")
    ap.add_argument("--x", type=float, required=True, help="新位置 X（世界坐标）")
    ap.add_argument("--y", type=float, required=True, help="新位置 Y（世界坐标）")
    ap.add_argument("--z", type=float, required=True, help="新位置 Z（世界坐标）")
    # 旋转按 XYZ 欧拉角（度）
    ap.add_argument("--rx", type=float, default=0.0, help="X 轴旋转（度）")
    ap.add_argument("--ry", type=float, default=0.0, help="Y 轴旋转（度）")
    ap.add_argument("--rz", type=float, default=0.0, help="Z 轴旋转（度）")
    return ap.parse_args()

def main():
    args = parse_args()
    in_path  = os.path.abspath(args.in_path)
    out_path = os.path.abspath(args.out_path)
    prim_path = args.prim_path

    if not os.path.exists(in_path):
        print(f"[ERROR] 输入文件不存在: {in_path}")
        sys.exit(1)

    # 用 omni.usd 上下文打开 Stage
    ctx = omni.usd.get_context()
    ctx.open_stage(in_path)
    stage = ctx.get_stage()
    if stage is None:
        print(f"[ERROR] 无法打开 Stage: {in_path}")
        sys.exit(1)

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        print(f"[ERROR] 找不到 Prim: {prim_path}")
        sys.exit(1)

    xform = UsdGeom.Xformable(prim)

    # 为避免多次运行叠加 XformOps，先清空再设置
    try:
        xform.ClearXformOpOrder()
    except Exception:
        pass

    # 位置（世界空间期望写到本地 xform 上，注意层次是否有父级变换。一般这样改够用）
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(args.x, args.y, args.z))

    # 旋转（XYZ 欧拉角，单位度）
    rotate_op = xform.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3f(args.rx, args.ry, args.rz))

    # 确保 XformOp 顺序：先平移再旋转（按需要可调整）
    xform.SetXformOpOrder([translate_op, rotate_op])

    # 导出为新文件（不会覆盖原始文件）
    root_layer: Sdf.Layer = stage.GetRootLayer()
    root_layer.Export(out_path)

    print(f"[OK] 已修改 {prim_path} 的坐标/旋转，并保存为新文件：\n  {out_path}")

    simulation_app.close()

if __name__ == "__main__":
    main()
