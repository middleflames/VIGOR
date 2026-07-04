"""Convenience launcher for VIGOR training / evaluation.
Examples:
    # evaluate a trained checkpoint on GPU 0
    python run.py --checkpoint checkpoint_best.pth --output_dir output/vigor --evaluate

    # train on GPUs 0,1,2, fine-tuning from the X-VLM bbox pre-trained checkpoint
    python run.py --checkpoint xvlm_pretrained.pth --output_dir output/vigor \
        --gpus 0,1,2 --load_bbox_pretrain
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="VIGOR train/eval launcher")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="checkpoint to load: a trained VIGOR checkpoint for --evaluate, or "
        "the X-VLM bbox pre-trained checkpoint for training (with "
        "--load_bbox_pretrain)",
    )
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--config", default="configs/vigor_talk2car.yaml", type=str)
    parser.add_argument(
        "--gpus",
        default="0",
        type=str,
        help="comma-separated GPU ids on this node, e.g. '0' or '0,1,2'",
    )
    parser.add_argument(
        "--bs",
        default=-1,
        type=int,
        help="total batch size across GPUs (per-gpu = bs // n_gpus); "
        "-1 uses the config value",
    )
    parser.add_argument("--master_port", default=12345, type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--load_bbox_pretrain", action="store_true")
    args = parser.parse_args()

    n_gpus = len([g for g in args.gpus.split(",") if g.strip() != ""])
    if n_gpus == 0:
        parser.error("--gpus must list at least one GPU id")

    cmd = (
        f"CUDA_VISIBLE_DEVICES={args.gpus} "
        f"{sys.executable} -m torch.distributed.launch "
        f"--nproc_per_node={n_gpus} --master_port={args.master_port} --use_env "
        f"train.py --config {args.config} --output_dir {args.output_dir} "
        f"--checkpoint {args.checkpoint} --bs {args.bs}"
        f"{' --load_bbox_pretrain' if args.load_bbox_pretrain else ''}"
        f"{' --evaluate' if args.evaluate else ''}"
    )
    print(cmd, flush=True)
    raise SystemExit(0 if os.system(cmd) == 0 else 1)


if __name__ == "__main__":
    main()
