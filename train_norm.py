#!/usr/bin/env python3
"""
Side experiment (not a main-table row): train/evaluate with the backbone's own input normalisation.

The main script feeds every backbone pixels scaled to [0,1] with no mean/std. This wrapper runs the
SAME script (same data pipeline, augmentation, losses, learning rates, schedule) but subtracts the
image_mean and divides by the image_std from the backbone's Hugging Face preprocessor_config.json
immediately before the encoder. Nothing else changes: the edge gate still sees the [0,1] image.

  SEED=42 RUN_TAG=fullnorm_s42 EVAL_SPLITS=val,test python train_norm.py
  SEED=42 RUN_TAG=webnorm_s42 DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m EVAL_SPLITS=val,test python train_norm.py

dinov3_forensics_train_v2.py is not modified; the encoder call is patched at run time.
"""
import os, json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
import dinov3_forensics_train_v2 as M


def backbone_norm(repo: str):
    cfg = json.load(open(hf_hub_download(repo, "preprocessor_config.json")))
    return [float(v) for v in cfg["image_mean"]], [float(v) for v in cfg["image_std"]]


def patch(mean, std):
    def encode(self, x: torch.Tensor):
        m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return self.encoder(pixel_values=(x - m) / s, output_hidden_states=True)
    M.DINOv3ViTSegFormer3C.encode = encode


if __name__ == "__main__":
    repo = os.environ.get("DINOV3_CKPT", M.DEFAULT_DINOV3_REPO)
    if os.environ.get("BACKBONE", "dinov3") != "dinov3":
        raise SystemExit("train_norm.py only supports the DINOv3 ViT backbones")
    tag = os.environ.get("RUN_TAG", "")
    if "norm" not in tag:
        raise SystemExit("set RUN_TAG to a name containing 'norm' (e.g. fullnorm_s42) so the run cannot be mistaken for a main row")
    mean, std = backbone_norm(repo)
    patch(mean, std)
    print(f"[norm] encoder input normalised with mean={mean} std={std} ({repo})", flush=True)
    M.main()
    out = Path.home() / "results_dinov3" / "vit" / Path(repo).stem.replace("/", "_") / tag
    if out.is_dir():
        (out / "input_norm.json").write_text(json.dumps({"repo": repo, "image_mean": mean, "image_std": std,
                                                        "applied": "before encoder only; edge gate sees [0,1] image"}, indent=2))
