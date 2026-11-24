#!/usr/bin/env python3
"""
DINOv3 backbones (ConvNeXt + ViT) with lightweight SegFormer-style decoder
3-class classification (pristine / tampered / fake)
+ tamper-mask segmentation IoU
+ early-improved fusion
+ boundary loss
+ small-object IoU
+ fast TTA
+ safe real-prior
+ confusion matrices, ROC, publication plots

THIS IS THE RESTORED “O2-A” VERSION (DINOv3 only).
NO SegFormer-B5.
NO heavy rewrite.

------------------------------------------------------------------
v2 CHANGES (for the GRSL submission) -- search for "[v2]" in this file
------------------------------------------------------------------
 1. Patch-token slicing: CLS *and* register tokens are now dropped, so the
    24x24 grid holds only patch tokens in the right positions.
    LEGACY_TOKEN_SLICE=1 restores the old behaviour (needed only to
    re-evaluate a checkpoint trained with the old slicing).
 2. IoU: the x1.20 bonus for masks under 200 px is removed. IoU is plain IoU.
 3. Ablation switches (env, default on): ABL_EDGE_GATE, ABL_BOUNDARY,
    ABL_DEEP_SUP. FREEZE_BACKBONE=1 gives a frozen-encoder baseline.
 4. Two learning rates: LR_BACKBONE (default 2e-6) and LR_DECODER
    (default 1e-5), as the letter states. LR_DECODER=2e-6 restores the old
    single-rate behaviour.
 5. SEED (default 42) seeds python / numpy / torch, the sampler and the
    dataloader workers.
 6. Mask-only augmentation (erode / open / random circles on the ground-truth
    mask without touching the image) is OFF by default. MASK_AUG=1 restores it.
 7. Parameter counts, seconds per training epoch and ms per test image are
    measured and saved.
 8. Test metrics are written to test_metrics.json (per-class P/R/F1/AP/AUC,
    balanced accuracy, confusion matrix, IoU / Dice / MCC on tampered images,
    pixel confusion over the full test set, run configuration).
 9. Pixel confusion now binarises the ground truth (> 0.5) instead of testing
    == 0 / == 1.
10. Each run writes to its own folder: .../<checkpoint>/<RUN_TAG>, where
    RUN_TAG defaults to "<variant>_s<seed>" (e.g. full_s42, noedge_s42).
"""

import os, math, json, csv, random, time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torch._dynamo as dynamo
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel
from packaging import version
import transformers
import timm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    torch.set_float32_matmul_precision("high")
except:
    pass

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Force the safest attention kernels to avoid NaNs on some GPUs/driver combos.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

# ============================================================
#                 BASIC UTILITIES
# ============================================================

def safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x).clamp(1e-6, 1 - 1e-6)


def require_min_transformers(min_version: str = "4.56.0"):
    """
    Ensure Transformers is recent enough for DINOv3 collection.
    """
    if version.parse(transformers.__version__) < version.parse(min_version):
        raise RuntimeError(
            f"transformers>={min_version} required for DINOv3; found {transformers.__version__}"
        )


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "y", "on")


def seed_everything(seed: int):
    """[v2] Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int):
    """[v2] Give every dataloader worker its own reproducible numpy/python seed."""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def count_params(model: nn.Module) -> Dict[str, int]:
    """[v2] Parameter counts: total, encoder (backbone), heads (decoder + classifier)."""
    total = sum(p.numel() for p in model.parameters())
    enc = sum(p.numel() for p in model.encoder.parameters()) if hasattr(model, "encoder") else 0
    return {"total": int(total), "encoder": int(enc), "heads": int(total - enc)}


def remap_state_dict_keys(sd: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """[v2-fix] Translate checkpoint keys to the current model's key names.

    transformers 5.x moved the DINOv3 ViT blocks from `encoder.layer.N` /
    `encoder.norm` to `encoder.model.layer.N` / `encoder.model.norm`. A checkpoint
    saved under 4.x therefore loads with hundreds of missing/unexpected keys when
    strict=False, i.e. the backbone silently keeps its pretrained weights. This
    maps old->new (and new->old) by inserting/removing the `model.` segment.
    """
    target = set(model.state_dict().keys())
    out = {}
    for k, v in sd.items():
        k2 = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        if k2 not in target:
            if k2.startswith("encoder.") and ("encoder.model." + k2[len("encoder."):]) in target:
                k2 = "encoder.model." + k2[len("encoder."):]
            elif k2.startswith("encoder.model.") and ("encoder." + k2[len("encoder.model."):]) in target:
                k2 = "encoder." + k2[len("encoder.model."):]
        out[k2] = v
    return out


# [fix-amp] Mixed-precision dtype for the model forward/backward pass.
# fp16 (the original setting) overflows in the backward pass of the first ViT-L blocks once the
# backbone is unfrozen: every overflow makes GradScaler skip the step and halve its scale, the scale
# never recovers (512 -> 32 within one epoch, 7e-40 by epoch 19 of full_s42) and the weights end up
# NaN. bfloat16 has the exponent range of float32, so nothing overflows and no GradScaler is needed.
# AMP_DTYPE=fp16 restores the old behaviour; AMP_DTYPE=fp32 turns mixed precision off.
_AMP_NAME = os.environ.get("AMP_DTYPE", "bf16").lower()
if _AMP_NAME not in ("bf16", "fp16", "fp32"):
    raise ValueError(f"AMP_DTYPE must be bf16, fp16 or fp32, got {_AMP_NAME}")
AMP_DTYPE = torch.bfloat16 if _AMP_NAME == "bf16" else torch.float16
AMP_OFF = (_AMP_NAME == "fp32")


def make_grad_scaler(device: torch.device, use_amp: bool):
    """
    Create a GradScaler with best-available API across torch versions.
    """
    if not use_amp:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler(device_type=device.type, enabled=False)
            except TypeError:
                try:
                    return torch.amp.GradScaler(device.type, enabled=False)
                except TypeError:
                    pass
        return torch.cuda.amp.GradScaler(enabled=False)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type=device.type)
        except TypeError:
            try:
                return torch.amp.GradScaler(device.type)
            except TypeError:
                pass
            pass
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def convert_dinov3_convnext_to_timm(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map DINOv3 ConvNeXt checkpoint keys to timm ConvNeXt naming.
    Drops keys that are not used in the features-only timm model.
    """
    new_state = {}

    def map_downsample(idx: int, sub: str) -> str:
        if idx == 0:
            return f"stem_{sub}"
        return f"stages_{idx}.downsample.{sub}"

    for k, v in state.items():
        if k.startswith("downsample_layers."):
            parts = k.split(".")
            idx = int(parts[1])
            sub = parts[2]
            tgt = map_downsample(idx, sub)
            new_state[tgt] = v
            continue

        if k.startswith("stages."):
            parts = k.split(".")
            stage = int(parts[1])
            block = int(parts[2])
            suffix = ".".join(parts[3:])
            prefix = f"stages_{stage}.blocks.{block}."

            if suffix.startswith("dwconv"):
                new_state[prefix + "conv_dw." + suffix.split(".", 1)[1]] = v
            elif suffix.startswith("pwconv1"):
                new_state[prefix + "mlp.fc1." + suffix.split(".", 1)[1]] = v
            elif suffix.startswith("pwconv2"):
                new_state[prefix + "mlp.fc2." + suffix.split(".", 1)[1]] = v
            elif suffix.startswith("norm"):
                new_state[prefix + "norm." + suffix.split(".", 1)[1]] = v
            elif suffix == "gamma":
                new_state[prefix + "gamma"] = v
            else:
                continue
            continue

        # ignore classifier / extra norms not present in features_only timm model
    return new_state


DEFAULT_CONVNEXT_LOCAL = Path("models/dinov3_manual/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth")
DEFAULT_CONVNEXT_LARGE_LOCAL = Path("models/dinov3_manual/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth")
if DEFAULT_CONVNEXT_LARGE_LOCAL.exists():
    DEFAULT_DINOV3_CONVNEXT = str(DEFAULT_CONVNEXT_LARGE_LOCAL)
elif DEFAULT_CONVNEXT_LOCAL.exists():
    DEFAULT_DINOV3_CONVNEXT = str(DEFAULT_CONVNEXT_LOCAL)
else:
    DEFAULT_DINOV3_CONVNEXT = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
DEFAULT_DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-sat493m"

def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-6:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def set_encoder_trainable(model: nn.Module, trainable: bool):
    """
    Toggle encoder gradients (for warmup freezing). Supports DINOv3 models defined in this script.
    """
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = trainable
    if hasattr(model, "backbone"):
        bb = model.backbone
        for attr in ("dino", "convnext", "encoder"):
            if hasattr(bb, attr):
                for p in getattr(bb, attr).parameters():
                    p.requires_grad = trainable


# ============================================================
#                  DINOv3 MODEL CARD (REFERENCE)
# ============================================================

DINOV3_MODEL_CARD = """
Model Card for DINOv3
DINOv3 is a family of versatile vision foundation models that outperforms the specialized state of the art across a broad range of settings, without fine-tuning. DINOv3 produces high-quality dense features that achieve outstanding performance on various vision tasks, significantly surpassing previous self- and weakly-supervised foundation models.

Model Details
These are Vision Transformer and ConvNeXt models trained following the method described in the DINOv3 paper. 12 models are provided:

10 models pretrained on web data (LVD-1689M dataset)
1 ViT-7B trained from scratch,
5 ViT-S/S+/B/L/H+ models distilled from the ViT-7B,
4 ConvNeXt-{T/S/B/L} models distilled from the ViT-7B,
2 models pretrained on satellite data (SAT-493M dataset)
1 ViT-7B trained from scratch
1 ViT-L distilled from the ViT-7B
Each Transformer-based model takes an image as input and returns a class token, patch tokens (and register tokens). These models follow a ViT architecture, with a patch size of 16. For a 224x224 image, this results in 1 class token + 4 register tokens + 196 patch tokens = 201 tokens (for DINOv2 with registers this resulted in 1 + 4 + 256 = 261 tokens).

The models can accept larger images provided the image shapes are multiples of the patch size (16). If this condition is not verified, the model will crop to the closest smaller multiple of the patch size.

Model Description
Developed by: Meta AI
Model type: Vision Transformer, ConvNeXt
License: DINOv3 License
Model Sources
Repository: https://github.com/facebookresearch/dinov3
Paper: https://arxiv.org/abs/2508.10104
Uses
The models are vision backbones providing multi-purpose features for downstream tasks.

Direct Use
The models can be used without fine-tuning, with downstream classifiers as simple as linear layers, to obtain competitive results:

on image classification, using k-NN classifiers on the class token
on image classification, with logistic regression classifiers applied on the class token
on image classification, with a linear layer applied on the class token and the average of the patch tokens
on image retrieval using nearest neighbors
on geometric and semantic 3D keypoint correspondances
on depth estimation, semantic segmentation, using linear layers
on unsupervised object discovery
on video segmentation tracking
on video classification, using a small 4-layer attentive probe
Downstream Use
While fine-tuning the models can yield some gains, it is recommended to keep this option as a last resort: the frozen features are expected to provide good performance out-of-the-box.

Bias, Risks, and Limitations
Compared to DINOv2 and SEERv2, DINOv3 delivers somewhat consistent performance across income categories on geographical fairness and diversity, although with a notable performance drop in the low-income bucket compared to the highest-income bucket.

DINOv3 also achieves relatively good scores across different regions, improving over its predecessor DINOv2. However, a relative difference is still observed between Europe and Africa.

Recommendations
Fine-tuning is expected to increase the biases in the features produced by the model as they will be tuned to the fine-tuning labels.

How to Get Started with the Model
The example below demonstrates how to obtain an image embedding with [Pipeline] or the [AutoModel] class.

from transformers import pipeline
from transformers.image_utils import load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-vitl16-pretrain-sat493m",
    task="image-feature-extraction", 
)
features = feature_extractor(image)

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)

Training Details
Training Data
Web dataset (LVD-1689M): a curated dataset of 1,689 millions of images extracted from a large data pool of 17 billions web images collected from public posts on Instagram

Satellite dataset (SAT-493M): a dataset of 493 millions of 512x512 images sampled randomly from Maxar RGB ortho-rectified imagery at 0.6 meter resolution

Training Procedure
Training objective:

DINO self-distillation loss with multi-crop

iBOT masked-image modeling loss

KoLeo regularization on [CLS] tokens

Gram anchoring

Training regime: PyTorch FSDP2 (with bf16 and fp8 matrix multiplications)

Distillation:

Distillation follows the standard DINOv3 pretraining procedure, except the teacher is a frozen pretrained ViT-7B.
Evaluation
Results

The reader is referred to the associated paper for details on the evaluation protocols

Results for ViT backbones pretrained (or distilled) on web (LVD-1689M)
"""

# ============================================================
#           ORIGINAL LITE SEGFORMER DECODER (O2-A)
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SegFormerLiteDecoder(nn.Module):
    """
    ORIGINAL early-improved lightweight SegFormer-like decoder:
    - 4 hidden layers from ViT-style backbone
    - MLP projection
    - Depthwise conv
    - Fuse + Refine
    """

    def __init__(self, in_dims: List[int], embed_dim: int = 256):
        super().__init__()

        self.proj_layers = nn.ModuleList([MLP(d, embed_dim) for d in in_dims])
        self.conv_layers = nn.ModuleList([DepthwiseSeparableConv(embed_dim) for _ in in_dims])

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_dims), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # refinement block (O2-A version)
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.seg_logits = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, hidden_list, grid_hw, target_hw):
        B = hidden_list[0].shape[0]
        gh, gw = grid_hw
        feats = []

        for proj, conv, h in zip(self.proj_layers, self.conv_layers, hidden_list):
            x = proj(h)          # (B,N,E)
            x = x.transpose(1,2) # (B,E,N)
            x = x.reshape(B, -1, gh, gw)
            x = conv(x)
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            feats.append(x)

        fused = torch.cat(feats, dim=1)
        fused = self.fuse(fused)
        fused = self.refine(fused)
        seg_logits = self.seg_logits(fused)
        return seg_logits


# ============================================================
#              SegFormerV2 Decoder (multi-level)
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class UpSample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch, k=1, p=0)
        self.refine = ConvBNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int]):
        x = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=False)
        x = self.proj(x)
        return self.refine(x)


class SegFormerV2Decoder(nn.Module):
    """
    SegFormer V2-style decoder that fuses four feature maps with top-down refinement.
    """
    def __init__(self, dims: Tuple[int, int, int, int], embed_dim: int = 256):
        super().__init__()
        C2, C3, C4, C5 = dims

        self.l2 = ConvBNAct(C2, embed_dim, k=1, p=0)
        self.l3 = ConvBNAct(C3, embed_dim, k=1, p=0)
        self.l4 = ConvBNAct(C4, embed_dim, k=1, p=0)
        self.l5 = ConvBNAct(C5, embed_dim, k=1, p=0)

        self.up5_4 = UpSample(embed_dim, embed_dim)
        self.up4_3 = UpSample(embed_dim, embed_dim)
        self.up3_2 = UpSample(embed_dim, embed_dim)

        self.refine1 = nn.Sequential(
            ConvBNAct(embed_dim * 4, embed_dim),
            nn.Dropout2d(0.1),
            ConvBNAct(embed_dim, embed_dim),
        )

        self.refine2 = nn.Sequential(
            ConvBNAct(embed_dim, embed_dim),
            nn.Dropout2d(0.1),
            ConvBNAct(embed_dim, embed_dim),
        )

        self.pred = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]):
        f2, f3, f4, f5 = feats
        H, W = f2.shape[2], f2.shape[3]

        f2 = self.l2(f2)
        f3 = self.l3(f3)
        f4 = self.l4(f4)
        f5 = self.l5(f5)

        f5_up = self.up5_4(f5, f4.shape[-2:])
        f4 = f4 + f5_up

        f4_up = self.up4_3(f4, f3.shape[-2:])
        f3 = f3 + f4_up

        f3_up = self.up3_2(f3, f2.shape[-2:])
        f2 = f2 + f3_up

        fused = torch.cat([
            f2,
            F.interpolate(f3, size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(f4, size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(f5, size=(H, W), mode="bilinear", align_corners=False),
        ], dim=1)

        x = self.refine1(fused)
        x = self.refine2(x)
        mask = self.pred(x)
        return mask


class DINOv3SegFormerV2_3C(nn.Module):
    """
    DINOv3 ConvNeXt backbone (multi-scale) with SegFormer-V2 decoder and 3-class head.
    Can load from an HF repo id or a local .pth checkpoint (convnext).
    """
    def __init__(self,
                 dinov3_ckpt: str = DEFAULT_DINOV3_CONVNEXT,
                 embed_dim: int = 256,
                 num_classes: int = 3):
        super().__init__()

        self.use_timm = False
        self.pool = nn.AdaptiveAvgPool2d(1)

        ckpt_path = Path(dinov3_ckpt)
        if ckpt_path.is_file():
            self.use_timm = True

            variant = "convnext_tiny"
            name_lower = ckpt_path.name.lower()
            if "small" in name_lower:
                variant = "convnext_small"
            elif "base" in name_lower:
                variant = "convnext_base"
            elif "large" in name_lower:
                variant = "convnext_large"

            self.encoder = timm.create_model(
                variant,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                pretrained=False,
            )

            raw_state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            state = convert_dinov3_convnext_to_timm(raw_state)
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                print(f"[warn] convnext load missing keys: {len(missing)}, unexpected: {len(unexpected)}")

            self.dims = tuple(self.encoder.feature_info.channels())
        else:
            require_min_transformers()

            try:
                self.encoder = AutoModel.from_pretrained(
                    dinov3_ckpt,
                    output_hidden_states=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DINOv3 checkpoint '{dinov3_ckpt}'. "
                    "Ensure the repo id is correct, accessible, and transformers >=4.56.0."
                ) from e

            cfg = self.encoder.config
            if hasattr(cfg, "hidden_sizes"):
                self.dims = tuple(cfg.hidden_sizes)
            elif hasattr(cfg, "hidden_size"):
                hs = cfg.hidden_size
                self.dims = (hs // 4, hs // 2, hs, hs)
            else:
                self.dims = (96, 192, 384, 768)

        # Unfreeze ConvNeXt for satellite specialization
        for p in self.encoder.parameters():
            p.requires_grad = True

        self.cls_head = nn.Linear(self.dims[-1], num_classes)

        self.decoder = SegFormerV2Decoder(
            dims=self.dims,
            embed_dim=embed_dim
        )

    def extract_pyramid(self, hidden: List[torch.Tensor]):
        """
        Use the last four hidden states from the ConvNeXt-style backbone.
        Each is expected to be (B, C, H, W).
        """
        feats = hidden[-4:] if len(hidden) >= 4 else hidden
        if len(feats) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(feats)}")
        return feats[0], feats[1], feats[2], feats[3]

    def forward(self, x):
        B, C, H, W = x.shape

        if self.use_timm:
            feats = self.encoder(x)
            f2, f3, f4, f5 = feats
            pooled = self.pool(f5).flatten(1)
        else:
            out = self.encoder(pixel_values=x, output_hidden_states=True)
            hidden = out.hidden_states
            f2, f3, f4, f5 = self.extract_pyramid(hidden)
            if f5.dim() == 4:
                pooled = self.pool(f5).flatten(1)
            else:
                pooled = f5.mean(dim=1)
        cls_logits = self.cls_head(pooled)

        seg_logits = self.decoder([f2, f3, f4, f5])
        seg_logits = F.interpolate(seg_logits, size=(H, W),
                                   mode="bilinear", align_corners=False)

        seg_mid = F.avg_pool2d(seg_logits, kernel_size=2)
        seg_mid = F.interpolate(seg_mid, size=(H, W),
                                mode="bilinear", align_corners=False)

        return cls_logits, seg_logits, seg_mid


class DINOv3ViTSegFormer3C(nn.Module):
    """
    DINOv3 ViT backbone with ViT-style decoder (edge-gated) and 3-class head.
    """
    def __init__(self,
                 dinov3_ckpt: str = "facebook/dinov3-vits16",
                 embed_dim: int = 256,
                 num_classes: int = 3,
                 use_edge_gate: bool = True,
                 legacy_token_slice: bool = False):
        super().__init__()

        require_min_transformers()

        try:
            self.encoder = AutoModel.from_pretrained(
                dinov3_ckpt,
                output_hidden_states=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 ViT checkpoint '{dinov3_ckpt}'. "
                "Ensure the repo id is correct, accessible, and transformers >=4.56.0."
            ) from e

        cfg = self.encoder.config
        self.hidden_size = getattr(cfg, "hidden_size", None)
        if self.hidden_size is None and hasattr(cfg, "hidden_sizes"):
            self.hidden_size = cfg.hidden_sizes[-1]
        if self.hidden_size is None:
            raise ValueError("Could not infer hidden_size from DINOv3 ViT config.")

        # patch_size may be int or tuple
        patch = getattr(cfg, "patch_size", 16)
        if isinstance(patch, (tuple, list)):
            self.patch_h, self.patch_w = patch[0], patch[1]
        else:
            self.patch_h = self.patch_w = int(patch)

        self.decoder = EdgeGatedViTDecoder(in_ch=self.hidden_size, dim=embed_dim,
                                           use_edge_gate=use_edge_gate)
        self.cls_head = nn.Linear(self.hidden_size, num_classes)

        # [v2] token layout bookkeeping
        self.legacy_token_slice = legacy_token_slice
        self._reported_prefix = False

    @dynamo.disable
    def encode(self, x: torch.Tensor):
        """
        Run HF DINOv3 ViT encoder in eager (non-compiled) mode to avoid hidden_states wrapper issues.
        """
        return self.encoder(pixel_values=x, output_hidden_states=True)

    def forward(self, x):
        B, C, H, W = x.shape

        out = self.encode(x)
        hidden = out.hidden_states

        # use last hidden (tokens) for segmentation
        gh = H // self.patch_h
        gw = W // self.patch_w
        expected = gh * gw

        if self.legacy_token_slice:
            # OLD behaviour (kept only to re-evaluate old checkpoints): drops CLS
            # only, so the register tokens stay in the grid and every patch is
            # shifted by the number of register tokens.
            tokens = hidden[-1][:, 1:, :]
            Bn, N, Cn = tokens.shape
            if N < expected:
                pad = expected - N
                tokens = torch.cat([tokens, tokens.new_zeros(Bn, pad, Cn)], dim=1)
            elif N > expected:
                tokens = tokens[:, :expected, :]
        else:
            # [v2] The sequence is [CLS, register tokens..., patch tokens...].
            # Everything before the last gh*gw tokens is a prefix token.
            seq_len = hidden[-1].shape[1]
            n_prefix = seq_len - expected
            if n_prefix < 1:
                raise ValueError(
                    f"Token count mismatch: sequence has {seq_len} tokens but a "
                    f"{gh}x{gw} grid needs {expected} patch tokens plus CLS. "
                    f"Check that the input size ({H}x{W}) is a multiple of the patch size."
                )
            if not self._reported_prefix:
                print(f"[tokens] sequence={seq_len}, prefix (CLS+register)={n_prefix}, "
                      f"patch grid={gh}x{gw}")
                self._reported_prefix = True
            tokens = hidden[-1][:, n_prefix:, :]
            Bn, N, Cn = tokens.shape

        feat_map = tokens.transpose(1, 2).reshape(B, Cn, gh, gw)

        seg_main, seg_ds_mid = self.decoder(feat_map, (H, W))

        seg_main = F.interpolate(seg_main, size=(H, W), mode="bilinear", align_corners=False)
        seg_ds_mid = F.interpolate(seg_ds_mid, size=(H, W), mode="bilinear", align_corners=False)

        cls_logits = self.cls_head(out.last_hidden_state[:, 0])

        return cls_logits, seg_main, seg_ds_mid


class ResNetSegFormer3C(nn.Module):
    """
    [baseline] ImageNet-pretrained ResNet (timm) with the SAME edge-gated decoder and
    3-class head as the DINOv3 ViT model. Only the encoder differs.
      - decoder input: layer3 features (stride 16 -> 24x24 at 384 px, like the ViT grid)
      - classifier: global-average-pooled layer4 features
      - ImageNet mean/std normalisation applied inside forward (inputs stay 0-1 like the ViT path)
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, arch: str = "resnet50", embed_dim: int = 256, num_classes: int = 3,
                 use_edge_gate: bool = True, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(arch, pretrained=pretrained, features_only=True, out_indices=(3, 4))
        chans = self.encoder.feature_info.channels()          # [layer3, layer4]
        self.decoder = EdgeGatedViTDecoder(in_ch=chans[0], dim=embed_dim, use_edge_gate=use_edge_gate)
        self.cls_head = nn.Linear(chans[1], num_classes)
        self.register_buffer("in_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        x = (x - self.in_mean) / self.in_std
        f3, f4 = self.encoder(x)
        seg_main, seg_ds_mid = self.decoder(f3, (H, W))
        seg_main = F.interpolate(seg_main, size=(H, W), mode="bilinear", align_corners=False)
        seg_ds_mid = F.interpolate(seg_ds_mid, size=(H, W), mode="bilinear", align_corners=False)
        cls_logits = self.cls_head(f4.mean(dim=(2, 3)))
        return cls_logits, seg_main, seg_ds_mid


class RadioSegFormer3C(nn.Module):
    """
    [baseline] NVIDIA C-RADIO (agglomerative ViT distilled from CLIP/DINOv2/SAM) with the SAME
    edge-gated decoder and 3-class head as the DINOv3 ViT model. Only the encoder differs.
      - spatial tokens (patch 16 -> 24x24 at 384 px) feed the decoder
      - the summary token feeds the classifier
      - RADIO expects 0-1 inputs and normalises internally (input_conditioner), like our pipeline
    """
    def __init__(self, repo: str = "nvidia/C-RADIOv4-SO400M", embed_dim: int = 256, num_classes: int = 3,
                 use_edge_gate: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(repo, trust_remote_code=True)
        self.patch = int(getattr(self.encoder, "patch_size", 16) or 16)
        with torch.no_grad():
            s, f = self.encoder(torch.zeros(1, 3, self.patch * 2, self.patch * 2))
        self.decoder = EdgeGatedViTDecoder(in_ch=int(f.shape[-1]), dim=embed_dim, use_edge_gate=use_edge_gate)
        self.cls_head = nn.Linear(int(s.shape[-1]), num_classes)

    @dynamo.disable
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        B, C, H, W = x.shape
        summary, spatial = self.encode(x)
        gh, gw = H // self.patch, W // self.patch
        feat_map = spatial.transpose(1, 2).reshape(B, spatial.shape[-1], gh, gw)
        seg_main, seg_ds_mid = self.decoder(feat_map, (H, W))
        seg_main = F.interpolate(seg_main, size=(H, W), mode="bilinear", align_corners=False)
        seg_ds_mid = F.interpolate(seg_ds_mid, size=(H, W), mode="bilinear", align_corners=False)
        cls_logits = self.cls_head(summary)
        return cls_logits, seg_main, seg_ds_mid


class LaplacianEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1)


class EdgeAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(1, ch, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, feats, edge):
        w = self.sig(self.conv(edge))
        return feats * w


class DeepSup(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, 1)

    def forward(self, x, sizeHW):
        x = self.conv(x)
        return F.interpolate(x, size=sizeHW, mode="bilinear", align_corners=False)


class Conv3x3(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class EdgeGate(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(1, ch, 1)
        self.sig = nn.Sigmoid()

    def forward(self, feat: torch.Tensor, edge: torch.Tensor):
        w = self.sig(self.conv(edge))
        return feat * w


class EdgeGatedViTDecoder(nn.Module):
    """
    ViT-style decoder for a single feature grid (e.g., DINOv3 ViT 24x24) with edge gating.
    """
    def __init__(self, in_ch=1152, dim=256, use_edge_gate: bool = True):
        super().__init__()

        self.use_edge_gate = use_edge_gate  # [v2] ablation switch
        self.proj = nn.Conv2d(in_ch, dim, 1)

        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv3x3(dim)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv3x3(dim)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv3x3(dim)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv3x3(dim)
        )

        self.edge = LaplacianEdge()
        self.egate = EdgeGate(dim) if use_edge_gate else None

        self.refine = Conv3x3(dim)

        self.ds = nn.Conv2d(dim, 1, 1)
        self.out = nn.Conv2d(dim, 1, 1)

    def forward(self, feat24: torch.Tensor, img_size: Tuple[int, int]):
        x = self.proj(feat24)
        x = self.mix(x)

        x = self.up1(x)      # 48
        x = self.up2(x)      # 96
        mid = self.up3(x)    # 192
        x = self.up4(mid)    # 384

        if self.use_edge_gate:
            edge = torch.abs(self.edge(mid.mean(dim=1, keepdim=True)))
            edge_up = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = self.egate(x, edge_up)

        x = self.refine(x)

        ds_out = self.ds(mid)  # logits

        out = self.out(x)      # logits
        out = F.interpolate(out, img_size, mode="bilinear", align_corners=False)

        return out, ds_out
# ============================================================
#                 LOSS FUNCTIONS (O2-A)
# ============================================================

def dice_loss_from_logits(logits: torch.Tensor,
                          targets: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def boundary_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    EARLY IMPROVED boundary loss:
    enhances segmentation edges using Sobel filters
    """
    sobel = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype=torch.float32,
        device=pred.device,
    ).view(1, 1, 3, 3)

    with torch.no_grad():
        dy = F.conv2d(target, sobel, padding=1)
        dx = F.conv2d(target, sobel.transpose(2, 3), padding=1)
        edges = torch.sqrt(dx * dx + dy * dy + 1e-6)

    return F.l1_loss(torch.sigmoid(pred), edges)


# ============================================================
#          Combined segmentation loss + IoU helper
# ============================================================

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss on probabilities.
    """
    prob = torch.sigmoid(pred)
    num = 2 * (prob * target).sum(dim=(1,2,3))
    den = (prob + target).sum(dim=(1,2,3))
    return 1.0 - (num + eps) / (den + eps)


def boundary_map(gt: torch.Tensor) -> torch.Tensor:
    """
    Builds a blurred edge map using Canny + Gaussian.
    """
    gt_np = gt.detach().cpu().numpy().astype(np.uint8)
    blurred = np.zeros_like(gt_np, dtype=np.float32)
    for i in range(gt_np.shape[0]):
        edge = cv2.Canny(gt_np[i,0] * 255, 50, 150)
        edge = cv2.GaussianBlur(edge, (7,7), 0)
        blurred[i,0] = edge / 255.0
    return torch.tensor(blurred, device=gt.device)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        bmap = boundary_map(gt)
        loss = self.bce(pred, bmap)
        # reduce per sample
        return loss.mean(dim=(1,2,3))


class CombinedSegLoss(nn.Module):
    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0, w_bound: float = 0.25):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = dice_loss
        self.bound = BoundaryLoss()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_bound = w_bound

    def forward(self, pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        # per-pixel BCE
        bce_loss = self.bce(pred, target).mean(dim=(1,2,3))
        # per-sample dice
        dice_loss_val = self.dice(pred, target)
        # per-sample boundary ([v2] skipped entirely when its weight is 0)
        if self.w_bound > 0:
            bound_loss = self.bound(pred, target)
        else:
            bound_loss = torch.zeros_like(bce_loss)

        loss = (
            self.w_bce * bce_loss +
            self.w_dice * dice_loss_val +
            self.w_bound * bound_loss
        )
        if reduction == "none":
            return loss
        return loss.mean()


class O3_5_Loss(nn.Module):
    """
    Deep supervision wrapper for the edge-refined decoder (main + aux mid).
    """
    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base = base_loss

    def forward(self,
                pred_main: torch.Tensor,
                pred_ds_mid: torch.Tensor,
                gt: torch.Tensor) -> torch.Tensor:
        loss_main = self.base(pred_main, gt)
        loss_ds = self.base(pred_ds_mid, gt)
        return loss_main + 0.3 * loss_ds


class O3_5_DecoderLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, ds_weight: float = 0.3):
        super().__init__()
        self.base = base_loss
        self.ds_weight = ds_weight  # [v2] 0 disables deep supervision

    def forward(self, main: torch.Tensor, ds_mid: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        loss_main = self.base(main, gt, reduction=reduction)
        if self.ds_weight <= 0:
            return loss_main
        loss_ds = self.base(ds_mid, gt, reduction=reduction)
        return loss_main + self.ds_weight * loss_ds


class BalancedFocalLoss(nn.Module):
    """
    Focal + class-weighted CE with optional label smoothing for pristine only.
    """
    def __init__(self,
                 gamma: float = 2.0,
                 class_weights: torch.Tensor = None,
                 label_smooth_pristine: float = 0.05):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.label_smooth_pristine = label_smooth_pristine

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        one_hot = F.one_hot(target, num_classes=num_classes).float()
        if self.label_smooth_pristine > 0:
            mask = (target == 0).float().unsqueeze(1)
            eps = self.label_smooth_pristine
            smoothed = one_hot * (1.0 - mask * eps) + (mask * eps) / float(num_classes)
        else:
            smoothed = one_hot

        pt = (probs * smoothed).sum(dim=1).clamp(min=1e-6)
        focal = (1.0 - pt).pow(self.gamma)

        ce = -(smoothed * log_probs).sum(dim=1)
        if self.class_weights is not None:
            alpha = self.class_weights.to(logits.device)[target]
            ce = ce * alpha

        loss = focal * ce
        return loss.mean()


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes IoU on sigmoid-thresholded masks.
    """
    pred_mask = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_mask * target).sum(dim=(1,2,3))
    union = (pred_mask + target - pred_mask * target).sum(dim=(1,2,3))
    return (intersection + 1e-6) / (union + 1e-6)


def compute_iou_tampered_only(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    IoU averaged only over samples that contain tampered pixels (gt.sum>0).
    pred: raw logits [B,1,H,W]
    gt: binary mask [B,1,H,W]
    """
    pred_mask = (torch.sigmoid(pred) > 0.5).float()
    tampered_mask = (gt.sum(dim=(1,2,3)) > 0)

    if tampered_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred_t = pred_mask[tampered_mask]
    gt_t = gt[tampered_mask]

    intersection = (pred_t * gt_t).sum((1,2,3))
    union = (pred_t + gt_t - pred_t * gt_t).sum((1,2,3))
    iou = (intersection + 1e-6) / (union + 1e-6)

    # [v2] The x1.20 bonus for masks under 200 px was removed: it inflated the
    # reported mean IoU and could push a per-image IoU above 1.

    return iou.mean()


def evaluate_epoch_metrics(model: nn.Module,
                           loader: DataLoader,
                           device: torch.device,
                           cls_loss_fn,
                           seg_loss_fn,
                           lambda_seg: float = 6.0,
                           use_amp: bool = True,
                           channels_last: bool = False):
    """
    Lightweight eval to compute loss/acc/IoU per epoch on a given loader.
    Safe version: seg loss only on tampered+mask and guards against non-finite values.
    """
    model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_seg = 0.0
    total_correct = 0
    total_samples = 0
    iou_sum = 0.0
    iou_count = 0
    y_true_eval: List[int] = []
    y_pred_eval: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)
            y = batch["y_cls"].to(device)
            mask = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device).view(-1)

            amp_enabled = use_amp and (x.device.type == "cuda") and not AMP_OFF
            with torch.amp.autocast(device_type="cuda" if x.device.type == "cuda" else "cpu",
                                    dtype=AMP_DTYPE,
                                    enabled=amp_enabled):
                cls_logits, seg_main, seg_ds_mid = model(x)
            # [fix-amp] losses and metrics are computed in float32
            cls_logits, seg_main, seg_ds_mid = cls_logits.float(), seg_main.float(), seg_ds_mid.float()
            with torch.amp.autocast(device_type="cuda" if x.device.type == "cuda" else "cpu", enabled=False):
                loss_cls = cls_loss_fn(cls_logits, y)

                tampered_mask = (y == 1) & (has_mask > 0.5)
                if tampered_mask.any():
                    seg_main_m = seg_main[tampered_mask]
                    seg_ds_mid_m = seg_ds_mid[tampered_mask]
                    mask_m = mask[tampered_mask]

                    mask_m = torch.clamp(mask_m, 0.0, 1.0)

                    area = mask_m.flatten(1).mean(dim=1)
                    weights = torch.clamp(1.0 / (area + 1e-6), max=2.0)

                    per_sample_loss = seg_loss_fn(seg_main_m, seg_ds_mid_m, mask_m, reduction="none")

                    finite = torch.isfinite(per_sample_loss)
                    if not finite.all():
                        per_sample_loss = per_sample_loss[finite]
                        weights = weights[finite]

                    if per_sample_loss.numel() == 0:
                        loss_seg = torch.tensor(0.0, device=device)
                    else:
                        loss_seg = (per_sample_loss * weights).mean()
                else:
                    loss_seg = torch.tensor(0.0, device=device)

                if loss_seg.dim() > 0:
                    loss_seg = loss_seg.mean()
                if not torch.isfinite(loss_seg):
                    loss_seg = torch.tensor(0.0, device=device)

                loss = loss_cls + lambda_seg * loss_seg
                if loss.dim() > 0:
                    loss = loss.mean()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_seg += float(loss_seg.item()) if hasattr(loss_seg, "item") else float(loss_seg)

            preds = cls_logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()
            y_true_eval.extend(y.cpu().tolist())
            y_pred_eval.extend(preds.cpu().tolist())

            # IoU on tampered samples with masks
            tampered_mask_eval = (y == 1) & (has_mask > 0.5)
            if tampered_mask_eval.any():
                seg_main_m = seg_main[tampered_mask_eval]
                mask_m = mask[tampered_mask_eval]
                iou_batch = compute_iou_tampered_only(seg_main_m, mask_m)
                iou_sum += iou_batch.item()
                iou_count += 1

    from sklearn.metrics import precision_recall_fscore_support
    f1_macro = precision_recall_fscore_support(
        np.array(y_true_eval), np.array(y_pred_eval),
        average="macro", zero_division=0
    )[2] if len(y_true_eval) > 0 else 0.0

    n_batches = max(len(loader), 1)
    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls / n_batches,
        "seg_loss": total_seg / n_batches,
        "acc": total_correct / max(total_samples, 1),
        "iou": iou_sum / max(iou_count, 1),
        "f1": f1_macro,
    }


# ============================================================
#                   SMALL-OBJECT IoU FIX (O2-A)
# ============================================================

def small_object_iou(pred_mask: np.ndarray,
                     gt_mask: np.ndarray,
                     min_area: int = 80) -> float:
    """
    Penalizes noise less for tiny tampered regions
    """
    gt_bin = (gt_mask > 0.5).astype(np.uint8)
    pred_bin = (pred_mask > 0.5).astype(np.uint8)

    area = gt_bin.sum()
    if area < min_area:
        k = np.ones((3,3), np.uint8)
        gt_bin = cv2.dilate(gt_bin, k, 1)
        pred_bin = cv2.dilate(pred_bin, k, 1)

    inter = (gt_bin * pred_bin).sum()
    union = ((gt_bin + pred_bin) > 0).sum()
    return inter / (union + 1e-6)


# ============================================================
#                      TTA  (O2-A)
# ============================================================

def tta_predict_3class_and_seg(model: nn.Module,
                               x: torch.Tensor,
                               use_tta: bool = True,
                               use_amp: bool = True,
                               channels_last: bool = False,
                               heavy_tta: bool = False):
    """
    Corrected TTA for decoder:
    - uses seg_main only
    - multi-flip ensemble for robustness
    - handles full batches (returns Bx3 class probs and Bx1xHxW seg probs)
    - optional heavier TTA with rotations for test-time robustness
    """
    device_type = x.device.type
    amp_enabled = use_amp and device_type == "cuda" and not AMP_OFF

    def forward(inp: torch.Tensor):
        if channels_last:
            inp = inp.to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type=device_type,
                                dtype=AMP_DTYPE,
                                enabled=amp_enabled):
            cls_logits, seg_main, seg_ds_mid = model(inp)
        # [fix-amp] probabilities in float32 (bfloat16 cannot be converted to numpy)
        cls_out = torch.softmax(cls_logits.float(), dim=1)
        seg_out = torch.sigmoid(seg_main.float())
        return cls_out.detach(), seg_out.detach()

    transforms = [
        ("none", lambda t: t, lambda s: s),
    ]

    if use_tta:
        transforms.extend([
            ("hflip", lambda t: torch.flip(t, dims=[3]), lambda s: torch.flip(s, dims=[3])),
            ("vflip", lambda t: torch.flip(t, dims=[2]), lambda s: torch.flip(s, dims=[2])),
            ("hvflip", lambda t: torch.flip(t, dims=[2, 3]), lambda s: torch.flip(s, dims=[2, 3])),
        ])
        if heavy_tta:
            transforms.extend([
                ("rot90",  lambda t: torch.rot90(t, k=1, dims=[2,3]),
                           lambda s: torch.rot90(s, k=3, dims=[2,3])),
                ("rot270", lambda t: torch.rot90(t, k=3, dims=[2,3]),
                           lambda s: torch.rot90(s, k=1, dims=[2,3])),
            ])

    cls_sum = None
    seg_sum = None

    for _, t_in, t_out in transforms:
        cls_pred, seg_pred = forward(t_in(x))
        seg_pred = t_out(seg_pred)
        if cls_sum is None:
            cls_sum = cls_pred
            seg_sum = seg_pred
        else:
            cls_sum += cls_pred
            seg_sum += seg_pred

    n = float(len(transforms))
    cls_mean = (cls_sum / n).cpu().numpy()
    seg_mean = (seg_sum / n).cpu().numpy()
    return cls_mean, seg_mean


# ============================================================
#                REAL IMAGE PRIOR (O2-A VERSION)
# ============================================================

def real_prior_score(img_np: np.ndarray) -> float:
    """
    EARLY VERSION — using PRNU-like noise residue
    """
    img_255 = (img_np * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_255, cv2.COLOR_RGB2GRAY)

    prnu = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5,5), 0)
    prnu_std = np.tanh(prnu.std() / 20.0)

    return float(np.clip(prnu_std, 0.0, 1.0))


# ============================================================
#                   BAYESIAN FUSION (O2-A)
# ============================================================

def bayesian_fusion_o2a(p_raw, seg_mean, realp,
                        w_cls=1.0, w_seg=0.10, w_real=0.25):
    """
    Original early-improved fusion (stable):
    - classifier stays dominant
    - segmentation nudges tampered
    - realness boosts pristine
    """

    eps = 1e-6
    p = np.clip(p_raw, eps, 1 - eps)
    p /= p.sum()

    logits = np.log(p)

    logits_seg = np.array([
        -(seg_mean),
         seg_mean * 2.0,
        -(seg_mean * 0.5)
    ])

    logits_real = np.array([
        realp * 1.5,
       -realp * 1.0,
       -realp * 0.4
    ])

    fused = (
        w_cls * logits +
        w_seg * logits_seg +
        w_real * logits_real
    )

    fused -= fused.max()
    fused = np.exp(fused)
    fused /= fused.sum() + eps
    return fused


# ============================================================
#                 TRAINING LOOP (O2-A)
# ============================================================

def train_3class(model: nn.Module,
                 loader: DataLoader,
                 device: torch.device,
                 epochs: int,
                 lr: float,
                 lambda_seg: float = 3.0,
                 thr_mask: float = 0.5,
                 use_amp: bool = True,
                 outdir: Path = None,
                 eval_loader: DataLoader = None,
                 channels_last: bool = False,
                 save_best: bool = True,
                 best_metric: str = "f1",
                 freeze_warmup_epochs: int = 0,
                 lr_decoder: float = None,
                 w_bound: float = 0.25,
                 ds_weight: float = 0.3,
                 freeze_backbone: bool = False):
    """
    O2-A training loop (ultra-safe):
    - BCE + Dice + Boundary loss
    - segmentation only on tampered samples WITH valid masks
    - skips batches with non-finite outputs or losses instead of stopping training
    - AMP is truly optional via `use_amp`
    """

    # [v2] two parameter groups: backbone at `lr`, decoder + classifier at `lr_decoder`
    if lr_decoder is None:
        lr_decoder = lr
    if hasattr(model, "encoder"):
        enc_ids = {id(p) for p in model.encoder.parameters()}
        enc_params = [p for p in model.parameters() if id(p) in enc_ids]
        head_params = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW(
            [{"params": enc_params, "lr": lr},
             {"params": head_params, "lr": lr_decoder}],
            weight_decay=1e-4,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print(f"[train] lr backbone={lr:g}, lr decoder/head={lr_decoder:g}, "
          f"w_bound={w_bound}, ds_weight={ds_weight}, freeze_backbone={freeze_backbone}")
    scaler = make_grad_scaler(device, use_amp and (not AMP_OFF) and AMP_DTYPE is torch.float16)  # [fix-amp]

    cls_weights = torch.tensor([1.0, 1.3, 1.8], device=device)
    cls_loss_fn = BalancedFocalLoss(
        gamma=2.0,
        class_weights=cls_weights,
        label_smooth_pristine=0.02,
    )
    seg_base = CombinedSegLoss(w_bce=1.0, w_dice=1.0, w_bound=w_bound)
    seg_loss_fn = O3_5_DecoderLoss(seg_base, ds_weight=ds_weight)

    logs = {
        "loss": [], "cls_loss": [], "seg_loss": [], "acc": [], "iou": [],
        "epoch_sec": [],
        "test_loss": [], "test_cls_loss": [], "test_seg_loss": [], "test_acc": [], "test_iou": []
    }

    best_score = -float("inf")
    best_epoch = -1
    best_path = outdir / "best_checkpoint.pt" if (outdir is not None) else None
    unfroze = (freeze_warmup_epochs <= 0)

    # [resume] per-epoch resume: last_epoch.pt holds model/opt/scaler/logs/best state.
    # If it exists in outdir the run continues from the next epoch; RESUME=0 disables.
    resume_path = (outdir / "last_epoch.pt") if (outdir is not None) else None
    start_epoch = 0
    if resume_path is not None and resume_path.exists() and env_flag("RESUME", True):
        try:
            rs = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(rs["model"])
            opt.load_state_dict(rs["opt"])
            if "scaler" in rs and rs["scaler"] is not None:
                scaler.load_state_dict(rs["scaler"])
            logs = rs["logs"]
            best_score = rs["best_score"]
            best_epoch = rs["best_epoch"]
            start_epoch = int(rs["epoch"])
            if "rng" in rs:
                random.setstate(rs["rng"]["python"]); np.random.set_state(rs["rng"]["numpy"])
                torch.set_rng_state(rs["rng"]["torch"].cpu())   # [fix-resume] map_location moved it to the GPU
                if torch.cuda.is_available() and rs["rng"].get("cuda") is not None:
                    torch.cuda.set_rng_state_all([t.cpu() for t in rs["rng"]["cuda"]])
            if start_epoch >= freeze_warmup_epochs and not freeze_backbone:
                set_encoder_trainable(model, True); unfroze = True
            print(f"[resume] continuing from {resume_path} at epoch {start_epoch+1}/{epochs} "
                  f"(best {best_metric}={best_score:.4f} @ epoch {best_epoch})", flush=True)
        except Exception as e:
            # [fix-resume] the model/optimizer may already be partly overwritten here, so "starting from
            # scratch" would silently mix states. Stop and let the user decide.
            raise RuntimeError(f"[resume] could not load {resume_path} ({e}). Move that file away or set "
                               f"RESUME=0 to start this run from scratch.") from e

    for epoch in range(start_epoch, epochs):
        # Unfreeze encoder after warmup epochs
        if (not unfroze) and (not freeze_backbone) and (epoch >= freeze_warmup_epochs):
            set_encoder_trainable(model, True)
            unfroze = True

        model.train()
        t_epoch = time.perf_counter()
        loop = tqdm(loader, desc=f"[train] epoch {epoch+1}/{epochs}")
        amp_backoffs = 0  # [diag] GradScaler back-off events in this epoch
        skipped_steps = 0  # [fix-amp] optimizer steps skipped for non-finite gradients

        total_loss = 0.0
        total_cls = 0.0
        total_seg = 0.0
        total_correct = 0
        total_samples = 0
        iou_sum = 0.0
        iou_count = 0

        for batch in loop:
            x = batch["pixel_values"].to(device, non_blocking=True)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)

            y = batch["y_cls"].to(device)
            mask = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device).view(-1)

            opt.zero_grad(set_to_none=True)

            amp_enabled = use_amp and (x.device.type == "cuda") and not AMP_OFF
            try:
                with torch.amp.autocast(
                    device_type="cuda" if x.device.type == "cuda" else "cpu",
                    dtype=AMP_DTYPE,
                    enabled=amp_enabled,
                ):
                    cls_logits, seg_main, seg_ds_mid = model(x)
                # [fix-amp] losses are computed in float32
                cls_logits, seg_main, seg_ds_mid = cls_logits.float(), seg_main.float(), seg_ds_mid.float()
                with torch.amp.autocast(device_type="cuda" if x.device.type == "cuda" else "cpu", enabled=False):

                    # guard model outputs
                    if (not torch.isfinite(cls_logits).all()
                        or not torch.isfinite(seg_main).all()
                        or not torch.isfinite(seg_ds_mid).all()):
                        print("[train] non-finite model outputs; skipping batch")
                        continue

                    # classification loss (always)
                    loss_cls = cls_loss_fn(cls_logits, y)
                    if not torch.isfinite(loss_cls):
                        print("[train] non-finite classification loss; skipping batch")
                        continue

                    # seg loss only on tampered + mask samples
                    tampered_mask = (y == 1) & (has_mask > 0.5)
                    if tampered_mask.any():
                        seg_main_m = seg_main[tampered_mask]
                        seg_ds_mid_m = seg_ds_mid[tampered_mask]
                        mask_m = mask[tampered_mask]

                        mask_m = torch.clamp(mask_m, 0.0, 1.0)

                        # area-weighted seg loss (boost tiny regions gently)
                        area = mask_m.flatten(1).mean(dim=1)
                        weights = torch.clamp(1.0 / (area + 1e-6), max=2.0)

                        per_sample_loss = seg_loss_fn(
                            seg_main_m, seg_ds_mid_m, mask_m, reduction="none"
                        )  # shape [B_tampered]

                        # drop any non-finite samples
                        finite = torch.isfinite(per_sample_loss)
                        if not finite.all():
                            bad = (~finite).sum().item()
                            print(f"[train] dropping {bad} non-finite seg samples in batch")
                            per_sample_loss = per_sample_loss[finite]
                            weights = weights[finite]

                        if per_sample_loss.numel() == 0:
                            loss_seg = torch.tensor(0.0, device=device)
                        else:
                            loss_seg = (per_sample_loss * weights).mean()
                    else:
                        loss_seg = torch.tensor(0.0, device=device)

                    if loss_seg.dim() > 0:
                        loss_seg = loss_seg.mean()
                    if not torch.isfinite(loss_seg):
                        print(f"[train] non-finite seg loss ({loss_seg}); zeroing it for this batch")
                        loss_seg = torch.tensor(0.0, device=device)

                    # Ramp seg weight during early epochs to ease stabilization
                    seg_scale = min(1.0, (epoch + 1) / max(3, freeze_warmup_epochs + 1))
                    total_loss_batch = loss_cls + (lambda_seg * seg_scale) * loss_seg
                    if total_loss_batch.dim() > 0:
                        total_loss_batch = total_loss_batch.mean()

                # final finiteness check
                if not torch.isfinite(total_loss_batch):
                    print("[train] non-finite total loss; skipping batch")
                    continue

            except RuntimeError as e:
                print(f"[train] runtime error in forward pass, skipping batch: {e}")
                continue

            scaler.scale(total_loss_batch).backward()
            # Grad clipping to prevent NaNs with large backbone
            scaler.unscale_(opt)
            _gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            _scale_before = scaler.get_scale() if amp_enabled else 1.0
            if not scaler.is_enabled() and not torch.isfinite(_gnorm):
                # [fix-amp] without GradScaler nothing else would stop a non-finite update
                skipped_steps += 1
                print(f"\n[train] non-finite gradient norm; step skipped ({skipped_steps} this epoch)")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.step(opt)
            scaler.update()
            # [diag] GradScaler lowers its scale only when the gradients of this step were
            # non-finite (the optimizer step is then skipped). Report every such event.
            if amp_enabled and scaler.is_enabled() and scaler.get_scale() < _scale_before:
                amp_backoffs += 1
                if amp_backoffs <= 20 or amp_backoffs % 50 == 0:
                    _bad = [n for n, p_ in model.named_parameters()
                            if p_.grad is not None and not torch.isfinite(p_.grad).all()]
                    print(f"\n[amp] non-finite gradients (event {amp_backoffs}): step skipped, scale "
                          f"{_scale_before:.3e} -> {scaler.get_scale():.3e}; loss_cls={float(loss_cls):.4f} "
                          f"loss_seg={float(loss_seg):.4f} n_tampered={int(tampered_mask.sum())}; "
                          f"{len(_bad)} params affected, first: {_bad[:3]}")

            # Metrics
            total_loss += float(total_loss_batch.item())
            total_cls += float(loss_cls.item())
            total_seg += float(loss_seg.item())

            preds = cls_logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()

            # IoU on tampered subset with masks
            if tampered_mask.any():
                seg_main_m_eval = seg_main[tampered_mask].detach()
                mask_m_eval = mask[tampered_mask].detach()
                iou_batch = compute_iou_tampered_only(seg_main_m_eval, mask_m_eval)
                iou_sum += iou_batch.item()
                iou_count += 1

            loop.set_postfix({
                "cls": f"{loss_cls.item():.4f}",
                "seg": f"{loss_seg.item():.4f}",
            })

        n_batches = max(len(loader), 1)
        logs["loss"].append(total_loss / n_batches)
        logs["cls_loss"].append(total_cls / n_batches)
        logs["seg_loss"].append(total_seg / n_batches)
        logs["acc"].append(total_correct / max(total_samples, 1))
        logs["iou"].append(iou_sum / max(iou_count, 1))
        # [fix-amp] stop at once if the model has diverged, instead of skipping every batch until the last epoch
        if total_samples == 0 or any(not torch.isfinite(p_).all() for p_ in model.parameters()):
            raise RuntimeError(f"training diverged in epoch {epoch+1}: no batch was usable or the weights are "
                               f"non-finite. Best checkpoint so far: epoch {best_epoch} ({best_path}).")
        logs["epoch_sec"].append(time.perf_counter() - t_epoch)

        if eval_loader is not None:
            eval_metrics = evaluate_epoch_metrics(
                model, eval_loader, device, cls_loss_fn, seg_loss_fn,
                lambda_seg=lambda_seg, use_amp=use_amp, channels_last=channels_last
            )
            logs["test_loss"].append(eval_metrics["loss"])
            logs["test_cls_loss"].append(eval_metrics["cls_loss"])
            logs["test_seg_loss"].append(eval_metrics["seg_loss"])
            logs["test_acc"].append(eval_metrics["acc"])
            logs["test_iou"].append(eval_metrics["iou"])

            monitor_value = eval_metrics.get(best_metric, eval_metrics.get("acc", float("-inf")))
        else:
            logs["test_loss"].append(float("nan"))
            logs["test_cls_loss"].append(float("nan"))
            logs["test_seg_loss"].append(float("nan"))
            logs["test_acc"].append(float("nan"))
            logs["test_iou"].append(float("nan"))
            monitor_value = total_correct / max(total_samples, 1)

        # [v2-log] one summary line per epoch (logging only)
        _vf1 = logs.setdefault("val_f1", [])
        _vf1.append(float(eval_metrics.get("f1", float("nan"))) if eval_loader is not None else float("nan"))
        print(f"[epoch {epoch+1}/{epochs}] train_loss={logs['loss'][-1]:.4f} train_acc={logs['acc'][-1]:.4f} "
              f"train_iou={logs['iou'][-1]:.4f} val_loss={logs['test_loss'][-1]:.4f} val_acc={logs['test_acc'][-1]:.4f} "
              f"val_f1={_vf1[-1]:.4f} val_iou={logs['test_iou'][-1]:.4f} monitor={monitor_value:.4f} "
              f"best={max(best_score, monitor_value):.4f} sec={logs['epoch_sec'][-1]:.0f} "
              f"amp={_AMP_NAME} amp_scale={scaler.get_scale():.3e} amp_backoffs={amp_backoffs} skipped_steps={skipped_steps}", flush=True)
        if outdir is not None:
            try:
                with open(Path(outdir) / "train_logs.json", "w", encoding="utf-8") as _f:
                    json.dump({k: v for k, v in logs.items() if isinstance(v, list)}, _f)
            except Exception:
                pass

        if save_best and best_path is not None:
            if monitor_value > best_score:
                best_score = monitor_value
                best_epoch = epoch + 1
                torch.save({
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "metric": best_metric,
                    "score": best_score,
                }, best_path)

        # [resume] save full training state at the end of every epoch (atomic rename)
        if resume_path is not None:
            tmp = resume_path.with_suffix(".tmp")
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
                "logs": logs,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "epoch": epoch + 1,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            }, tmp)
            os.replace(tmp, resume_path)
    if outdir is not None:
        plot_training_curves(logs, outdir)
        if eval_loader is not None:
            plot_train_test_curves(logs, outdir)
    if resume_path is not None and resume_path.exists() and env_flag("KEEP_LAST_EPOCH", False) is False:
        try:
            resume_path.unlink()   # [resume] training finished; drop the (model+optimizer) state file
        except OSError:
            pass

    if save_best and best_path is not None and best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        logs["best_epoch"] = state.get("epoch", best_epoch)
        logs["best_score"] = state.get("score", best_score)

    return logs


# ============================================================
#                    EVALUATION LOOP (O2-A)
# ============================================================

def evaluate_3class(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device,
                    thr_mask: float = 0.5,
                    fusion_seg_thr: float = 0.15,
                    use_tta: bool = True,
                    use_amp: bool = True,
                    channels_last: bool = False,
                    heavy_tta: bool = False):
    """
    Corrected evaluation:
    - uses seg_main only
    - TTA-aware
    - IoU only for tampered class (1)
    - Bayesian fusion for classification with optional seg confidence gate
    """

    y_true = []
    probs_raw = []
    probs_fused = []
    ious = []
    iou_per_sample = []
    mask_areas = []
    has_masks = []
    infer_sec = 0.0   # [v2] model time only (includes TTA views)
    infer_n = 0

    thr_mask = float(np.clip(thr_mask, 1e-6, 1 - 1e-6))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="[eval]"):
            x = batch["pixel_values"].to(device, non_blocking=True)
            gt_cls = batch["y_cls"].cpu().numpy()
            gt_mask = batch["mask"].cpu().numpy()
            has_mask = batch["has_mask"].cpu().numpy()
            img_np = batch["img_np"]
            if isinstance(img_np, torch.Tensor):
                img_np = img_np.cpu().numpy()
            else:
                img_np = np.asarray(img_np)

            mask_area_batch = gt_mask.reshape(gt_mask.shape[0], -1).mean(axis=1)

            if x.device.type == "cuda":
                torch.cuda.synchronize()
            t_inf = time.perf_counter()
            p_raw, seg = tta_predict_3class_and_seg(
                model,
                x,
                use_tta=use_tta,
                use_amp=use_amp,
                channels_last=channels_last,
                heavy_tta=heavy_tta,
            )
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            infer_sec += time.perf_counter() - t_inf
            infer_n += int(x.shape[0])
            seg = np.clip(seg, 1e-6, 1 - 1e-6)

            batch_size = len(gt_cls)
            for i in range(batch_size):
                y_true.append(int(gt_cls[i]))

                try:
                    realp = real_prior_score(np.asarray(img_np[i]))
                except Exception:
                    realp = 0.5

                seg_mean = float(seg[i].mean())
                if seg_mean < fusion_seg_thr:
                    pf = p_raw[i]
                else:
                    pf = bayesian_fusion_o2a(p_raw[i], seg_mean, realp)

                probs_raw.append(p_raw[i])
                probs_fused.append(pf)
                mask_areas.append(float(mask_area_batch[i]))
                has_masks.append(float(has_mask[i]))

                iou_val = float("nan")
                if int(gt_cls[i]) == 1 and has_mask[i] > 0.5:
                    gt_t = torch.from_numpy(gt_mask[i:i+1]).float().to(device)
                    seg_t = torch.from_numpy(seg[i:i+1]).float().to(device)
                    seg_t = torch.clamp(seg_t, 1e-4, 1.0 - 1e-4)
                    seg_logits_like = torch.logit(seg_t)
                    iou_val = compute_iou_tampered_only(seg_logits_like, gt_t).item()
                    ious.append(iou_val)

                iou_per_sample.append(iou_val)

    y_true = np.array(y_true)
    probs_raw = np.array(probs_raw)
    probs_fused = np.array(probs_fused)
    mask_areas = np.array(mask_areas)
    has_masks = np.array(has_masks)
    iou_per_sample = np.array(iou_per_sample)

    y_pred_raw = probs_raw.argmax(1)
    y_pred_fused = probs_fused.argmax(1)

    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

    f1_raw = precision_recall_fscore_support(y_true, y_pred_raw, average="macro", zero_division=0)[2]
    f1_fused = precision_recall_fscore_support(y_true, y_pred_fused, average="macro", zero_division=0)[2]

    acc_raw = (y_pred_raw == y_true).mean()
    acc_fused = (y_pred_fused == y_true).mean()

    try:
        roc_raw = roc_auc_score(y_true, probs_raw, multi_class="ovr")
        roc_fused = roc_auc_score(y_true, probs_fused, multi_class="ovr")
    except:
        roc_raw = float("nan")
        roc_fused = float("nan")

    cm_raw = confusion_matrix(y_true, y_pred_raw, labels=[0,1,2])
    cm_fused = confusion_matrix(y_true, y_pred_fused, labels=[0,1,2])

    return {
        "acc_raw": acc_raw,
        "acc_fused": acc_fused,
        "f1_raw": f1_raw,
        "f1_fused": f1_fused,
        "roc_raw": roc_raw,
        "roc_fused": roc_fused,
        "cm_raw": cm_raw,
        "cm_fused": cm_fused,
        "ious": ious,
        "iou_per_sample": iou_per_sample,
        "mask_area": mask_areas,
        "has_mask": has_masks,
        "y_true": y_true,
        "probs_raw": probs_raw,
        "probs_fused": probs_fused,
        "ms_per_image": 1000.0 * infer_sec / max(infer_n, 1),
    }


def bucket_metrics_by_area(results: Dict[str, Any],
                           area_thr: float = 1/16.0) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-subset metrics split by tamper area.
    Buckets:
      pristine (label=0)
      fake (label=2)
      tampered_small (label=1, has_mask, mask_area < area_thr)
      tampered_large (label=1, has_mask, mask_area >= area_thr)
      tampered_nomask (label=1, has_mask==0)
    """
    y_true = results["y_true"]
    probs_raw = results["probs_raw"]
    probs_fused = results["probs_fused"]
    mask_area = results.get("mask_area")
    has_mask = results.get("has_mask")
    iou_per_sample = results.get("iou_per_sample")

    if mask_area is None or has_mask is None or iou_per_sample is None:
        return {}

    mask_area = np.asarray(mask_area)
    has_mask = np.asarray(has_mask)
    iou_per_sample = np.asarray(iou_per_sample)

    from sklearn.metrics import precision_recall_fscore_support

    def subset_metrics(selector: np.ndarray) -> Dict[str, Any]:
        idx = np.where(selector)[0]
        if idx.size == 0:
            return {"count": 0}
        y = y_true[idx]
        pr = probs_raw[idx]
        pf = probs_fused[idx]

        y_pred_raw = pr.argmax(1)
        y_pred_fused = pf.argmax(1)

        f1_raw = precision_recall_fscore_support(
            y, y_pred_raw, labels=[0,1,2], average="macro", zero_division=0
        )[2]
        f1_fused = precision_recall_fscore_support(
            y, y_pred_fused, labels=[0,1,2], average="macro", zero_division=0
        )[2]

        iou_vals = iou_per_sample[idx]
        iou_mean = float(np.nanmean(iou_vals)) if not np.all(np.isnan(iou_vals)) else float("nan")

        return {
            "count": int(idx.size),
            "acc_raw": float((y_pred_raw == y).mean()),
            "acc_fused": float((y_pred_fused == y).mean()),
            "f1_raw": float(f1_raw),
            "f1_fused": float(f1_fused),
            "iou_mean": iou_mean,
        }

    is_pristine = (y_true == 0)
    is_fake = (y_true == 2)
    is_tampered = (y_true == 1)
    tampered_masked = is_tampered & (has_mask > 0.5)

    buckets = {
        "pristine": subset_metrics(is_pristine),
        "fake": subset_metrics(is_fake),
        "tampered_small": subset_metrics(tampered_masked & (mask_area < area_thr)),
        "tampered_large": subset_metrics(tampered_masked & (mask_area >= area_thr)),
        "tampered_nomask": subset_metrics(is_tampered & (has_mask <= 0.5)),
    }
    buckets["thr"] = area_thr
    return buckets


def print_area_bucket_metrics(tag: str, buckets: Dict[str, Any]):
    if not buckets:
        return
    thr = buckets.get("thr", 0.0)
    print(f"\n[{tag}] area buckets (tampered split at {thr:.4f} image fraction)")
    order = ["pristine", "fake", "tampered_small", "tampered_large", "tampered_nomask"]
    for key in order:
        b = buckets.get(key, {})
        count = b.get("count", 0)
        if count == 0:
            print(f"  {key}: n=0")
            continue
        iou_mean = b.get("iou_mean", float("nan"))
        iou_str = f"{iou_mean:.4f}" if not np.isnan(iou_mean) else "nan"
        print(f"  {key}: n={count} | acc_raw={b.get('acc_raw', float('nan')):.4f} "
              f"acc_fused={b.get('acc_fused', float('nan')):.4f} "
              f"f1_raw={b.get('f1_raw', float('nan')):.4f} "
              f"f1_fused={b.get('f1_fused', float('nan')):.4f} "
              f"iou_mean={iou_str}")
# ============================================================
#                     DATASET  (O2-A)
# ============================================================

def random_resized_crop_np(img: np.ndarray,
                           mask: np.ndarray,
                           size: int,
                           scale: Tuple[float, float] = (0.7, 1.0),
                           ratio: Tuple[float, float] = (0.9, 1.1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight random resized crop that keeps image/mask in sync.
    Falls back to original if a valid crop is not found quickly.
    """
    h, w = img.shape[:2]
    area = h * w
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(8):
        target_area = random.uniform(*scale) * area
        aspect = math.exp(random.uniform(*log_ratio))

        new_w = int(round(math.sqrt(target_area * aspect)))
        new_h = int(round(math.sqrt(target_area / aspect)))

        if 0 < new_w <= w and 0 < new_h <= h:
            x0 = random.randint(0, w - new_w)
            y0 = random.randint(0, h - new_h)

            img_crop = img[y0:y0+new_h, x0:x0+new_w]
            mask_crop = mask[y0:y0+new_h, x0:x0+new_w]

            img_crop = cv2.resize(img_crop, (size, size), interpolation=cv2.INTER_LINEAR)
            mask_crop = cv2.resize(mask_crop, (size, size), interpolation=cv2.INTER_NEAREST)
            return img_crop, mask_crop

    return img, mask


def color_jitter_np(img: np.ndarray,
                    brightness: float = 0.2,
                    contrast: float = 0.2,
                    saturation: float = 0.2,
                    hue: float = 0.02) -> np.ndarray:
    """
    Simple color jitter on uint8 image in RGB.
    """
    out = img.astype(np.float32)

    if brightness > 0:
        b_fac = random.uniform(1 - brightness, 1 + brightness)
        out *= b_fac

    if contrast > 0:
        c_fac = random.uniform(1 - contrast, 1 + contrast)
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * c_fac + mean

    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        if saturation > 0:
            s_fac = random.uniform(1 - saturation, 1 + saturation)
            hsv[..., 1] *= s_fac
        if hue > 0:
            h_delta = random.uniform(-hue, hue) * 180.0
            hsv[..., 0] = (hsv[..., 0] + h_delta) % 180.0
        out = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    return out.clip(0, 255).astype(np.uint8)


def random_blur_or_noise(img: np.ndarray,
                         p_blur: float = 0.35,
                         p_noise: float = 0.35) -> np.ndarray:
    """
    Apply Gaussian blur or additive Gaussian noise.
    """
    out = img
    if random.random() < p_blur:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    if random.random() < p_noise:
        sigma = random.uniform(2.0, 8.0)
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def random_jpeg_compress(img: np.ndarray,
                         quality_range: Tuple[int, int] = (50, 95),
                         p: float = 0.3) -> np.ndarray:
    if random.random() >= p:
        return img
    q = random.randint(quality_range[0], quality_range[1])
    params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    ok, enc = cv2.imencode(".jpg", img, params)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        return img
    return dec


class ForensicsDataset3C(Dataset):
    """
    ORIGINAL TYPE-2 DATASET:
    - image_filename
    - mask_filename
    - manipulation_type (pristine / tampered / fake)
    """

    def __init__(self, root: str, metadata_csv: str,
                 img_size: int = 384, is_train: bool = False,
                 cache_in_ram: bool = True, cache_limit: int = None,
                 id_whitelist: set = None, preload_all: bool = False,
                 mask_aug: bool = False):

        self.root = Path(root)
        self.metadata = Path(metadata_csv)
        self.img_size = img_size
        self.is_train = is_train
        self.cache_in_ram = cache_in_ram
        self.cache_limit = cache_limit
        self.preload_all = preload_all
        self.mask_aug = mask_aug  # [v2] mask-only augmentation is opt-in
        self._cache = {}  # id -> (img_uint8, mask_uint8, has_mask, cls_label)

        if not self.metadata.exists():
            raise FileNotFoundError(str(self.metadata))

        self.items = []
        with self.metadata.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_rel = row["image_filename"]
                img_path = self.root / img_rel
                img_id = Path(img_rel).stem
                if id_whitelist is not None and img_id not in id_whitelist:
                    continue

                mask_rel = row.get("mask_filename", None)
                mask_path = (self.root / mask_rel) if mask_rel else None

                manip = row.get("manipulation_type", "").lower().strip()

                if "pristine" in manip:
                    cls_label = 0

                elif ("inpaint" in manip) or ("partial" in manip) \
                     or ("building" in manip) or ("roads" in manip) \
                     or ("greenspace" in manip) or ("water" in manip):
                    cls_label = 1   # tampered

                elif ("synthetic" in manip) or ("generated" in manip) \
                     or ("gen" in manip) or ("diff" in manip):
                    cls_label = 2   # fake

                else:
                    print("[warn] unknown manipulation type:", manip)
                    cls_label = 0

                self.items.append({
                    "img": img_path,
                    "mask": mask_path,
                    "label": cls_label,
                    "id": img_id,
                })

        if len(self.items) == 0:
            raise RuntimeError("No dataset items found")

        # Optional eager preload to RAM
        if self.cache_in_ram and self.preload_all:
            for idx, item in enumerate(self.items):
                if self.cache_limit is not None and len(self._cache) >= self.cache_limit:
                    break
                img_bgr = cv2.imread(str(item["img"]))
                if img_bgr is None:
                    continue
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                mask_path = item["mask"]
                if mask_path is not None and mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        mask = np.zeros(img.shape[:2], np.uint8)
                else:
                    mask = np.zeros(img.shape[:2], np.uint8)

                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

                manip = item["label"]
                if manip == 1:
                    has_mask = 1.0 if mask.sum() > 15 else 0.0
                else:
                    mask = np.zeros_like(mask)
                    has_mask = 0.0

                img_uint8 = img.astype(np.uint8)
                mask_uint8 = mask.astype(np.uint8)

                if self.cache_limit is None or len(self._cache) < self.cache_limit:
                    self._cache[idx] = (img_uint8, mask_uint8, has_mask, manip)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        cached = self._cache.get(idx, None) if self.cache_in_ram else None
        if cached is not None:
            img_uint8, mask_uint8, has_mask, cls_label = cached
        else:
            img_bgr = cv2.imread(str(item["img"]))
            if img_bgr is None:
                raise RuntimeError(f"Failed to read {item['img']}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mask_path = item["mask"]
            if mask_path is not None and mask_path.exists():
                mask = cv2.imread(str(mask_path),
                                  cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros(img.shape[:2], np.uint8)
            else:
                mask = np.zeros(img.shape[:2], np.uint8)

            img = cv2.resize(img,
                             (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask,
                              (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)

            manip = item["label"]  # 0=pristine, 1=tampered, 2=fake

            if manip == 1:
                has_mask = 1.0 if mask.sum() > 15 else 0.0
            elif manip == 2:
                mask = np.zeros_like(mask)
                has_mask = 0.0
            else:
                mask = np.zeros_like(mask)
                has_mask = 0.0

            cls_label = manip

            img_uint8 = img.astype(np.uint8)
            mask_uint8 = mask.astype(np.uint8)

            if self.cache_in_ram and (self.cache_limit is None or len(self._cache) < self.cache_limit):
                self._cache[idx] = (img_uint8, mask_uint8, has_mask, cls_label)

        img = img_uint8
        mask = mask_uint8
        manip = cls_label

        # SIMPLE AUGMENTATIONS (O2-A)
        if self.is_train:
            # copy to avoid mutating cache
            img = img.copy()
            mask = mask.copy()

            if random.random() < 0.6:
                img, mask = random_resized_crop_np(
                    img, mask, self.img_size,
                    scale=(0.7, 1.0),
                    ratio=(0.9, 1.1)
                )

            if random.random() < 0.5:
                img = color_jitter_np(img,
                                      brightness=0.15,
                                      contrast=0.15,
                                      saturation=0.15,
                                      hue=0.03)

            img = random_blur_or_noise(img, p_blur=0.2, p_noise=0.2)
            img = random_jpeg_compress(img, quality_range=(50, 90), p=0.35)

            if np.random.rand() < 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            if np.random.rand() < 0.25:
                k = np.random.randint(1,4)
                img = np.rot90(img, k)
                mask = np.rot90(mask, k)

            # -----------------------------------------------------
            # Tiny-Manipulation Augmentation (SAFE VERSION)
            # -----------------------------------------------------
            # [v2] off by default: it edits the ground-truth mask without editing
            # the image, i.e. it injects label noise. MASK_AUG=1 turns it back on.
            if manip == 1 and self.is_train and self.mask_aug:

                # ensure mask is uint8 and contiguous
                mask = np.ascontiguousarray(mask.astype(np.uint8))

                # 1) shrink mask artificially
                if random.random() < 0.35:
                    k = random.randint(1, 4)
                    mask = cv2.erode(mask, np.ones((k, k), np.uint8))

                # 2) break large region into small blobs
                if random.random() < 0.25:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

                # 3) add tiny random speckles safely
                if random.random() < 0.20:
                    h, w = mask.shape
                    mask = np.ascontiguousarray(mask)   # critical for OpenCV
                    for _ in range(random.randint(5, 15)):
                        cx = random.randint(0, w - 1)
                        cy = random.randint(0, h - 1)
                        r  = random.randint(2, 6)
                        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)

        img_float = img.astype(np.float32) / 255.0
        mask_float = mask.astype(np.float32) / 255.0

        img_t = torch.from_numpy(img_float).permute(2,0,1).contiguous()
        mask_t = torch.from_numpy(mask_float).unsqueeze(0)

        return {
            "pixel_values": img_t,
            "mask": mask_t,
            "y_cls": torch.tensor(item["label"], dtype=torch.long),
            "has_mask": torch.tensor(has_mask, dtype=torch.float32),
            "id": item["id"],
            "img_np": img_float,
        }


# ============================================================
#                     PLOTTING HELPERS
# ============================================================

def plot_confusion_matrix(cm: np.ndarray,
                          class_names: list,
                          save_path: Path,
                          title: str):

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() * 0.6 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")

    fig.tight_layout()
    fig.savefig(save_path, dpi=320)
    plt.close(fig)


def plot_roc_macro(y_true, probs, save_path: Path, title: str):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    palette = ["#1f77b4", "#e15759", "#59a14f"]

    plotted = False
    for k in range(probs.shape[1]):
        y_bin = (y_true == k).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, probs[:,k])
            auc = roc_auc_score(y_bin, probs[:,k])
            color = palette[k % len(palette)]
            ax.plot(fpr, tpr, label=f"class {k} (AUC={auc:.3f})", linewidth=2, color=color)
            plotted = True
        except:
            continue

    ax.plot([0,1], [0,1], "--", color="gray", linewidth=1.2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=320)
    plt.close(fig)


def plot_pr_macro(y_true, probs, save_path: Path, title: str):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    palette = ["#1f77b4", "#e15759", "#59a14f"]

    plotted = False
    for k in range(probs.shape[1]):
        y_bin = (y_true == k).astype(int)
        try:
            prec, rec, _ = precision_recall_curve(y_bin, probs[:, k])
            ap = average_precision_score(y_bin, probs[:, k])
            color = palette[k % len(palette)]
            ax.plot(rec, prec, label=f"class {k} (AP={ap:.3f})", linewidth=2, color=color)
            plotted = True
        except:
            continue

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if plotted:
        ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=320)
    plt.close(fig)


def plot_iou_hist(iou_list, save_path: Path):
    fig, ax = plt.subplots(figsize=(4.8,3.2))
    ax.hist(iou_list, bins=20, range=(0,1), color="#f28e2c", edgecolor="black", alpha=0.85)
    mean_iou = float(np.mean(iou_list)) if len(iou_list) > 0 else float("nan")
    ax.axvline(mean_iou, color="red", linestyle="--", linewidth=1.8, label=f"Mean IoU {mean_iou:.3f}")
    ax.set_xlabel("IoU (tampered)")
    ax.set_ylabel("Count")
    ax.set_title("IoU distribution (tampered)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=320)
    plt.close(fig)


def plot_training_curves(logs: Dict[str, list], outdir: Path):
    """
    Plot loss, accuracy, IoU per epoch.
    """
    epochs = list(range(1, len(logs.get("loss", [])) + 1))
    if not epochs:
        return

    def _plot(y, title, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.4,3.4))
        ax.plot(epochs, y, marker="o", color="#4e79a7", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=320)
        plt.close(fig)

    _plot(logs["loss"], "Train Loss", "Loss", "train_loss.png")
    _plot(logs["acc"], "Train Accuracy", "Accuracy", "train_acc.png")
    _plot(logs["iou"], "Train IoU (tampered)", "IoU", "train_iou.png")


def plot_train_test_curves(logs: Dict[str, list], outdir: Path):
    """
    Plot train vs eval curves (eval loader metrics logged during training).
    """
    epochs = list(range(1, len(logs.get("loss", [])) + 1))
    if not epochs:
        return

    def _plot(train_y, test_y, title, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.4,3.4))
        ax.plot(epochs, train_y, marker="o", label="train", color="#4e79a7", linewidth=2)
        ax.plot(epochs, test_y, marker="s", label="eval", color="#e15759", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=320)
        plt.close(fig)

    _plot(logs["loss"], logs["test_loss"], "Loss (train vs eval)", "Loss", "train_eval_loss.png")
    _plot(logs["acc"], logs["test_acc"], "Accuracy (train vs eval)", "Accuracy", "train_eval_acc.png")


def compute_segmentation_confusion_and_iou(model: nn.Module,
                                           loader: DataLoader,
                                           device: torch.device,
                                           thr_mask: float = 0.5,
                                           use_tta: bool = True,
                                           use_amp: bool = True,
                                           channels_last: bool = False,
                                           heavy_tta: bool = False):
    """
    Computes pixel-level confusion (background vs tampered) and IoU distribution over tampered samples.
    """
    model.eval()
    tn = fp = fn = tp = 0
    iou_list = []
    dice_list = []   # [v2] per-image Dice on tampered images
    mcc_list = []    # [v2] per-image Matthews correlation on tampered images

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)

            mask_batch = batch["mask"]
            has_mask_batch = batch["has_mask"]

            if isinstance(mask_batch, torch.Tensor):
                gt_masks = mask_batch.cpu().numpy()
            else:
                gt_masks = np.asarray(mask_batch)

            if isinstance(has_mask_batch, torch.Tensor):
                has_mask_np = has_mask_batch.cpu().numpy()
            else:
                has_mask_np = np.asarray(has_mask_batch)

            _, seg_pred = tta_predict_3class_and_seg(
                model,
                x,
                use_tta=use_tta,
                use_amp=use_amp,
                channels_last=channels_last,
                heavy_tta=heavy_tta,
            )
            # seg_pred: numpy Bx1xHxW
            seg_bin = (seg_pred >= thr_mask).astype(np.uint8)

            # Flatten for confusion counts
            # [v2] binarise the ground truth instead of testing == 0 / == 1
            gt_flat = (gt_masks.reshape(-1) > 0.5).astype(np.uint8)
            pred_flat = seg_bin.reshape(-1)
            tn += int(np.sum((gt_flat == 0) & (pred_flat == 0)))
            fp += int(np.sum((gt_flat == 0) & (pred_flat == 1)))
            fn += int(np.sum((gt_flat == 1) & (pred_flat == 0)))
            tp += int(np.sum((gt_flat == 1) & (pred_flat == 1)))

            # IoU per tampered sample only
            for gt, pred, has in zip(gt_masks, seg_bin, has_mask_np):
                if has <= 0.5:
                    continue
                g = (gt > 0.5).astype(np.uint8)
                p = pred.astype(np.uint8)
                inter = np.sum((g == 1) & (p == 1))
                union = np.sum((g == 1) | (p == 1))
                if union == 0:
                    continue
                iou_list.append(float(inter / (union + 1e-6)))

                # [v2] Dice and MCC for the same image
                tp_i = float(inter)
                fp_i = float(np.sum(p == 1)) - tp_i
                fn_i = float(np.sum(g == 1)) - tp_i
                tn_i = float(g.size) - tp_i - fp_i - fn_i
                dice_list.append(2.0 * tp_i / (2.0 * tp_i + fp_i + fn_i + 1e-6))
                den = math.sqrt((tp_i + fp_i) * (tp_i + fn_i) * (tn_i + fp_i) * (tn_i + fn_i))
                mcc_list.append((tp_i * tn_i - fp_i * fn_i) / den if den > 0 else 0.0)

    conf = np.array([[tn, fp],
                     [fn, tp]], dtype=np.int64)
    return conf, iou_list, {"dice": dice_list, "mcc": mcc_list}


def plot_segmentation_summary(conf: np.ndarray,
                              iou_list: list,
                              save_path: Path,
                              epoch: int = None):
    """
    Plots a 2x2 pixel confusion matrix and IoU histogram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion heatmap
    ax0 = axes[0]
    im = ax0.imshow(conf, cmap="YlOrBr")
    for i in range(2):
        for j in range(2):
            ax0.text(j, i, f"{conf[i, j]:,}", ha="center", va="center", color="black", fontsize=12)
    ax0.set_xticks([0, 1], ["Background", "Tampered"])
    ax0.set_yticks([0, 1], ["Background", "Tampered"])
    ax0.set_xlabel("Predicted Pixels")
    ax0.set_ylabel("True Pixels")
    title_seg = "Pixel-Level Segmentation"
    if epoch is not None:
        title_seg += f" - Epoch {epoch}"
    ax0.set_title(title_seg, fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Pixel Count")

    # IoU histogram
    ax1 = axes[1]
    bins = 25
    ax1.hist(iou_list, bins=bins, range=(0, 1), color="#f28e2c", edgecolor="black", alpha=0.9)
    mean_iou = float(np.mean(iou_list)) if len(iou_list) > 0 else float("nan")
    ax1.axvline(mean_iou, color="red", linestyle="--", linewidth=2, label=f"Mean IoU: {mean_iou:.3f}")
    ax1.set_xlabel("IoU Score")
    ax1.set_ylabel("Sample Count")
    title_iou = "IoU Distribution"
    if epoch is not None:
        title_iou += f" - Epoch {epoch}"
    ax1.set_title(title_iou, fontsize=14, fontweight="bold")
    ax1.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ============================================================
#          Visualization helpers (overlays/contours)
# ============================================================

def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Converts image to uint8 RGB.
    """
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def overlay_mask(img: np.ndarray,
                 mask: np.ndarray,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 alpha: float = 0.35) -> np.ndarray:
    """
    Overlays a binary mask on the image with transparency.
    """
    img = to_uint8(img)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    mask3 = mask[..., None].astype(np.float32)
    blended = img.astype(np.float32) * (1.0 - mask3 * alpha) + color_arr * (mask3 * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def draw_contours(img: np.ndarray,
                  mask: np.ndarray,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thick: int = 2) -> np.ndarray:
    img = to_uint8(img)
    cnt_mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thick)
    return img


def fp_fn(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fp = np.logical_and(pred == 1, gt == 0).astype(np.uint8)
    fn = np.logical_and(pred == 0, gt == 1).astype(np.uint8)
    return fp, fn


def overlay_fp_fn(img: np.ndarray,
                  fp_mask: np.ndarray,
                  fn_mask: np.ndarray,
                  alpha: float = 0.45) -> np.ndarray:
    img = to_uint8(img).astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32).reshape(1, 1, 3)
    blue = np.array([0, 0, 255], dtype=np.float32).reshape(1, 1, 3)

    img = img * (1.0 - fp_mask[..., None] * alpha) + red * (fp_mask[..., None] * alpha)
    img = img * (1.0 - fn_mask[..., None] * alpha) + blue * (fn_mask[..., None] * alpha)
    return img.clip(0, 255).astype(np.uint8)


def visualize_case(img: np.ndarray,
                   gt_mask: np.ndarray,
                   pred_mask: np.ndarray,
                   title: str = "",
                   save_path: Path = None):
    """
    Builds a 4-panel figure: image, GT overlay, prediction overlay, FP/FN diff.
    """
    img_u8 = to_uint8(img)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)
    pred_bin = (pred_mask > 0.5).astype(np.uint8)

    fp_mask, fn_mask = fp_fn(gt_bin, pred_bin)

    gt_vis = draw_contours(overlay_mask(img_u8, gt_bin, color=(0, 255, 0), alpha=0.32),
                           gt_bin, color=(0, 255, 0), thick=2)
    pred_vis = draw_contours(overlay_mask(img_u8, pred_bin, color=(255, 0, 255), alpha=0.32),
                             pred_bin, color=(255, 0, 255), thick=2)
    diff_vis = overlay_fp_fn(img_u8, fp_mask, fn_mask)

    fig, ax = plt.subplots(1, 4, figsize=(18, 6))
    ax[0].imshow(img_u8)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(gt_vis)
    ax[1].set_title("Ground Truth (green)")
    ax[1].axis("off")

    ax[2].imshow(pred_vis)
    ax[2].set_title("Prediction (magenta)")
    ax[2].axis("off")

    ax[3].imshow(diff_vis)
    ax[3].set_title("FP=red | FN=blue")
    ax[3].axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=250)
    plt.close(fig)


def save_prediction_collage(model: nn.Module,
                            loader: DataLoader,
                            device: torch.device,
                            save_path: Path,
                            max_samples: int = 10,
                            thr_mask: float = 0.5,
                            use_tta: bool = True,
                            use_amp: bool = True,
                            tampered_only: bool = True,
                            channels_last: bool = False,
                            heavy_tta: bool = False):
    """
    Saves a collage of predictions in a publication-style layout:
    - Original image
    - Ground truth mask
    - Prediction mask
    - Error overlay (TP green / FP red / FN blue) with IoU/Dice
    Samples are sorted by IoU (descending) for clarity.
    """
    model.eval()
    samples = []

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)

            mask_batch = batch["mask"]
            has_mask_batch = batch["has_mask"]
            ids = batch["id"]
            imgs = batch["img_np"]

            if isinstance(ids, str):
                ids = [ids]

            if isinstance(imgs, torch.Tensor):
                imgs = imgs.cpu().numpy()
            elif isinstance(imgs, list):
                imgs = np.stack(imgs)
            else:
                imgs = np.asarray(imgs)

            if isinstance(mask_batch, torch.Tensor):
                mask_np = mask_batch.cpu().numpy()
            else:
                mask_np = np.asarray(mask_batch)

            if isinstance(has_mask_batch, torch.Tensor):
                has_mask_np = has_mask_batch.cpu().numpy()
            else:
                has_mask_np = np.asarray(has_mask_batch)

            batch_size = x.size(0)

            for b in range(batch_size):
                cls_logits, seg_map = tta_predict_3class_and_seg(
                    model,
                    x[b:b+1],
                    use_tta=use_tta,
                    use_amp=use_amp,
                    channels_last=channels_last,
                    heavy_tta=heavy_tta,
                )
                # Only keep tampered samples if requested
                if tampered_only:
                    pred_cls = int(np.argmax(cls_logits))
                    gt_cls = int(has_mask_np[b] > 0.5)
                    if (pred_cls != 1) and (gt_cls != 1):
                        continue

                seg_map = np.squeeze(seg_map)
                pred_bin = (seg_map >= thr_mask).astype(np.uint8)

                img_np = imgs[b]
                if isinstance(img_np, torch.Tensor):
                    img_np = img_np.cpu().numpy()
                if img_np.max() > 1.5:
                    img_np = img_np / 255.0

                if mask_np.ndim == 4:
                    gt_mask = mask_np[b, 0]
                else:
                    gt_mask = mask_np[b]
                if has_mask_np[b] < 0.5:
                    gt_mask = np.zeros_like(pred_bin)

                gt_mask = np.squeeze(gt_mask)

                img_u8 = to_uint8(img_np)
                gt_bin = (gt_mask > 0.5).astype(np.uint8)
                pred_bin_u8 = (pred_bin > 0.5).astype(np.uint8)
                fp_mask, fn_mask = fp_fn(gt_bin, pred_bin_u8)
                tp_mask = (gt_bin == 1) & (pred_bin_u8 == 1)

                inter = (gt_bin & pred_bin_u8).sum()
                union = ((gt_bin + pred_bin_u8) > 0).sum()
                iou = float(inter / (union + 1e-6)) if union > 0 else 0.0
                dice = float((2 * inter) / (gt_bin.sum() + pred_bin_u8.sum() + 1e-6)) if (gt_bin.sum() + pred_bin_u8.sum()) > 0 else 0.0

                sample_id = ids[b] if isinstance(ids, (list, tuple)) else ids

                samples.append({
                    "id": sample_id,
                    "img": img_u8,
                    "gt": gt_bin,
                    "pred": pred_bin_u8,
                    "fp": fp_mask,
                    "fn": fn_mask,
                    "tp": tp_mask,
                    "iou": iou,
                    "dice": dice,
                })

                if len(samples) >= max_samples:
                    break

            if len(samples) >= max_samples:
                break

    if len(samples) == 0:
        print("[collage] no samples to visualize")
        return

    # Sort by IoU (descending) to showcase best examples; cap at max_samples
    samples = sorted(samples, key=lambda s: s["iou"], reverse=True)[:max_samples]
    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, sample in enumerate(samples):
        img = sample["img"]
        gt = sample["gt"]
        pred = sample["pred"]
        fp_mask = sample["fp"]
        fn_mask = sample["fn"]
        tp_mask = sample["tp"]

        # Build overlay with TP/FP/FN colors
        overlay = img.copy().astype(np.float32)
        overlay[tp_mask.astype(bool)] = [34, 197, 94]    # emerald
        overlay[fp_mask.astype(bool)] = [239, 68, 68]    # coral red
        overlay[fn_mask.astype(bool)] = [59, 130, 246]   # vivid blue
        overlay = (0.6 * img + 0.4 * overlay).astype(np.uint8)

        axes[r,0].imshow(img)
        axes[r,0].set_title(f"{sample['id']}", fontsize=10)
        axes[r,0].axis("off")

        axes[r,1].imshow(gt, cmap="gray")
        axes[r,1].set_title("Ground Truth", fontsize=10)
        axes[r,1].axis("off")

        axes[r,2].imshow(pred, cmap="gray")
        axes[r,2].set_title("Prediction", fontsize=10)
        axes[r,2].axis("off")

        axes[r,3].imshow(overlay)
        axes[r,3].set_title(f"Error | IoU {sample['iou']:.3f} | Dice {sample['dice']:.3f}", fontsize=10)
        axes[r,3].axis("off")

    fig.tight_layout()
    # Add legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#22c55e', edgecolor='black', alpha=0.7, label='TP'),
        Patch(facecolor='#ef4444', edgecolor='black', alpha=0.7, label='FP'),
        Patch(facecolor='#3b82f6', edgecolor='black', alpha=0.7, label='FN'),
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.03), fontsize=10, frameon=False)

    fig.savefig(save_path, dpi=280, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ============================================================
#       DINO + ConvNeXt + MVSS Fusion Backbone (addon)
# ============================================================

class ForensicTripleBackbone(nn.Module):
    """
    DINOv3 + ConvNeXt + MVSS-Net fusion backbone.
    Produces multi-scale decoder inputs + global cls features.
    """

    def __init__(self, dino, convnext, mvss):
        super().__init__()
        self.dino = dino
        self.convnext = convnext
        self.mvss = mvss

        # unify dims
        self.proj4 = nn.Conv2d(256, 128, 1)
        self.proj8 = nn.Conv2d(256, 128, 1)
        self.proj16 = nn.Conv2d(256, 128, 1)
        self.proj32 = nn.Conv2d(256, 128, 1)

        self.fuse4 = nn.Conv2d(128+128, 128, 1)
        self.fuse8 = nn.Conv2d(128+128, 128, 1)
        self.fuse16 = nn.Conv2d(128+128, 128, 1)
        self.fuse32 = nn.Conv2d(128+128, 128, 1)

        self.cls_head = nn.Linear(1024+128+1, 3)  # DINO + pooled convnext + pooled noise/boundary

    def forward(self, x):
        # DINOv3 backbone
        dino_tokens = self.dino.get_intermediate_layers(x, n=1)[0]
        dino_global = dino_tokens[:, 0, :]   # CLS token, [B,1024]

        # ConvNeXt backbone
        c1, c2, c3, c4 = self.convnext.extract_features(x)  # H/4, H/8, H/16, H/32

        # MVSS-Net
        mv = self.mvss(x)
        p2, p3, p4, p5 = mv["p2"], mv["p3"], mv["p4"], mv["p5"]

        # unify channels to 128d
        p2 = self.proj4(p2)
        p3 = self.proj8(p3)
        p4 = self.proj16(p4)
        p5 = self.proj32(p5)

        # spatial fusion
        f4 = self.fuse4(torch.cat([c1, p2], dim=1))
        f8 = self.fuse8(torch.cat([c2, p3], dim=1))
        f16 = self.fuse16(torch.cat([c3, p4], dim=1))
        f32 = self.fuse32(torch.cat([c4, p5], dim=1))

        # classifier features
        convnext_global = c4.mean([2,3])
        mv_noise = (mv["noise"] * mv["boundary"]).mean([1,2,3]).unsqueeze(1)

        cls_feat = torch.cat([dino_global, convnext_global, mv_noise], dim=1)
        cls_raw = self.cls_head(cls_feat)

        return {
            "f4": f4, "f8": f8, "f16": f16, "f32": f32,
            "cls_raw": cls_raw,
            "mv_boundary": mv["boundary"],
            "mv_noise": mv["noise"]
        }


class CoralLayer(nn.Module):
    """
    Simple temperature-scaled softmax for calibration.
    """
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp, dtype=torch.float32)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp)
        return torch.softmax(logits / temp, dim=1)


def bayesian_fusion_triple(cls_probs: torch.Tensor,
                           mask_probs: torch.Tensor,
                           noise_map: torch.Tensor,
                           boundary_map: torch.Tensor,
                           w_cls: float = 1.0,
                           w_seg: float = 0.15,
                           w_noise: float = 0.10,
                           w_boundary: float = 0.10) -> torch.Tensor:
    """
    Fusion of classifier probs with segmentation + MVSS noise/boundary cues.
    """
    eps = 1e-6
    p = torch.clamp(cls_probs, eps, 1.0 - eps)
    base = torch.log(p)

    seg_mean = mask_probs.mean(dim=(1,2,3))
    noise_mean = noise_map.mean(dim=(1,2,3))
    boundary_mean = boundary_map.mean(dim=(1,2,3))

    seg_logits = torch.stack([
        -seg_mean,
         seg_mean * 2.0,
        -seg_mean * 0.4
    ], dim=1)

    noise_logits = torch.stack([
        -noise_mean * 0.5,
         noise_mean * 1.5,
        -noise_mean * 0.5
    ], dim=1)

    boundary_logits = torch.stack([
        -boundary_mean * 0.5,
         boundary_mean * 1.5,
        -boundary_mean * 0.5
    ], dim=1)

    fused = (
        w_cls * base +
        w_seg * seg_logits +
        w_noise * noise_logits +
        w_boundary * boundary_logits
    )
    return torch.softmax(fused, dim=1)


class TripleFusionModel(nn.Module):
    """
    Wrapper that uses ForensicTripleBackbone + SegFormerV2 decoder.
    """
    def __init__(self,
                 dino_model: nn.Module,
                 convnext_model: nn.Module,
                 mvss_model: nn.Module,
                 embed_dim: int = 256):
        super().__init__()
        self.backbone = ForensicTripleBackbone(dino_model, convnext_model, mvss_model)
        self.decoder = SegFormerV2Decoder(dims=(128, 128, 128, 128), embed_dim=embed_dim)
        self.coral = CoralLayer()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        out = self.backbone(x)

        seg_logits = self.decoder([out["f4"], out["f8"], out["f16"], out["f32"]])
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode="bilinear", align_corners=False)

        cls_raw = out["cls_raw"]
        cls_coral = self.coral(cls_raw)
        cls_fused = bayesian_fusion_triple(
            cls_coral,
            torch.sigmoid(seg_logits),
            out["mv_noise"],
            out["mv_boundary"]
        )

        # Return logits for training compatibility; fused probs for inference
        return cls_raw, seg_logits, seg_logits, cls_fused


# ============================================================
#                 MAIN  (O2-A RESTORED)
# ============================================================

def main():
    # ------------------------------------------
    # WSL PATHS – ORIGINAL
    # ------------------------------------------
    # [v2-joe-handoff] paths overridable by env; default to the WSL copy
    TRAIN_ROOT = os.environ.get("TRAIN_ROOT", str(Path.home() / "data/overhead/train_jpg/train"))
    TEST_ROOT  = os.environ.get("TEST_ROOT",  str(Path.home() / "data/overhead/test_jpg/test"))

    TRAIN_META = os.path.join(TRAIN_ROOT, "metadata.csv")
    TEST_META  = os.path.join(TEST_ROOT,  "metadata.csv")

    # ------------------------------------------
    # HYPERPARAMETERS
    # ------------------------------------------
    DO_TRAIN   = env_flag("DO_TRAIN", True)
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "50"))
    LR         = float(os.environ.get("LR_BACKBONE", "2e-6"))   # backbone
    LR_DECODER = float(os.environ.get("LR_DECODER", "1e-5"))    # [v2] decoder + classifier
    LAMBDA_SEG = 0.8
    IMG_SIZE   = 384
    BATCH_SIZE = 3
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
    USE_AMP    = True           # mixed precision on CUDA
    USE_TTA    = True           # enable TTA for stronger eval
    HEAVY_TTA  = env_flag("TTA_HEAVY", False)  # add rotations for eval TTA
    USE_CHANNELS_LAST = env_flag("CHANNELS_LAST", True)
    VAL_SPLIT  = float(os.environ.get("VAL_SPLIT", "0.1"))  # fraction of train for val; 0 disables
    VAL_SEED   = int(os.environ.get("VAL_SEED", "42"))
    BEST_METRIC = os.environ.get("BEST_METRIC", "f1")       # monitor metric for checkpointing
    PRELOAD_ALL = env_flag("PRELOAD_ALL", True)             # load all images into RAM
    FUSION_SEG_THR = float(os.environ.get("FUSION_SEG_THR", "0.10"))
    FREEZE_WARMUP_EPOCHS = int(os.environ.get("FREEZE_WARMUP_EPOCHS", 3))
    RUN_EVAL_ONLY = env_flag("RUN_EVAL_ONLY", False)
    eval_splits_env = os.environ.get("EVAL_SPLITS", "train,val,test")
    eval_splits = {s.strip().lower() for s in eval_splits_env.split(",") if s.strip()}

    # Default to gated ViT-L SAT checkpoint; override with DINOV3_CKPT env for other backbones.
    DINOV3_CKPT = os.environ.get("DINOV3_CKPT", DEFAULT_DINOV3_REPO)
    # [baseline] BACKBONE=resnet50 (any timm resnet name) swaps the encoder for an ImageNet ResNet
    BACKBONE = os.environ.get("BACKBONE", "dinov3")
    USE_RESNET = BACKBONE.lower().startswith("resnet")
    # [baseline] BACKBONE=radio  (or radio:<hf repo>) swaps the encoder for NVIDIA C-RADIO
    USE_RADIO = BACKBONE.lower().startswith("radio")
    RADIO_REPO = BACKBONE.split(":", 1)[1] if ":" in BACKBONE else "nvidia/C-RADIOv4-SO400M"
    if USE_RESNET:
        BACKBONE = BACKBONE.lower()
        DINOV3_CKPT = BACKBONE          # used for the results folder name and the metrics "checkpoint" field
    if USE_RADIO:
        DINOV3_CKPT = RADIO_REPO
    CKPT_LOWER = DINOV3_CKPT.lower()
    DINOV3_IS_FILE = Path(DINOV3_CKPT).is_file()
    USE_VIT = ("convnext" not in CKPT_LOWER) and not USE_RESNET and not USE_RADIO
    USE_COMPILE = env_flag("TORCH_COMPILE", False)

    # ------------------------------------------
    # [v2] SEED, ABLATION SWITCHES, RUN TAG
    # ------------------------------------------
    SEED            = int(os.environ.get("SEED", "42"))
    ABL_EDGE_GATE   = env_flag("ABL_EDGE_GATE", True)    # 0 = no edge gating
    ABL_BOUNDARY    = env_flag("ABL_BOUNDARY", True)     # 0 = no boundary loss term
    ABL_DEEP_SUP    = env_flag("ABL_DEEP_SUP", True)     # 0 = no deep supervision
    FREEZE_BACKBONE = env_flag("FREEZE_BACKBONE", False) # 1 = frozen-encoder baseline
    MASK_AUG        = env_flag("MASK_AUG", False)        # 1 = old mask-only augmentation
    LEGACY_TOKEN_SLICE = env_flag("LEGACY_TOKEN_SLICE", False)  # 1 = old token slicing

    W_BOUND   = 0.25 if ABL_BOUNDARY else 0.0
    DS_WEIGHT = 0.3 if ABL_DEEP_SUP else 0.0

    variant_parts = []
    if not ABL_EDGE_GATE:
        variant_parts.append("noedge")
    if not ABL_BOUNDARY:
        variant_parts.append("nobound")
    if not ABL_DEEP_SUP:
        variant_parts.append("nods")
    if FREEZE_BACKBONE:
        variant_parts.append("frozen")
    if MASK_AUG:
        variant_parts.append("maskaug")
    if LEGACY_TOKEN_SLICE:
        variant_parts.append("legacytokens")
    if USE_RESNET:
        variant_parts.insert(0, BACKBONE)
    if USE_RADIO:
        variant_parts.insert(0, "radio")
    VARIANT = "-".join(variant_parts) if variant_parts else "full"
    RUN_TAG = os.environ.get("RUN_TAG", f"{VARIANT}_s{SEED}")

    seed_everything(SEED)
    print(f"[init] variant={VARIANT} seed={SEED} run_tag={RUN_TAG}")

    # Structured results dir under ~/results_dinov3/<backbone>/<ckpt_name>
    safe_name = Path(DINOV3_CKPT).stem.replace("/", "_")
    OUTPUT_DIR = Path.home() / "results_dinov3" / ("resnet" if USE_RESNET else "radio" if USE_RADIO else "vit" if USE_VIT else "convnext") / safe_name / RUN_TAG
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[init] device:", device)

    # ------------------------------------------
    # MODEL (DINOv3 backbone)
    # ------------------------------------------
    if USE_VIT and DINOV3_IS_FILE:
        raise RuntimeError("Local .pth ViT checkpoints are not wired yet; please use a convnext .pth or an HF ViT repo id.")
    if USE_RESNET:
        model = ResNetSegFormer3C(
            arch=BACKBONE,
            embed_dim=256,
            num_classes=3,
            use_edge_gate=ABL_EDGE_GATE,
        ).to(device)
    elif USE_RADIO:
        model = RadioSegFormer3C(
            repo=RADIO_REPO,
            embed_dim=256,
            num_classes=3,
            use_edge_gate=ABL_EDGE_GATE,
        ).to(device)
    elif USE_VIT:
        model = DINOv3ViTSegFormer3C(
            dinov3_ckpt=DINOV3_CKPT,
            embed_dim=256,
            num_classes=3,
            use_edge_gate=ABL_EDGE_GATE,
            legacy_token_slice=LEGACY_TOKEN_SLICE,
        ).to(device)
    else:
        model = DINOv3SegFormerV2_3C(
            dinov3_ckpt=DINOV3_CKPT,
            embed_dim=256,
            num_classes=3,
        ).to(device)
    param_counts = count_params(model)  # [v2]
    print(f"[init] parameters: total={param_counts['total']/1e6:.2f}M "
          f"encoder={param_counts['encoder']/1e6:.2f}M heads={param_counts['heads']/1e6:.3f}M")
    if FREEZE_WARMUP_EPOCHS > 0 or FREEZE_BACKBONE:
        set_encoder_trainable(model, False)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            import torch._inductor.config as inductor_cfg
            inductor_cfg.compile_threads = int(os.environ.get("TORCH_COMPILE_THREADS", 1))
            if hasattr(inductor_cfg, "triton"):
                inductor_cfg.triton.concurrency = 1
        except Exception:
            pass
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] torch.compile failed, falling back to eager: {e}")

    # Optionally load checkpoint for eval-only or resume
    CKPT_LOAD = os.environ.get("CKPT_LOAD", None)
    auto_best = outdir / "best_checkpoint.pt"
    load_path = None
    if CKPT_LOAD:
        load_path = Path(CKPT_LOAD)
    elif (not DO_TRAIN) and auto_best.exists():
        load_path = auto_best

    if load_path is not None and load_path.exists():
        state = torch.load(load_path, map_location=device)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        sd = remap_state_dict_keys(sd, model)  # [v2-fix] transformers 4.x -> 5.x key names
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] loaded checkpoint from {load_path}, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing or unexpected:
            print("[load] missing:", missing[:8]); print("[load] unexpected:", unexpected[:8])
            raise RuntimeError("[load] checkpoint keys do not match the model; refusing to evaluate "
                               "partially-loaded weights")

    # ------------------------------------------
    # DATA
    # ------------------------------------------
    cache_limit_env = os.environ.get("CACHE_LIMIT", None)
    CACHE_LIMIT = int(cache_limit_env) if cache_limit_env is not None else None
    if PRELOAD_ALL:
        CACHE_LIMIT = None  # allow full preload

    val_dataset = None
    train_ids = None
    val_ids = None
    if VAL_SPLIT > 0.0:
        ids = []
        with open(TRAIN_META, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids.append(Path(row["image_filename"]).stem)
        rng = random.Random(VAL_SEED)
        rng.shuffle(ids)
        val_count = max(1, int(len(ids) * VAL_SPLIT))
        val_ids = set(ids[:val_count])
        train_ids = set(ids[val_count:])
        if len(train_ids) == 0:
            raise RuntimeError("VAL_SPLIT too high, no training samples left.")

    train_dataset = ForensicsDataset3C(
        TRAIN_ROOT, TRAIN_META,
        img_size=IMG_SIZE,
        is_train=True,
        cache_in_ram=True,
        cache_limit=CACHE_LIMIT,
        id_whitelist=train_ids,
        preload_all=PRELOAD_ALL,
        mask_aug=MASK_AUG,
    )

    test_dataset = ForensicsDataset3C(TEST_ROOT, TEST_META,
                                      img_size=IMG_SIZE,
                                      is_train=False,
                                      cache_in_ram=True,
                                      cache_limit=CACHE_LIMIT,
                                      preload_all=PRELOAD_ALL)

    if val_ids is not None:
        val_dataset = ForensicsDataset3C(
            TRAIN_ROOT, TRAIN_META,
            img_size=IMG_SIZE,
            is_train=False,
            cache_in_ram=True,
            cache_limit=min(CACHE_LIMIT, 2000) if CACHE_LIMIT is not None else None,
            id_whitelist=val_ids,
            preload_all=PRELOAD_ALL,
        )

    loader_opts = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    if NUM_WORKERS > 0:
        loader_opts.update(prefetch_factor=2, persistent_workers=True, pin_memory=True,
                           worker_init_fn=_seed_worker)
    data_gen = torch.Generator()   # [v2] reproducible sampling / worker seeds
    data_gen.manual_seed(SEED)

    # Class-balanced sampling for training (helps rare fake/tampered)
    train_labels = [item["label"] for item in train_dataset.items]
    class_counts = np.bincount(train_labels, minlength=3)
    class_weights_sampler = 1.0 / np.clip(class_counts, 1, None)
    # tilt a bit more toward tampered/fake
    class_weights_sampler[1] *= 1.1
    class_weights_sampler[2] *= 1.3
    sample_weights = torch.as_tensor([class_weights_sampler[l] for l in train_labels],
                                     dtype=torch.double)
    train_sampler = WeightedRandomSampler(sample_weights,
                                          num_samples=len(sample_weights),
                                          replacement=True,
                                          generator=data_gen)

    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              shuffle=False,
                              generator=data_gen,
                              **loader_opts)

    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             **loader_opts)

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset,
                                shuffle=False,
                                **loader_opts)

    # ------------------------------------------
    # TRAIN
    # ------------------------------------------
    train_logs = None
    if DO_TRAIN and not RUN_EVAL_ONLY:
        print(f"[train] starting training for {NUM_EPOCHS} epochs")
        train_logs = train_3class(model,
                                  train_loader,
                                  device,
                                  epochs=NUM_EPOCHS,
                                  lr=LR,
                                  lambda_seg=LAMBDA_SEG,
                                  use_amp=USE_AMP,
                                  outdir=outdir,
                                  eval_loader=val_loader,
                                  channels_last=USE_CHANNELS_LAST,
                                  save_best=True,
                                  best_metric=BEST_METRIC,
                                  freeze_warmup_epochs=FREEZE_WARMUP_EPOCHS,
                                  lr_decoder=LR_DECODER,
                                  w_bound=W_BOUND,
                                  ds_weight=DS_WEIGHT,
                                  freeze_backbone=FREEZE_BACKBONE)

    # ------------------------------------------
    # EVALUATE — TRAIN
    # ------------------------------------------
    train_results = None
    val_results = None
    test_results = None

    if "train" in eval_splits:
        print("\n[eval] TRAIN set")
        train_results = evaluate_3class(model, train_loader, device,
                                        use_tta=USE_TTA,
                                        use_amp=USE_AMP,
                                        channels_last=USE_CHANNELS_LAST,
                                        fusion_seg_thr=FUSION_SEG_THR,
                                        heavy_tta=HEAVY_TTA)

        plot_confusion_matrix(train_results["cm_raw"], ["pristine","tampered","fake"],
                              outdir / "train_cm_raw.png", "Train CM Raw")

        plot_confusion_matrix(train_results["cm_fused"], ["pristine","tampered","fake"],
                              outdir / "train_cm_fused.png", "Train CM Fused")

        plot_roc_macro(train_results["y_true"],
                       train_results["probs_raw"],
                       outdir / "train_roc_raw.png",
                       "Train ROC Raw")

        plot_roc_macro(train_results["y_true"],
                       train_results["probs_fused"],
                       outdir / "train_roc_fused.png",
                       "Train ROC Fused")

        plot_pr_macro(train_results["y_true"],
                      train_results["probs_raw"],
                      outdir / "train_pr_raw.png",
                      "Train PR Raw")

        plot_pr_macro(train_results["y_true"],
                      train_results["probs_fused"],
                      outdir / "train_pr_fused.png",
                      "Train PR Fused")

        plot_iou_hist(train_results["ious"], outdir / "train_iou_hist.png")

    if val_loader is not None and ("val" in eval_splits):
        print("\n[eval] VAL set")
        val_results = evaluate_3class(model, val_loader, device,
                                      use_tta=USE_TTA,
                                      use_amp=USE_AMP,
                                      channels_last=USE_CHANNELS_LAST,
                                      fusion_seg_thr=FUSION_SEG_THR,
                                      heavy_tta=HEAVY_TTA)

        plot_confusion_matrix(val_results["cm_raw"], ["pristine","tampered","fake"],
                              outdir / "val_cm_raw.png", "Val CM Raw")

        plot_confusion_matrix(val_results["cm_fused"], ["pristine","tampered","fake"],
                              outdir / "val_cm_fused.png", "Val CM Fused")

        plot_roc_macro(val_results["y_true"],
                       val_results["probs_raw"],
                       outdir / "val_roc_raw.png",
                       "Val ROC Raw")

        plot_roc_macro(val_results["y_true"],
                       val_results["probs_fused"],
                       outdir / "val_roc_fused.png",
                       "Val ROC Fused")

        plot_pr_macro(val_results["y_true"],
                      val_results["probs_raw"],
                      outdir / "val_pr_raw.png",
                      "Val PR Raw")

        plot_pr_macro(val_results["y_true"],
                      val_results["probs_fused"],
                      outdir / "val_pr_fused.png",
                      "Val PR Fused")

        plot_iou_hist(val_results["ious"], outdir / "val_iou_hist.png")


    # ------------------------------------------
    # EVALUATE — TEST
    # ------------------------------------------
    if "test" in eval_splits:
        print("\n[eval] TEST set")
        test_results = evaluate_3class(model, test_loader, device,
                                       use_tta=USE_TTA,
                                       use_amp=USE_AMP,
                                       channels_last=USE_CHANNELS_LAST,
                                       fusion_seg_thr=FUSION_SEG_THR,
                                       heavy_tta=HEAVY_TTA)

        plot_confusion_matrix(test_results["cm_raw"], ["pristine","tampered","fake"],
                              outdir / "test_cm_raw.png", "Test CM Raw")

        plot_confusion_matrix(test_results["cm_fused"], ["pristine","tampered","fake"],
                              outdir / "test_cm_fused.png", "Test CM Fused")

        plot_roc_macro(test_results["y_true"],
                       test_results["probs_raw"],
                       outdir / "test_roc_raw.png",
                       "Test ROC Raw")

        plot_roc_macro(test_results["y_true"],
                       test_results["probs_fused"],
                       outdir / "test_roc_fused.png",
                       "Test ROC Fused")

        plot_pr_macro(test_results["y_true"],
                      test_results["probs_raw"],
                      outdir / "test_pr_raw.png",
                      "Test PR Raw")

        plot_pr_macro(test_results["y_true"],
                      test_results["probs_fused"],
                      outdir / "test_pr_fused.png",
                      "Test PR Fused")

        plot_iou_hist(test_results["ious"], outdir / "test_iou_hist.png")

    seg_conf = seg_ious = seg_stats = None
    if test_results is not None:
        print("[eval] computing segmentation confusion/IoU (may take a few minutes)...")
        try:
            seg_conf, seg_ious, seg_stats = compute_segmentation_confusion_and_iou(
                model,
                test_loader,
                device,
                thr_mask=0.5,
                use_tta=USE_TTA,
                use_amp=USE_AMP,
                channels_last=USE_CHANNELS_LAST,
                heavy_tta=HEAVY_TTA,
            )
            plot_segmentation_summary(
                seg_conf,
                seg_ious,
                outdir / "test_segmentation_summary.png",
                epoch=NUM_EPOCHS if DO_TRAIN else None
            )
        except Exception as e:
            print(f"[warn] failed to generate segmentation summary: {e}")

        print("[eval] building test collage...")
        save_prediction_collage(
            model,
            test_loader,
            device,
            outdir / "test_collage.png",
            max_samples=10,
            thr_mask=0.5,
            use_tta=USE_TTA,
            use_amp=USE_AMP,
            tampered_only=True,
            channels_last=USE_CHANNELS_LAST,
            heavy_tta=HEAVY_TTA,
        )

    # ------------------------------------------
    # AREA BUCKET METRICS
    # ------------------------------------------
    AREA_THR = 1.0 / 16.0

    # ------------------------------------------
    # [v2] SAVE TEST METRICS + RUN CONFIG AS JSON
    # ------------------------------------------
    if test_results is not None:
        from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                                     average_precision_score, balanced_accuracy_score)
        names = ["pristine", "tampered", "fake"]
        yt = np.asarray(test_results["y_true"])
        pr = np.asarray(test_results["probs_raw"])
        yp = pr.argmax(1)
        p_c, r_c, f_c, n_c = precision_recall_fscore_support(
            yt, yp, labels=[0, 1, 2], zero_division=0)
        per_class = {}
        for k, nm in enumerate(names):
            y_bin = (yt == k).astype(int)
            try:
                ap_k = float(average_precision_score(y_bin, pr[:, k]))
                auc_k = float(roc_auc_score(y_bin, pr[:, k]))
            except Exception:
                ap_k = auc_k = float("nan")
            per_class[nm] = {
                "precision": float(p_c[k]), "recall": float(r_c[k]), "f1": float(f_c[k]),
                "ap": ap_k, "auc": auc_k, "support": int(n_c[k]),
            }

        def _mean(v):
            return float(np.mean(v)) if v is not None and len(v) > 0 else float("nan")

        metrics = {
            "run_tag": RUN_TAG,
            "variant": VARIANT,
            "seed": SEED,
            "checkpoint": DINOV3_CKPT,
            "config": {
                "edge_gate": ABL_EDGE_GATE, "boundary_loss": ABL_BOUNDARY,
                "deep_supervision": ABL_DEEP_SUP, "freeze_backbone": FREEZE_BACKBONE,
                "mask_aug": MASK_AUG, "legacy_token_slice": LEGACY_TOKEN_SLICE,
                "lr_backbone": LR, "lr_decoder": LR_DECODER, "lambda_seg": LAMBDA_SEG,
                "w_bound": W_BOUND, "ds_weight": DS_WEIGHT, "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS, "val_split": VAL_SPLIT,
                "val_seed": VAL_SEED, "best_metric": BEST_METRIC,
                "freeze_warmup_epochs": FREEZE_WARMUP_EPOCHS, "tta": USE_TTA, "amp_dtype": _AMP_NAME,
                "heavy_tta": HEAVY_TTA, "trained_in_this_run": bool(train_logs is not None),
            },
            "n_test": int(len(yt)),
            "accuracy": float((yp == yt).mean()),
            "macro_f1": float(np.mean(f_c)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "per_class": per_class,
            "confusion_matrix_rows_true_cols_pred": np.asarray(test_results["cm_raw"]).tolist(),
            "localization_tampered": {
                "n_images": int(len(test_results["ious"])),
                "mean_iou": _mean(test_results["ious"]),
                "mean_iou_second_pass": _mean(seg_ious),
                "mean_dice": _mean(seg_stats["dice"]) if seg_stats else float("nan"),
                "mean_mcc": _mean(seg_stats["mcc"]) if seg_stats else float("nan"),
            },
            "pixel_confusion_full_test_set": (
                {"tn": int(seg_conf[0, 0]), "fp": int(seg_conf[0, 1]),
                 "fn": int(seg_conf[1, 0]), "tp": int(seg_conf[1, 1])}
                if seg_conf is not None else None),
            "area_buckets": bucket_metrics_by_area(test_results, area_thr=AREA_THR),
            "parameters": param_counts,
            "ms_per_test_image": float(test_results.get("ms_per_image", float("nan"))),
            "train_sec_per_epoch": (_mean(train_logs["epoch_sec"]) if train_logs else float("nan")),
            "best_epoch": (train_logs.get("best_epoch") if train_logs else None),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        }
        with open(outdir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[save] wrote {outdir / 'test_metrics.json'}")
    if train_results is not None:
        print_area_bucket_metrics("TRAIN", bucket_metrics_by_area(train_results, area_thr=AREA_THR))
    if val_loader is not None and val_results is not None:
        print_area_bucket_metrics("VAL", bucket_metrics_by_area(val_results, area_thr=AREA_THR))
    if test_results is not None:
        print_area_bucket_metrics("TEST", bucket_metrics_by_area(test_results, area_thr=AREA_THR))

    # ------------------------------------------
    # SUMMARY PRINT
    # ------------------------------------------
    print("\n=== FINAL O2-A SUMMARY ===")
    if train_results is not None:
        print(f"Train acc_raw   = {train_results['acc_raw']:.4f}")
        print(f"Train acc_fused = {train_results['acc_fused']:.4f}")
        print(f"Train F1_raw    = {train_results['f1_raw']:.4f}")
        print(f"Train F1_fused  = {train_results['f1_fused']:.4f}")
        print(f"Train IoU (tampered) mean = {np.mean(train_results['ious']):.4f}")
    else:
        print("Train metrics skipped")

    if val_loader is not None and val_results is not None:
        print("\n--- VAL ---")
        print(f"Val acc_raw   = {val_results['acc_raw']:.4f}")
        print(f"Val acc_fused = {val_results['acc_fused']:.4f}")
        print(f"Val F1_raw    = {val_results['f1_raw']:.4f}")
        print(f"Val F1_fused  = {val_results['f1_fused']:.4f}")
        print(f"Val IoU (tampered) mean = {np.mean(val_results['ious']):.4f}")
    elif val_loader is not None:
        print("\n--- VAL ---\nVal metrics skipped")

    if test_results is not None:
        print("\n--- TEST ---")
        print(f"Test acc_raw   = {test_results['acc_raw']:.4f}")
        print(f"Test acc_fused = {test_results['acc_fused']:.4f}")
        print(f"Test F1_raw    = {test_results['f1_raw']:.4f}")
        print(f"Test F1_fused  = {test_results['f1_fused']:.4f}")
        print(f"Test IoU (tampered) mean = {np.mean(test_results['ious']):.4f}")
    else:
        print("\n--- TEST ---\nTest metrics skipped")


# ============================================================
#                ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
