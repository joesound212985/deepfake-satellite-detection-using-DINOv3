"""Count forward-pass FLOPs per image (384x384, batch 1, single view, eval mode) for each
model variant, on CPU, with torch.utils.flop_counter.FlopCounterMode.
This torch version counts a multiply-add as 2 FLOPs (checked: Linear(1000,1000) on one sample
= 2,000,000) and does not count scaled_dot_product_attention, so a handler for it is added
below. Both conventions are written: gflops (2 per multiply-add) and gmacs (1 per multiply-add,
the fvcore/ptflops figure most papers quote). Normalisation, activations, interpolation and
element-wise ops count as zero.
Read-only with respect to training: builds fresh models, touches no results folders."""
import json, sys, torch
from torch.utils.flop_counter import FlopCounterMode
import dinov3_forensics_train_v2 as M

def _sdpa(q, k, v, *a, out_shape=None, **kw):
    # QK^T: prod(q) * M ; attn @ V: prod(q[:-1]) * M * v[-1] ; 2 FLOPs per multiply-add
    import math
    M_ = k[-2]
    return 2 * (math.prod(q) * M_ + math.prod(q[:-1]) * M_ * v[-1])
CM = {torch.ops.aten.scaled_dot_product_attention: _sdpa}

WEB = sys.argv[1] if len(sys.argv) > 1 else "facebook/dinov3-vitl16-pretrain-lvd1689m"
V = {
 "full / frozen (DINOv3-sat ViT-L, edge gate)": lambda: M.DINOv3ViTSegFormer3C(dinov3_ckpt=M.DEFAULT_DINOV3_REPO, embed_dim=256, num_classes=3, use_edge_gate=True, legacy_token_slice=False),
 "plain (DINOv3-sat ViT-L, no edge gate)":      lambda: M.DINOv3ViTSegFormer3C(dinov3_ckpt=M.DEFAULT_DINOV3_REPO, embed_dim=256, num_classes=3, use_edge_gate=False, legacy_token_slice=False),
 "web (DINOv3 web ViT-L)":                      lambda: M.DINOv3ViTSegFormer3C(dinov3_ckpt=WEB, embed_dim=256, num_classes=3, use_edge_gate=True, legacy_token_slice=False),
 "resnet50":                                    lambda: M.ResNetSegFormer3C(arch="resnet50", embed_dim=256, num_classes=3, use_edge_gate=True),
 "radio (C-RADIOv4-SO400M)":                    lambda: M.RadioSegFormer3C(repo="nvidia/C-RADIOv4-SO400M", embed_dim=256, num_classes=3, use_edge_gate=True),
}
out = {}
x = torch.rand(1, 3, 384, 384)
for name, build in V.items():
    try:
        m = build().eval()
        with torch.no_grad(), FlopCounterMode(display=False, custom_mapping=CM) as fc:
            m(x)
        out[name] = {"gflops": fc.get_total_flops() / 1e9, "gmacs": fc.get_total_flops() / 2e9, "params_M": sum(p.numel() for p in m.parameters()) / 1e6}
        print(f"{name}: {out[name]['gflops']:.2f} GFLOPs = {out[name]['gmacs']:.2f} GMACs, {out[name]['params_M']:.2f} M params", flush=True)
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
json.dump({"input": "1x3x384x384", "convention": "gflops: 2 per multiply-add (torch FlopCounterMode + sdpa handler); gmacs = gflops/2; uncounted ops = 0", "web_ckpt": WEB, "results": out}, open("flops.json", "w"), indent=2)

# per-top-level-module breakdown for the full model
m = V["full / frozen (DINOv3-sat ViT-L, edge gate)"]().eval()
with torch.no_grad(), FlopCounterMode(m, depth=1, display=False, custom_mapping=CM) as fc:
    m(x)
bd = {k: sum(v.values()) / 1e9 for k, v in fc.get_flop_counts().items() if k.count(".") <= 1}
print("breakdown (GFLOPs):", json.dumps(bd, indent=1))
d = json.load(open("flops.json")); d["full_breakdown_gflops"] = bd; json.dump(d, open("flops.json", "w"), indent=2)
