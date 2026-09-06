#!/usr/bin/env python3
"""
asd_fer_zoo.py
==============
Model registry and training regimes to explore on the ASD facial-emotion corpus.

THE GOVERNING FACT
------------------
You have ~1,700 development images from a small number of children. At that scale
ARCHITECTURE IS NOT THE BOTTLENECK -- your own benchmark already shows ten backbones
landing inside each other's confidence intervals. Two things move the number, and
neither of them is "try an eleventh backbone":

  1. WHAT THE WEIGHTS ALREADY KNOW.  ImageNet pretraining knows objects, not faces and
     not expressions. Initialising from a face- or expression-pretrained model, or from
     a self-supervised model with strong facial part correspondence, changes the result
     far more than swapping ResNet for Swin.
  2. HOW MUCH OF THE MODEL YOU ARE ALLOWED TO MOVE.  Full fine-tuning of an 87M-parameter
     ViT on 1,700 images is an invitation to memorise. Frozen features with a small head,
     or LoRA on the attention projections, usually wins at this scale -- and when it does,
     that IS a result worth reporting at a clinical venue.

This file gives you both axes: a tiered model registry (`MODEL_ZOO`) and four training
regimes (`build_model(..., mode=...)`) that plug into asd_fer_baseline.py unchanged.

USAGE
-----
    # list what is available and how big it is
    python asd_fer_zoo.py --list
    python asd_fer_zoo.py --list --tier foundation_frozen

    # from asd_fer_baseline.py, swap the model factory:
    #   from asd_fer_zoo import build_model
    #   model = build_model(name, n_classes=6, mode="lora")

    # sanity-check that a model and mode actually construct
    python asd_fer_zoo.py --check vit_base_patch16_dinov3.lvd1689m --mode lora
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ======================================================================================
# The registry
# ======================================================================================
@dataclass
class Entry:
    name: str                 # timm identifier, or a marker for an external checkpoint
    tier: str
    why: str
    img_size: int = 224
    source: str = "timm"      # timm | external | features
    note: str = ""


MODEL_ZOO: list[Entry] = [

    # ---------------------------------------------------------------------------------
    # TIER 1 -- CNN/Transformer hybrids you MUST include as baselines
    #
    # Your stated plan is to design a hybrid CNN-Transformer. The first thing a reviewer
    # will ask is why an off-the-shelf hybrid was not enough. If CoAtNet, MaxViT and
    # CAFormer are not in your table, that question ends the review. These are not
    # "extra backbones", they are the control condition for your contribution.
    # ---------------------------------------------------------------------------------
    Entry("coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "hybrid_control",
          "Conv stages then relative-attention stages. The canonical CNN-Transformer "
          "hybrid; the direct competitor to whatever you build."),
    Entry("maxvit_tiny_tf_224.in1k", "hybrid_control",
          "Block-local plus grid-global attention interleaved with MBConv. A different "
          "hybridisation strategy from CoAtNet, so it tests the idea rather than one design."),
    Entry("caformer_s36.sail_in22k_ft_in1k", "hybrid_control",
          "MetaFormer: convolutional token mixing in early stages, self-attention late. "
          "Strong at small scale and cheap to train."),

    # ---------------------------------------------------------------------------------
    # TIER 2 -- Self-supervised foundation features, used FROZEN
    #
    # At n=1,700 a frozen encoder with a small trained head is a serious contender, not a
    # weak baseline. DINOv3 in particular carries region-level facial correspondence
    # without any face-specific training, and its INTERMEDIATE blocks carry it better
    # than its final block -- which is why `probe_layers` below pulls from mid-depth.
    # ---------------------------------------------------------------------------------
    Entry("vit_base_patch16_dinov3.lvd1689m", "foundation_frozen",
          "Best facial part correspondence of any general-purpose encoder. Use frozen, "
          "probing intermediate blocks. This is the highest expected-value single "
          "experiment on the list.", img_size=224),
    Entry("vit_large_patch16_dinov3.lvd1689m", "foundation_frozen",
          "Larger DINOv3. Frozen only -- do not attempt to fine-tune 300M parameters on "
          "1,700 images.", img_size=224),
    Entry("convnext_base.dinov3_lvd1689m", "foundation_frozen",
          "DINOv3 distilled into a ConvNeXt. Cheaper, and gives you a convolutional "
          "counterpart under identical pretraining -- a clean architectural ablation."),
    Entry("vit_base_patch14_dinov2.lvd142m", "foundation_frozen",
          "DINOv2. Include as the previous generation so the comparison is honest.",
          img_size=518),
    Entry("eva02_base_patch14_224.mim_in22k", "foundation_frozen",
          "Masked image modelling with CLIP-derived targets. Different SSL objective "
          "from DINO, so it probes whether the objective or the scale is what helps."),
    Entry("convnextv2_base.fcmae", "foundation_frozen",
          "Fully-convolutional MAE. The convolutional MIM comparison point."),
    Entry("vit_base_patch16_siglip_224.webli", "foundation_frozen",
          "Language-supervised features. Expect these to UNDERPERFORM on facial anatomy "
          "-- CLIP-style encoders localise faces but discriminate facial parts poorly. "
          "Worth one run precisely because it is the negative control."),

    # ---------------------------------------------------------------------------------
    # TIER 3 -- Expression-specialised models and FER-domain pretraining
    #
    # Not on timm; you fetch checkpoints from the authors. This is where the biggest
    # single gain probably lives, because the weights already encode expression.
    # ---------------------------------------------------------------------------------
    Entry("POSTER++", "fer_specialist", source="external",
          why="Cross-fusion of landmark and image streams; the reference FER architecture "
              "of the last few years, published in Pattern Recognition.",
          note="github.com/talented-q/poster_v2 -- RAF-DB and AffectNet checkpoints"),
    Entry("DAN", "fer_specialist", source="external",
          why="Distract-your-Attention Network. In a 2026 cross-scenario robustness "
              "study, DAN trained on AffectNet gave the strongest generalisation of any "
              "system tested, specialised or general-purpose.",
          note="github.com/yaoing/DAN -- use the AffectNet checkpoint, not RAF-DB"),
    Entry("APViT", "fer_specialist", source="external",
          why="Attentive-pooling ViT for FER; patch selection suits small datasets.",
          note="github.com/youqingxiaozhua/APViT"),
    Entry("EmoNet", "fer_specialist", source="external",
          why="Predicts valence/arousal as well as categories. The continuous outputs are "
              "a useful auxiliary signal when categorical labels are scarce and noisy.",
          note="github.com/face-analysis/emonet"),

    # ---------------------------------------------------------------------------------
    # TIER 4 -- Face-identity and face-SSL encoders
    #
    # Trained on faces at scale. Strong features for anything face-shaped, and the
    # embedding you already use for identity grouping in build_manifest.py.
    # ---------------------------------------------------------------------------------
    Entry("ArcFace-R100", "face_encoder", source="external",
          why="Face-recognition embedding. Encodes identity strongly -- which is exactly "
              "why it is a DOUBLE-EDGED baseline: if it does well on your data, that is "
              "evidence of subject leakage, not of expression modelling. Run it as a "
              "leakage probe.",
          note="insightface model zoo"),
    Entry("FaRL", "face_encoder", source="external",
          why="Vision-language pretraining on 20M face-text pairs. Face-domain "
              "equivalent of CLIP.",
          note="github.com/FacePerceiver/FaRL"),
    Entry("FSFM", "face_encoder", source="external",
          why="Self-supervised facial representation trained for face security tasks; "
              "generalises well off-distribution.",
          note="arXiv 2412.12032"),

    # ---------------------------------------------------------------------------------
    # TIER 5 -- Non-deep baselines. Do not skip these.
    #
    # If 30 action-unit intensities plus a gradient-boosted tree land inside the
    # confidence interval of an 87M-parameter transformer, that is the most interesting
    # result in your paper and the one a clinical readership will actually cite.
    # ---------------------------------------------------------------------------------
    Entry("openface_au+gbm", "interpretable", source="features",
          why="OpenFace 2.0 or Py-Feat action-unit intensities into LightGBM. ~30 "
              "features, interpretable, directly connected to the ASD facial "
              "phenomenology literature, trains in seconds."),
    Entry("landmarks+asymmetry+gbm", "interpretable", source="features",
          why="68 landmarks plus explicit left/right hemiface asymmetry statistics. This "
              "is your architectural hypothesis stated as a feature set -- if it works, "
              "it is direct evidence for the hemiface design before you build it."),
]

TIERS = {
    "hybrid_control":    "CNN-Transformer hybrids — the control your contribution must beat",
    "foundation_frozen": "Self-supervised foundation encoders, used frozen",
    "fer_specialist":    "Expression-pretrained models (external checkpoints)",
    "face_encoder":      "Face-identity / face-SSL encoders (external checkpoints)",
    "interpretable":     "Action-unit and landmark baselines",
}


# ======================================================================================
# LoRA — low-rank adaptation of attention projections
# ======================================================================================
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable rank-r update: W + (alpha/r)·B·A.

    Why this matters here: full fine-tuning moves ~87M parameters using ~1,700 images.
    LoRA at r=8 on the attention projections moves under 1% of that while keeping the
    pretrained representation intact. On small clinical datasets it routinely matches or
    beats full fine-tuning, and it makes seed-to-seed variance much smaller -- which,
    given how noisy your current comparisons are, is worth something on its own.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)      # starts as an exact no-op

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


class LoRAConv1x1(nn.Module):
    """Same idea for 1x1 convolutions.

    Needed because several hybrids -- CoAtNet's relative-position variants among them --
    implement attention q/k/v and output projections as 1x1 Conv2d rather than Linear.
    A Linear-only LoRA silently finds nothing to adapt in those models.
    """

    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_channels, 1, 1))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, r, 1, 1))
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        import torch.nn.functional as F
        delta = F.conv2d(F.conv2d(self.drop(x), self.lora_A), self.lora_B)
        return self.base(x) + delta * self.scaling


def inject_lora(model: nn.Module, r=8, alpha=16, dropout=0.05,
                targets=("qkv", "q_proj", "k_proj", "v_proj", "attn.proj",
                         "attn_block", "attn_grid")) -> int:
    """Wrap matching attention projections with a LoRA adapter. Returns how many.

    Only attention projections are targeted -- adapting the MLP blocks as well roughly
    doubles the trainable count for little gain at this data scale, and adapting the
    positional-bias MLPs (CoAtNet's `rel_pos.mlp.fc*`) adapts geometry rather than
    features, which is not what you want.
    """
    n = 0
    for mod_name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if "rel_pos" in full or not any(t in full for t in targets):
                continue
            if isinstance(child, nn.Linear):
                setattr(mod, child_name, LoRALinear(child, r, alpha, dropout)); n += 1
            elif isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):
                setattr(mod, child_name, LoRAConv1x1(child, r, alpha, dropout)); n += 1
    return n


# ======================================================================================
# Frozen multi-layer probe
# ======================================================================================
class MultiLayerProbe(nn.Module):
    """Frozen encoder + trainable head over features pooled from SEVERAL depths.

    The layer choice is not arbitrary. For DINOv3 ViT-L/16, region-level facial
    correspondence peaks around block 18 of 24 and DEGRADES by the final block, because
    late features are globally mixed. Probing only the last layer -- what a standard
    linear probe does -- throws away the part of the representation that knows a brow
    from a cheek. Defaults below pull from roughly 60%, 75% and 100% of depth.
    """

    def __init__(self, encoder: nn.Module, n_classes: int, feat_dims: list[int],
                 hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        d = sum(feat_dims)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def train(self, mode=True):        # encoder stays in eval: no BN/dropout drift
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)          # list of (B, N, C) or (B, C, H, W)
        pooled = []
        for f in feats:
            if f.dim() == 4:                 # conv feature map
                pooled.append(f.mean(dim=(2, 3)))
            elif f.dim() == 3:               # tokens: mean over patch tokens
                pooled.append(f.mean(dim=1))
            else:
                pooled.append(f)
        return self.head(torch.cat(pooled, dim=1).float())


def _probe_indices(name: str) -> tuple[list[int], bool]:
    """Choose which depths to probe, correctly for the two families.

    Plain transformers (ViT, DeiT, EVA) have many uniform blocks, so we take roughly
    60%, 75% and 100% of DEPTH -- bracketing the mid-network region where facial part
    correspondence is strongest, since the final block is globally mixed and worse at it.

    Hierarchical models (ConvNeXt, Swin, CAFormer, MaxViT) expose four STAGES with
    different channel widths; there the meaningful choice is the last few stages.

    Getting this wrong is not a small error. Asking timm for `features_only` and then
    indexing by the length of its default output truncates a 12-block ViT to its first
    three blocks -- a 22M-parameter stump probing early edge filters, which is the exact
    opposite of what you want.
    """
    import timm
    skel = timm.create_model(name, pretrained=False, num_classes=0)
    n_blocks = len(getattr(skel, "blocks", []) or [])
    del skel
    if n_blocks >= 8:
        return sorted({max(0, int(n_blocks * f) - 1)
                       for f in (0.60, 0.75, 1.00)}), True
    probe = timm.create_model(name, pretrained=False, features_only=True)
    n_stage = len(probe.feature_info.channels())
    del probe
    return list(range(max(0, n_stage - 3), n_stage)), False


# ======================================================================================
# Factory
# ======================================================================================
def build_model(name: str, n_classes: int = 6, mode: str = "full",
                pretrained: bool = True, lora_r: int = 8, drop_path: float = 0.1,
                probe_hidden: int = 512):
    """Build a model in one of four training regimes.

    mode:
      "full"     -- fine-tune everything (what you do now; the control)
      "lora"     -- freeze the backbone, train LoRA adapters + the head
      "probe"    -- freeze the backbone, train a head on pooled multi-depth features
      "linear"   -- freeze the backbone, train a single linear layer on final features

    Report all four for at least your top two backbones. The comparison is a genuine
    contribution at a small-clinical-data venue, and it costs you almost nothing because
    the frozen modes train in minutes.
    """
    import timm

    if mode in ("probe", "linear"):
        if mode == "linear":
            enc = timm.create_model(name, pretrained=pretrained, num_classes=0)
            dim = enc.num_features
            for p in enc.parameters():
                p.requires_grad_(False)
            enc.eval()

            class LinearProbe(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder, self.head = enc, nn.Linear(dim, n_classes)

                def train(self, m=True):
                    super().train(m); self.encoder.eval(); return self

                def forward(self, x):
                    with torch.no_grad():
                        f = self.encoder(x)
                    return self.head(f.float())
            return LinearProbe()

        # multi-depth probe via timm's features_only / intermediate-layer API
        keep, is_plain_vit = _probe_indices(name)
        try:
            enc = timm.create_model(name, pretrained=pretrained, features_only=True,
                                    out_indices=tuple(keep))
            dims = enc.feature_info.channels()
        except (RuntimeError, AssertionError, KeyError, IndexError, TypeError):
            # models without features_only support: use forward_intermediates
            base = timm.create_model(name, pretrained=pretrained, num_classes=0)

            class Inter(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.m = base

                def forward(self, x):
                    return list(self.m.forward_intermediates(
                        x, indices=keep, norm=True, intermediates_only=True))
            enc = Inter()
            dims = [base.num_features] * len(keep)
        return MultiLayerProbe(enc, n_classes, dims, hidden=probe_hidden)

    # full / lora
    kw = {}
    if mode == "full":
        kw["drop_path_rate"] = drop_path
    try:
        model = timm.create_model(name, pretrained=pretrained, num_classes=n_classes, **kw)
    except TypeError:
        model = timm.create_model(name, pretrained=pretrained, num_classes=n_classes)

    if mode == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        n = inject_lora(model, r=lora_r)
        if n == 0:
            raise ValueError(
                f"no LoRA target layers matched in {name}. LoRA here adapts attention "
                "projections (Linear or 1x1 Conv), so it fits transformers and hybrids "
                "but not pure CNNs such as ResNet, DenseNet, VGG or ConvNeXt -- for "
                "those use mode='probe' (frozen, strong) or mode='full'.")
        head = model.get_classifier()
        for p in head.parameters():
            p.requires_grad_(True)
        for mod in model.modules():         # norms are cheap and help a lot
            if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d)):
                for p in mod.parameters():
                    p.requires_grad_(True)
    return model


def count_params(model) -> tuple[int, int]:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, tr


# ======================================================================================
# Two-stage transfer: ImageNet -> large FER corpus -> your ASD corpus
# ======================================================================================
STAGE2_RECIPE = """
TWO-STAGE TRANSFER  (do this before trying any new architecture)
================================================================
Expected payoff here is larger than any architecture swap on this list, and it is the
one experiment that produces a finding rather than a leaderboard row.

  Stage A  ImageNet weights                      (what you have now)
  Stage B  fine-tune on a large general FER corpus
  Stage C  fine-tune on your ASD corpus, subject-independent

For Stage B use AffectNet if you can obtain it -- a 2026 cross-scenario robustness study
found AffectNet-trained models generalised markedly better than RAF-DB-trained ones. If
AffectNet access is slow, RAF-DB or FER+ still beats going straight from ImageNet.

Map the label spaces explicitly and put the mapping in the paper. Your six classes are a
subset of the usual seven or eight; drop 'disgust' and 'contempt' from Stage B rather
than folding them into a nearby class.

Ablate all three: ImageNet-only vs +FER vs +FER+ASD. "Expression-domain pretraining is
worth X macro-F1 on ASD faces, and the gain is concentrated in the rare classes" is a
sentence a reviewer at JBHI or AIIM will find genuinely useful -- and it is a result
nobody has published for the ASD population.

One caution: check that your ASD sources do not overlap the FER corpus you pretrain on,
and say in the paper that you checked. Merged public face data has a habit of doing this.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tier", default=None)
    ap.add_argument("--check", default=None, help="model name to instantiate")
    ap.add_argument("--mode", default="full",
                    choices=["full", "lora", "probe", "linear"])
    ap.add_argument("--recipe", action="store_true")
    a = ap.parse_args()

    if a.recipe:
        print(STAGE2_RECIPE); return

    if a.list:
        for tier, desc in TIERS.items():
            if a.tier and tier != a.tier:
                continue
            rows = [e for e in MODEL_ZOO if e.tier == tier]
            if not rows:
                continue
            print(f"\n{'='*86}\n{tier.upper()}  —  {desc}\n{'='*86}")
            for e in rows:
                tag = "" if e.source == "timm" else f"  [{e.source}]"
                print(f"\n  {e.name}{tag}")
                for line in _wrap(e.why, 78):
                    print(f"      {line}")
                if e.note:
                    print(f"      → {e.note}")
        print()
        return

    if a.check:
        m = build_model(a.check, 6, mode=a.mode, pretrained=False)
        tot, tr = count_params(m)
        print(f"{a.check}  mode={a.mode}")
        print(f"  total params     {tot/1e6:8.2f} M")
        print(f"  trainable params {tr/1e6:8.2f} M  ({100*tr/max(tot,1):.2f}%)")
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = m(x)
        print(f"  output shape     {tuple(y.shape)}")
        return

    ap.print_help()


def _wrap(s, w):
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > w:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()