"""Generate a realistic synthetic run so every figure can be produced and inspected
without a GPU. Class counts, imbalance and performance level match the numbers in the
progress report; subject structure and fear/surprise confusability are simulated."""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd

CLASSES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]
COUNTS = dict(anger=210, fear=87, joy=861, natural=230, sadness=459, surprise=156)
SOURCES = ["nora", "ferac", "talaat", "hasibur"]
rng = np.random.RandomState(7)

out = Path("runs/demo")
if out.exists(): shutil.rmtree(out)
(out / "preds").mkdir(parents=True)

# ---- subjects ------------------------------------------------------------------------
# ~4.2 images per child, which is what makes image-level splitting leak
rows = []
gid = 0
for cls, n in COUNTS.items():
    left = n
    while left > 0:
        k = min(left, max(1, int(rng.gamma(3.0, 1.4))))
        src = SOURCES[rng.choice(4, p=[.38, .22, .22, .18])]
        for _ in range(k):
            rows.append({"label": cls, "group": f"ID{gid:04d}", "source": src})
        gid += 1; left -= k
df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
df["path"] = [f"/data/{i:05d}.jpg" for i in range(len(df))]
df["dup_cluster"] = df["path"]
df.to_csv(out / "manifest_demo.csv", index=False)
print(f"{len(df)} images, {df['group'].nunique()} subject groups, "
      f"{len(df)/df['group'].nunique():.1f} images/subject")

y = np.array([CLASSES.index(l) for l in df["label"]])
groups = df["group"].to_numpy()
sources = df["source"].to_numpy()
N = len(df)

# hold out ~15% by group for the locked test set
ug = np.array(df["group"].unique()); rng.shuffle(ug)
test_g = set(ug[:int(.15 * len(ug))])
te = df["group"].isin(test_g).to_numpy()
dev_i, test_i = np.where(~te)[0], np.where(te)[0]

# ---- confusable class geometry -------------------------------------------------------
# fear<->surprise share AU1+AU2+AU5; anger<->sadness share brow lowering.
P = rng.randn(6, 12) * 1.0
P[CLASSES.index("surprise")] = P[CLASSES.index("fear")] * .72 + rng.randn(12) * .42
P[CLASSES.index("sadness")] = P[CLASSES.index("anger")] * .55 + rng.randn(12) * .68
subj_diff = {g: rng.randn() for g in np.unique(groups)}          # per-child difficulty
src_bias = {s: rng.randn(6) * .30 for s in SOURCES}              # dataset fingerprint

MODELS = {  # name -> (signal strength, confidence temperature)
    "vgg16": (1.32, 1.15), "swin_base_patch4_window7_224": (1.31, 1.02),
    "inception_v3": (1.30, 1.20), "deit_small_patch16_224": (1.26, 1.05),
    "densenet121": (1.25, 1.18), "vit_base_patch16_224": (1.24, 1.00),
    "swin_tiny_patch4_window7_224": (1.22, 1.04), "efficientnet_b0": (1.18, 1.22),
    "mobilenetv2_100": (1.12, 1.28), "resnet50": (1.09, 1.30),
}

def logits_for(strength, temp, seed):
    """Noise level calibrated so the simulated benchmark lands on the accuracy and
    macro-F1 actually reported in the progress document (~0.72 / ~0.62), with the same
    best-to-worst spread. Weaker backbones simply see noisier features."""
    r = np.random.RandomState(seed)
    noise = 1.55 + (1.32 - strength) * 0.60
    W = r.randn(12, 6) * .35 + P.T * 1.30
    feat = P[y] + np.array([[subj_diff[g]] for g in groups]) * .55 \
        + r.randn(N, 12) * noise
    z = feat @ W
    z += np.array([src_bias[s] for s in sources]) * .55
    z[np.arange(N), y] += r.randn(N) * .25
    return (z / temp).astype(np.float32)

for mi, (name, (s, t)) in enumerate(MODELS.items()):
    for seed in (0, 1, 2):
        # deterministic: Python's str hash is salted per process, so never seed with it
        z = logits_for(s, t, seed * 977 + mi * 13)
        oof = np.zeros((len(dev_i), 6), np.float32); oof[:] = z[dev_i]
        tl = np.stack([z[test_i] + rng.randn(len(test_i), 6) * .18 for _ in range(5)])
        np.savez_compressed(
            out / "preds" / f"{name}__seed{seed}.npz",
            oof_logits=oof, oof_fold=np.tile(np.arange(5), len(dev_i) // 5 + 1)[:len(dev_i)],
            y_dev=y[dev_i], group_dev=groups[dev_i], source_dev=sources[dev_i],
            path_dev=df["path"].to_numpy()[dev_i],
            test_logits=tl.astype(np.float32), y_test=y[test_i],
            group_test=groups[test_i], source_test=sources[test_i],
            path_test=df["path"].to_numpy()[test_i])

        # ---- training history with a genuine train/val divergence ----------------------
        hist = []
        for k in range(5):
            r = np.random.RandomState(seed * 31 + k)
            ep, H = 26, []
            for e in range(ep):
                prog = (e + 1) / ep
                trl = 1.80 * np.exp(-3.1 * prog) + .10 + r.randn() * .022
                val = 1.72 * np.exp(-3.6 * prog) + .52 + .40 * max(0, prog - .48) ** 1.7 \
                      + r.randn() * .036
                trf = min(.985, .17 + .80 * (1 - np.exp(-3.3 * prog)) + r.randn() * .014)
                vaf = min(.72, .13 + (s - .55) * .62 * (1 - np.exp(-4.1 * prog))
                          - .05 * max(0, prog - .62) + r.randn() * .019)
                H.append({"epoch": e, "train_loss": float(trl), "val_loss": float(val),
                          "train_macro_f1": float(trf), "val_macro_f1": float(vaf),
                          "lr_head": float(3e-4 * (1 + np.cos(np.pi * prog)) / 2)})
            hist.append({"fold": k, "best_val_macro_f1": max(h["val_macro_f1"] for h in H),
                         "minutes": float(9 + r.rand() * 5), "history": H})
        (out / "preds" / f"{name}__seed{seed}_history.json").write_text(json.dumps(hist))

# ---- a LODO table --------------------------------------------------------------------
lodo = []
for held in SOURCES:
    for m in list(MODELS)[:4]:
        base = MODELS[m][0]
        lodo.append({"held_out_source": held, "model": m,
                     "n_test": int((sources == held).sum()),
                     "f1_macro": float(np.clip((base - .55) * .60
                                               + rng.randn() * .045, .2, .8)),
                     "accuracy": float(np.clip((base - .5) * .62 + rng.randn() * .04,
                                               .2, .85))})
pd.DataFrame(lodo).to_csv(out / "T8_lodo.csv", index=False)
print("demo run written to", out)