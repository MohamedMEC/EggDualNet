"""
╔══════════════════════════════════════════════════════════════════════╗
║          EggDualNet — Master Experiment Script                       ║
║  All architecture, data loading, training, TTA, baselines in one    ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO USE:
  Set RUN_MODE at the bottom of this file to one of:
    "eggdualnet"   → Train EggDualNet from scratch (5-fold CV)
    "tta"          → TTA evaluation using saved checkpoints
    "efficientnet" → Baseline: EfficientNet-B0 + CBAM (5-fold CV)
    "densenet"     → Baseline: DenseNet121 standalone (5-fold CV)
    "resnet"       → Baseline: ResNet50 (5-fold CV)
    "all_baselines"→ Run EfficientNet + DenseNet + ResNet sequentially

  Adjust CKPT_DIR and DATASET_DIR paths for your Kaggle environment.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, glob, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIG  (edit these) ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
DATASET_DIR = "/kaggle/input/datasets/mohamedkhanmec/eggplantdata/Original Dataset"
CKPT_DIR    = "/kaggle/input/datasets/mohamedkhanmec/checkpoints5"   # for TTA mode
SAVE_DIR    = "/kaggle/working"   # where to save new checkpoints

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 224
BATCH_SIZE  = 16
LR          = 5e-5
WD          = 1e-4
MAX_EPOCHS  = 50
PATIENCE    = 10
N_FOLDS     = 5
SEED        = 42
NUM_CLASSES = 10

CLASS_NAMES = [
    "Aphids", "Cercospora Leaf Spot", "Defect Eggplant", "Flea Beetles",
    "Fresh Eggplant", "Fresh Eggplant Leaf", "Leaf Wilt",
    "Phytophthora Blight", "Powdery Mildew", "Tobacco Mosaic Virus",
]

ETIOLOGY_MAP = {
    # 0=Fungal, 1=Healthy, 2=Pest, 3=Physiological, 4=Viral
    "Aphids": 2, "Cercospora Leaf Spot": 0, "Defect Eggplant": 3,
    "Flea Beetles": 2, "Fresh Eggplant": 1, "Fresh Eggplant Leaf": 1,
    "Leaf Wilt": 3, "Phytophthora Blight": 0, "Powdery Mildew": 0,
    "Tobacco Mosaic Virus": 4,
}

print(f"Device : {DEVICE}")
print(f"Dataset: {DATASET_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# ── DATA LOADING  (folder-matching bug fixed) ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def load_all_data(root):
    """
    Load all images + labels from root folder.
    Exact class-name match first, then startswith + length-diff ≤ 3.
    This avoids 'fresheggplant' folder matching 'fresheggplantleaf' class.
    """
    paths, labels, etiology = [], [], []
    folders = [e for e in os.scandir(root) if e.is_dir()]

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        needle = cls_name.lower().replace(" ", "").replace("_", "")

        # 1. Exact match
        matched = None
        for f in folders:
            if f.name.lower().replace(" ", "").replace("_", "") == needle:
                matched = f; break

        # 2. Fallback: startswith + close length
        if matched is None:
            for f in folders:
                hay = f.name.lower().replace(" ", "").replace("_", "")
                if needle.startswith(hay) and abs(len(needle) - len(hay)) <= 3:
                    matched = f; break

        if matched is None:
            print(f"  [WARNING] No folder for: {cls_name}"); continue

        found = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            found.extend(glob.glob(os.path.join(matched.path, ext)))
        print(f"  [{cls_idx:2d}] {cls_name:25s} → {matched.name} ({len(found)} imgs)")

        for p in found:
            paths.append(p)
            labels.append(cls_idx)
            etiology.append(ETIOLOGY_MAP[cls_name])

    return np.array(paths), np.array(labels), np.array(etiology)


# ── Transforms ────────────────────────────────────────────────────────────────
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
tf_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# TTA: 5 augmentation views
tta_transforms = [
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((int(IMG_SIZE * 1.15), int(IMG_SIZE * 1.15))),
                        transforms.CenterCrop(IMG_SIZE),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
]


# ── Dataset class ─────────────────────────────────────────────────────────────
class EggDataset(Dataset):
    def __init__(self, paths, labels, transform, etiology=None):
        self.paths = paths; self.labels = labels
        self.transform = transform; self.etiology = etiology
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        if self.etiology is not None:
            return img, self.labels[idx], self.etiology[idx]
        return img, self.labels[idx]


# ── Class weights ─────────────────────────────────────────────────────────────
def class_weights(y, n_classes=NUM_CLASSES):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / (counts + 1e-6)
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float32).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# ── MODEL ARCHITECTURES ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ─── 1. EggDualNet components ─────────────────────────────────────────────────
class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y   = torch.cat([x_h, x_w], dim=2)
        y   = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class PatchCNNEncoder(nn.Module):
    """hidden=128 (standard). Supports hidden=256 for checkpoint compatibility."""
    def __init__(self, out_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden, out_dim)
    def forward(self, x): return self.proj(self.net(x).flatten(1))


class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
    def forward(self, x, adj): return F.relu(self.linear(torch.bmm(adj, x)))


class SpatialGCNBranch(nn.Module):
    def __init__(self, patch_size=32, node_dim=256, gcn_dim=512, encoder_hidden=128):
        super().__init__()
        self.patch_size = patch_size
        self.encoder  = PatchCNNEncoder(node_dim, hidden=encoder_hidden)
        self.gcn1     = GCNLayer(node_dim, gcn_dim)
        self.gcn2     = GCNLayer(gcn_dim,  gcn_dim)
        self.out_proj = nn.Linear(gcn_dim, gcn_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ps      = self.patch_size
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        N       = patches.shape[2] * patches.shape[3]
        patches = patches.contiguous().view(B * N, C, ps, ps)
        node_f  = self.encoder(patches).view(B, N, -1)
        adj     = torch.ones(B, N, N, device=x.device) / N
        h       = self.gcn2(self.gcn1(node_f, adj), adj)
        return self.out_proj(h.mean(dim=1))


class EggDualNet(nn.Module):
    """
    Full EggDualNet: DenseNet121 global branch + Spatial GCN local branch.
    Returns (disease_logits, etiology_logits).
    """
    def __init__(self, encoder_hidden=128):
        super().__init__()
        self.global_features = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
        self.coord_att    = CoordinateAttention(1024, 1024)
        self.gcn_branch   = SpatialGCNBranch(encoder_hidden=encoder_hidden)
        self.fusion_norm  = nn.LayerNorm(1536)
        self.drop         = nn.Dropout(0.4)
        self.head_disease = nn.Linear(1536, NUM_CLASSES)
        self.head_etiology= nn.Linear(1536, 5)

    def forward(self, x):
        feat  = self.coord_att(self.global_features(x))
        g     = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        s     = self.gcn_branch(x)
        fused = self.drop(self.fusion_norm(torch.cat([g, s], dim=1)))
        return self.head_disease(fused), self.head_etiology(fused)


# ─── 2. CBAM modules (for EfficientNet baseline) ──────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, mid, bias=False),
                                nn.ReLU(),
                                nn.Linear(mid, channels, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c = x.shape[:2]
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))


# ─── 3. Baseline model builders ───────────────────────────────────────────────
def build_eggdualnet(encoder_hidden=128):
    return EggDualNet(encoder_hidden=encoder_hidden)

def build_efficientnet_cbam():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            base = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.features = base.features
            self.cbam = CBAM(1280)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.drop = nn.Dropout(0.3)
            self.head = nn.Linear(1280, NUM_CLASSES)
        def forward(self, x):
            f = self.cbam(self.features(x))
            return self.head(self.drop(self.pool(f).flatten(1)))
    return _Model()

def build_densenet121():
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(1024, NUM_CLASSES)
    return m

def build_resnet50():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# ── TRAINING & EVALUATION HELPERS ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, multi_task=False):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        imgs = batch[0].to(DEVICE)
        lbls = batch[1]
        if multi_task:
            logits, _ = model(imgs)
        else:
            logits = model(imgs)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(lbls.numpy())
    preds, trues = np.array(preds), np.array(trues)
    return (f1_score(trues, preds, average='macro') * 100,
            accuracy_score(trues, preds) * 100,
            cohen_kappa_score(trues, preds))


def run_cv(model_name, model_fn, paths, labels, etiology=None,
           multi_task=False, save_ckpts=False):
    """
    Generic 5-fold CV runner.
    model_fn: callable that returns a fresh model.
    multi_task: if True, model returns (disease_logits, etiology_logits)
                and uses combined loss (0.7 disease + 0.3 etiology).
    save_ckpts: save best checkpoint per fold to SAVE_DIR.
    """
    print(f"\n{'='*65}")
    print(f"  Running: {model_name}  |  5-Fold CV")
    print(f"{'='*65}")

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {"f1": [], "acc": [], "kap": []}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──")

        # Build datasets
        tr_et  = etiology[tr_idx]  if etiology is not None else None
        val_et = etiology[val_idx] if etiology is not None else None
        tr_ds  = EggDataset(paths[tr_idx], labels[tr_idx], tf_train, tr_et)
        val_ds = EggDataset(paths[val_idx], labels[val_idx], tf_val,  val_et)
        tr_ld  = DataLoader(tr_ds,  BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model    = model_fn().to(DEVICE)
        cw       = class_weights(labels[tr_idx])
        crit_d   = nn.CrossEntropyLoss(weight=cw)
        crit_e   = nn.CrossEntropyLoss()
        optimizer= torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

        best_f1, no_improve, best_state = 0.0, 0, None

        for epoch in range(1, MAX_EPOCHS + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            for batch in tr_ld:
                imgs = batch[0].to(DEVICE)
                lbls = batch[1].to(DEVICE)
                optimizer.zero_grad()
                if multi_task:
                    et   = batch[2].to(DEVICE)
                    d_out, e_out = model(imgs)
                    loss = 0.7 * crit_d(d_out, lbls) + 0.3 * crit_e(e_out, et)
                else:
                    loss = crit_d(model(imgs), lbls)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            f1, acc, kap = evaluate(model, val_ld, multi_task)
            print(f"  Ep {epoch:02d} | Loss={total_loss/len(tr_ld):.4f} | "
                  f"F1={f1:.2f}% | Acc={acc:.2f}% | κ={kap:.4f} | {time.time()-t0:.1f}s")

            if f1 > best_f1:
                best_f1   = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  → Early stop at epoch {epoch}"); break

        model.load_state_dict(best_state); model.to(DEVICE)
        f1, acc, kap = evaluate(model, val_ld, multi_task)
        results["f1"].append(f1); results["acc"].append(acc); results["kap"].append(kap)
        print(f"  ✅ Fold {fold} BEST → F1={f1:.2f}%  Acc={acc:.2f}%  κ={kap:.4f}")

        if save_ckpts:
            ckpt_path = os.path.join(SAVE_DIR, f"EggDualNet_fold{fold}_best.pth")
            torch.save({"model_state_dict": best_state}, ckpt_path)
            print(f"     Saved → {ckpt_path}")

    f1s  = np.array(results["f1"])
    accs = np.array(results["acc"])
    kaps = np.array(results["kap"])

    print(f"\n{'─'*55}")
    print(f"  {model_name} — 5-Fold Summary")
    print(f"  Per-fold F1  : {[f'{v:.2f}' for v in f1s]}")
    print(f"  Macro F1     : {f1s.mean():.2f}% ± {f1s.std():.2f}%")
    print(f"  Accuracy     : {accs.mean():.2f}% ± {accs.std():.2f}%")
    print(f"  Cohen κ      : {kaps.mean():.4f} ± {kaps.std():.4f}")
    print(f"{'─'*55}")
    return {"model": model_name,
            "f1_mean": f1s.mean(), "f1_std": f1s.std(),
            "acc_mean": accs.mean(), "acc_std": accs.std(),
            "kap_mean": kaps.mean(), "kap_std": kaps.std()}


# ══════════════════════════════════════════════════════════════════════════════
# ── TTA EVALUATION ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def tta_predict(model, paths, labels):
    """Run 5-view TTA on val split. Returns (preds, labels)."""
    model.eval()
    all_probs = []
    for tf in tta_transforms:
        ds  = EggDataset(paths, labels, tf)
        ld  = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        probs = []
        for imgs, _ in ld:
            logits, _ = model(imgs.to(DEVICE))
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs, axis=0))
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs.argmax(axis=1)


def run_tta(paths, labels):
    """Load saved fold checkpoints and evaluate with TTA."""
    print(f"\n{'='*65}")
    print(f"  EggDualNet + TTA  |  Loading checkpoints from {CKPT_DIR}")
    print(f"{'='*65}")

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {"f1": [], "acc": [], "kap": []}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(paths, labels), 1):
        ckpt_path = os.path.join(CKPT_DIR, f"EggDualNet_fold{fold}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"\n[SKIP] Fold {fold} — not found: {ckpt_path}"); continue

        print(f"\n── Fold {fold}/5  loading {ckpt_path}")
        ckpt       = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Auto-detect hidden size from checkpoint
        enc_hidden = state_dict["gcn_branch.encoder.net.6.weight"].shape[0]
        print(f"  Detected encoder_hidden={enc_hidden}")
        model = EggDualNet(encoder_hidden=enc_hidden).to(DEVICE)
        model.load_state_dict(state_dict, strict=False)

        preds = tta_predict(model, paths[val_idx], labels[val_idx])
        f1  = f1_score(labels[val_idx], preds, average='macro') * 100
        acc = accuracy_score(labels[val_idx], preds) * 100
        kap = cohen_kappa_score(labels[val_idx], preds)
        results["f1"].append(f1); results["acc"].append(acc); results["kap"].append(kap)
        print(f"  ✅ Fold {fold} TTA → F1={f1:.2f}%  Acc={acc:.2f}%  κ={kap:.4f}")

    f1s  = np.array(results["f1"])
    accs = np.array(results["acc"])
    kaps = np.array(results["kap"])
    print(f"\n{'─'*55}")
    print(f"  EggDualNet + TTA — 5-Fold Summary")
    print(f"  Per-fold F1  : {[f'{v:.2f}' for v in f1s]}")
    print(f"  Macro F1     : {f1s.mean():.2f}% ± {f1s.std():.2f}%")
    print(f"  Accuracy     : {accs.mean():.2f}% ± {accs.std():.2f}%")
    print(f"  Cohen κ      : {kaps.mean():.4f} ± {kaps.std():.4f}")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN RUNNER ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── SET THIS ──────────────────────────────────────────────────────────────
    RUN_MODE = "eggdualnet"   # Options: eggdualnet | tta | efficientnet |
    #                                     densenet  | resnet | all_baselines
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\nMode: {RUN_MODE}\n")

    # Load data (always needed)
    print("Loading dataset...")
    paths, labels, etiology = load_all_data(DATASET_DIR)
    print(f"Total: {len(paths)} images, {len(set(labels))} classes\n")

    # ── Known results for reference ───────────────────────────────────────────
    print("─" * 55)
    print("  Reference results (paper):")
    print("  EggDualNet (no TTA) : F1=95.72%±0.99%  Acc=96.53%±0.63%  κ=0.9605")
    print("  EggDualNet + TTA    : F1=97.90%±2.02%  Acc=98.30%±1.38%  κ=0.9806")
    print("  ResNet50            : F1=96.40%±0.37%  Acc=97.27%±0.32%  κ=0.9689")
    print("  DenseNet121         : F1=96.26%±1.37%  Acc=97.02%±1.16%  κ=0.9660")
    print("  EfficientNet+CBAM   : F1=95.84%±0.90%  Acc=96.60%±0.77%  κ=0.9613")
    print("  DINOv2-Base (frozen): F1=92.84%±1.09%  Acc=94.80%±0.82%  κ=0.9407")
    print("─" * 55 + "\n")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if RUN_MODE == "eggdualnet":
        run_cv("EggDualNet (Ours)", build_eggdualnet,
               paths, labels, etiology, multi_task=True, save_ckpts=True)

    elif RUN_MODE == "tta":
        run_tta(paths, labels)

    elif RUN_MODE == "efficientnet":
        run_cv("EfficientNet-B0+CBAM", build_efficientnet_cbam,
               paths, labels)

    elif RUN_MODE == "densenet":
        run_cv("DenseNet121", build_densenet121,
               paths, labels)

    elif RUN_MODE == "resnet":
        run_cv("ResNet50", build_resnet50,
               paths, labels)

    elif RUN_MODE == "all_baselines":
        all_results = []
        for name, fn in [("EfficientNet-B0+CBAM", build_efficientnet_cbam),
                         ("DenseNet121",           build_densenet121),
                         ("ResNet50",              build_resnet50)]:
            all_results.append(run_cv(name, fn, paths, labels))

        print("\n" + "="*70)
        print("  FINAL SUMMARY")
        print("="*70)
        print(f"{'Model':<28} {'Macro F1':>16} {'Accuracy':>18} {'κ':>16}")
        print("─"*70)
        for r in all_results:
            print(f"{r['model']:<28} "
                  f"{r['f1_mean']:.2f}%±{r['f1_std']:.2f}%  "
                  f"{r['acc_mean']:.2f}%±{r['acc_std']:.2f}%  "
                  f"{r['kap_mean']:.4f}±{r['kap_std']:.4f}")
        print("="*70)

    else:
        print(f"Unknown RUN_MODE: '{RUN_MODE}'")
        print("Valid options: eggdualnet | tta | efficientnet | densenet | resnet | all_baselines")
