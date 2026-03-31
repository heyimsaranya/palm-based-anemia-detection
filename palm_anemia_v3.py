"""
Palm Anemia Classification — v3
Fixes: hardcoded DATA_DIR + deep scan to find actual image folders
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: DEBUG — run this first to see your folder structure
# ─────────────────────────────────────────────
print("=" * 60)
print("FULL DIRECTORY TREE under /kaggle/input")
print("=" * 60)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
for root, dirs, files in os.walk("/kaggle/input"):
    level  = root.replace("/kaggle/input", "").count(os.sep)
    indent = "  " * level
    imgs   = [f for f in files if Path(f).suffix.lower() in IMG_EXTS]
    print(f"{indent}{os.path.basename(root)}/  [{len(imgs)} images]")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 2: Auto-find folders that actually contain images
# ─────────────────────────────────────────────
def find_image_folders(base="/kaggle/input"):
    """Returns all leaf directories that contain image files."""
    found = []
    for root, dirs, files in os.walk(base):
        imgs = [f for f in files if Path(f).suffix.lower() in IMG_EXTS]
        if imgs:
            found.append((root, len(imgs)))
    return found

image_folders = find_image_folders()
print("\nFolders containing images:")
for folder, count in image_folders:
    print(f"  {folder}  ({count} images)")

# ─────────────────────────────────────────────
# STEP 3: CONFIG — DATA_DIR is set to the parent of image folders
# ─────────────────────────────────────────────
class Config:
    # Hardcoded fallback; will be overridden by auto-detect below
    DATA_DIR     = "/kaggle/input/datasets/shreyacgosavi/palm-dataset-anemia/Palm"
    IMG_SIZE     = 224
    BATCH_SIZE   = 32
    EPOCHS       = 30
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS  = 2
    SEED         = 42
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    CLASSES      = ["Anemic", "Non-Anemic"]
    TRAIN_SPLIT  = 0.70
    VAL_SPLIT    = 0.15
    TEST_SPLIT   = 0.15

cfg = Config()

# Auto-detect DATA_DIR: find the parent dir that has 2+ image-containing subfolders
def auto_detect_data_dir(image_folders):
    parents = {}
    for folder, count in image_folders:
        parent = str(Path(folder).parent)
        parents.setdefault(parent, []).append((folder, count))

    for parent, children in parents.items():
        if len(children) >= 2:
            print(f"\n✓ Auto-detected DATA_DIR: {parent}")
            print(f"  Sub-folders: {[Path(c[0]).name for c in children]}")
            return parent

    # If only one level deep (images directly in Palm/ with no subfolders)
    # Group by grandparent
    if len(image_folders) >= 2:
        print(f"\n✓ Using parent of first image folder: {Path(image_folders[0][0]).parent}")
        return str(Path(image_folders[0][0]).parent)

    # Single folder — images not split into class folders, need manual labels from filename
    if len(image_folders) == 1:
        print(f"\n⚠ Only one image folder found: {image_folders[0][0]}")
        print("  Will try to infer labels from filenames (Anemic/Non-Anemic in name).")
        return image_folders[0][0]

    return cfg.DATA_DIR  # fallback

cfg.DATA_DIR = auto_detect_data_dir(image_folders)
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
print(f"\nDevice: {cfg.DEVICE}")
print(f"DATA_DIR: {cfg.DATA_DIR}")

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE + 32, cfg.IMG_SIZE + 32)),
    transforms.RandomCrop(cfg.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_test_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
def assign_label_from_folder(name):
    n = name.lower().replace("-","").replace("_","").replace(" ","")
    return 1 if ("non" in n or "healthy" in n or "normal" in n) else 0

def assign_label_from_filename(fname):
    n = fname.lower()
    return 1 if ("non" in n or "healthy" in n or "normal" in n) else 0

class PalmDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels     = labels
        self.transform  = transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]


def build_datasets(data_dir):
    data_dir  = Path(data_dir)
    subdirs   = [d for d in data_dir.iterdir() if d.is_dir()]
    file_paths, labels = [], []

    if subdirs:
        # Normal case: class subfolders
        print("\nUsing folder-based labels:")
        for folder in sorted(subdirs):
            label = assign_label_from_folder(folder.name)
            imgs  = [f for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS]
            if not imgs:
                print(f"  ⚠ Skipping '{folder.name}' — no images")
                continue
            print(f"  '{folder.name}'  →  label={label}  ({len(imgs)} imgs)")
            file_paths.extend(imgs)
            labels.extend([label] * len(imgs))
    else:
        # Fallback: images directly in data_dir, infer label from filename
        print("\nNo subfolders found — inferring labels from filenames:")
        imgs = [f for f in data_dir.rglob("*") if f.suffix.lower() in IMG_EXTS]
        if not imgs:
            raise FileNotFoundError(f"No images found in {data_dir}")
        for img in imgs:
            labels.append(assign_label_from_filename(img.name))
            file_paths.append(img)
        c0 = labels.count(0); c1 = labels.count(1)
        print(f"  Label 0 (Anemic): {c0}  |  Label 1 (Non-Anemic): {c1}")

    assert len(file_paths) > 0, "No images found!"
    if len(set(labels)) < 2:
        raise ValueError(
            f"Only 1 class found (labels={set(labels)}).\n"
            f"Folder names: {[d.name for d in subdirs]}\n"
            "Rename folders to include 'Anemic' / 'Non-Anemic' or 'Healthy'."
        )

    print(f"\nTotal: {len(file_paths)} images | Classes: {cfg.CLASSES}")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        file_paths, labels,
        test_size=(cfg.VAL_SPLIT + cfg.TEST_SPLIT),
        stratify=labels, random_state=cfg.SEED)
    ratio = cfg.TEST_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=ratio,
        stratify=y_tmp, random_state=cfg.SEED)
    print(f"Split → Train:{len(X_tr)} | Val:{len(X_val)} | Test:{len(X_te)}")

    return (PalmDataset(X_tr,  y_tr,  train_transform),
            PalmDataset(X_val, y_val, val_test_transform),
            PalmDataset(X_te,  y_te,  val_test_transform),
            y_tr)

def make_loaders(train_ds, val_ds, test_ds, y_train):
    counts   = np.bincount(y_train)
    weights  = 1.0 / counts
    sample_w = [weights[y] for y in y_train]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w))
    return (
        DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                   num_workers=cfg.NUM_WORKERS, pin_memory=True),
        DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                   num_workers=cfg.NUM_WORKERS),
        DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,
                   num_workers=cfg.NUM_WORKERS),
    )

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes=2):
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_feat, 256), nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    for block in list(model.features.children())[-2:]:
        for p in block.parameters(): p.requires_grad = True
    return model.to(cfg.DEVICE)

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(cfg.DEVICE), lbls.to(cfg.DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out  = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == lbls).sum().item()
        total      += imgs.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    p_all, l_all, pr_all = [], [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(cfg.DEVICE), lbls.to(cfg.DEVICE)
        out  = model(imgs)
        loss = criterion(out, lbls)
        total_loss += loss.item() * imgs.size(0)
        probs  = torch.softmax(out, 1)[:,1]
        preds  = out.argmax(1)
        correct += (preds == lbls).sum().item()
        total   += imgs.size(0)
        p_all.extend(preds.cpu().numpy())
        l_all.extend(lbls.cpu().numpy())
        pr_all.extend(probs.cpu().numpy())
    return total_loss/total, correct/total, np.array(l_all), np.array(p_all), np.array(pr_all)

def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler    = torch.cuda.amp.GradScaler()
    best_acc  = 0.0
    history   = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    print(f"\n{'Ep':>4} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7}")
    print("-" * 45)
    for epoch in range(1, cfg.EPOCHS+1):
        if epoch == 11:
            for p in model.parameters(): p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=cfg.LR*0.1,
                                    weight_decay=cfg.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.EPOCHS-epoch)
            print("  [Ep11] Full backbone unfrozen")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, va_acc, *_ = evaluate(model, val_loader, criterion)
        scheduler.step()
        for k,v in zip(history,[tr_loss,tr_acc,va_loss,va_acc]): history[k].append(v)
        print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc:>6.2%}  {va_loss:>8.4f}  {va_acc:>6.2%}", end="")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("  ✓", end="")
        print()
    print(f"\nBest Val Acc: {best_acc:.2%}")
    return history

# ─────────────────────────────────────────────
# PLOTS & EVALUATION
# ─────────────────────────────────────────────
def plot_history(h):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(h["train_loss"])+1)
    for ax, k, t in zip(axes, ["loss","acc"], ["Loss","Accuracy"]):
        ax.plot(ep, h[f"train_{k}"], label="Train")
        ax.plot(ep, h[f"val_{k}"],   label="Val")
        ax.set_title(t); ax.legend(); ax.set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.show()

def full_evaluation(model, test_loader, criterion):
    model.load_state_dict(torch.load("best_model.pth", map_location=cfg.DEVICE))
    _, acc, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=cfg.CLASSES))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.CLASSES, yticklabels=cfg.CLASSES)
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150); plt.show()
    fpr,tpr,_ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr,tpr,lw=2,label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150); plt.show()
    print(f"ROC-AUC: {auc:.4f}")

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.grads = self.acts = None
        target_layer.register_forward_hook(
            lambda m,i,o: setattr(self,"acts",o.detach()))
        target_layer.register_full_backward_hook(
            lambda m,gi,go: setattr(self,"grads",go[0].detach()))
    def __call__(self, img_t, class_idx=None):
        self.model.eval()
        img_t  = img_t.unsqueeze(0).to(cfg.DEVICE)
        logits = self.model(img_t)
        if class_idx is None: class_idx = logits.argmax(1).item()
        self.model.zero_grad(); logits[0, class_idx].backward()
        w   = self.grads.mean(dim=[2,3], keepdim=True)
        cam = torch.relu((w*self.acts).sum(1)).squeeze().cpu().numpy()
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cam, class_idx

def show_gradcam(model, dataset, num_images=4):
    import cv2
    tgt  = list(model.features.children())[-1][-1]
    gcam = GradCAM(model, tgt)
    inv  = transforms.Normalize(
        mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225],
        std=[1/0.229,1/0.224,1/0.225])
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*3, 6))
    for col, idx in enumerate(np.random.choice(len(dataset), num_images, replace=False)):
        img_t, label = dataset[idx]
        cam, pred    = gcam(img_t)
        img_np       = inv(img_t).permute(1,2,0).clamp(0,1).numpy()
        cam_r        = cv2.resize(cam, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        overlay      = 0.5*img_np + 0.5*plt.cm.jet(cam_r)[...,:3]
        axes[0,col].imshow(img_np);  axes[0,col].set_title(f"True:{cfg.CLASSES[label]}"); axes[0,col].axis("off")
        axes[1,col].imshow(overlay); axes[1,col].set_title(f"Pred:{cfg.CLASSES[pred]}");  axes[1,col].axis("off")
    plt.suptitle("Grad-CAM", fontsize=14); plt.tight_layout()
    plt.savefig("gradcam.png", dpi=150); plt.show()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds, test_ds, y_train = build_datasets(cfg.DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, y_train)

    model = build_model(num_classes=2)
    print(f"EfficientNetV2-S | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history = train(model, train_loader, val_loader)
    plot_history(history)

    criterion = nn.CrossEntropyLoss()
    full_evaluation(model, test_loader, criterion)
    show_gradcam(model, test_ds, num_images=4)
    print("\nDone! Outputs: training_curves.png | confusion_matrix.png | roc_curve.png | gradcam.png | best_model.pth")
