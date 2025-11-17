#!/usr/bin/env python3
"""
pipeline.py — VisionInfantNet

End-to-end training, evaluation, and serving for image-based infant vocalization
classification using fastai + timm (ConvNeXt by default).

Supports:
- Folder datasets: data/
    train/<class>/*.png
    valid/<class>/*.png   (optional; will be auto-split if missing)
    test/<class>/*.png    (optional)
- CSV datasets: a CSV with columns: filepath,label  (use --csv path/to.csv)

Commands:
    python pipeline.py train --data_dir data --arch convnext_tiny --epochs 10
    python pipeline.py eval  --data_dir data --ckpt models/best
    python pipeline.py serve --export_path models/ConvNext.pkl --host 0.0.0.0 --port 7860

Notes:
- Exports:
  * FastAI export (pickle): models/ConvNext.pkl
  * Safe weights:          models/best.pth (learn.save('best'))
- FastAPI requires: fastapi, uvicorn
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# fastai 2.7.x API
from fastai.vision.all import *
from fastai.metrics import RocAuc
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback

# timm integration
import timm

# sklearn for reports
from sklearn.metrics import classification_report, confusion_matrix as sk_confusion_matrix


# -----------------------------
# Utilities
# -----------------------------
DEFAULT_CLASSES = ["Junk", "Non-canonical", "Canonical", "Cry", "Laugh"]
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def _parse_classes(s: Optional[str]) -> List[str]:
    if not s:
        return DEFAULT_CLASSES
    return [c.strip() for c in s.split(",") if c.strip()]


def _ensure_exists(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")


def seed_everything(seed: int = 2025):
    set_seed(seed, reproducible=True)
    random.seed(seed)
    np.random.seed(seed)


def _infer_has_valid_test(data_dir: Path):
    has_valid = (data_dir / "valid").exists()
    has_test = (data_dir / "test").exists()
    return has_valid, has_test


# -----------------------------
# Data
# -----------------------------
def build_dataloaders(
    data_dir: Path,
    csv_path: Optional[Path] = None,
    img_size: int = 224,
    bs: int = 32,
    classes: Optional[List[str]] = None,
    valid_pct: float = 0.1,
    seed: int = 2025,
) -> ImageDataLoaders:
    """
    Creates DataLoaders either from a CSV (filepath,label) or from folder structure.
    """
    seed_everything(seed)
    item_tfms = [Resize(img_size)]
    batch_tfms = aug_transforms(
        flip_vert=False,
        max_rotate=10,
        max_zoom=1.1,
        max_lighting=0.2,
        max_warp=0.1,
        p_affine=0.75,
        p_lighting=0.75,
    )

    if csv_path:
        _ensure_exists(csv_path, "CSV")
        import pandas as pd

        df = pd.read_csv(csv_path)
        if not {"filepath", "label"}.issubset(df.columns):
            raise ValueError("CSV must contain columns: filepath,label")

        # Resolve file paths relative to data_dir if they are not absolute
        def _resolve(p):
            p = Path(p)
            return p if p.is_absolute() else (data_dir / p)

        df["filepath"] = df["filepath"].map(lambda x: str(_resolve(x)))

        dls = ImageDataLoaders.from_df(
            df,
            valid_pct=valid_pct,
            seed=seed,
            fn_col="filepath",
            label_col="label",
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
            bs=bs,
        )
        if classes:
            dls.vocab = classes
        return dls

    # Folder mode
    _ensure_exists(data_dir, "Data directory")
    has_valid, _ = _infer_has_valid_test(data_dir)
    if has_valid:
        # Expect train/ and valid/ (and optionally test/)
        dls = ImageDataLoaders.from_folder(
            data_dir,
            train="train",
            valid="valid",
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
            bs=bs,
        )
    else:
        # Single folder auto-split from train/
        if not (data_dir / "train").exists():
            raise FileNotFoundError(
                f"Expected {data_dir}/train with class subfolders when no CSV and no valid/"
            )
        dls = ImageDataLoaders.from_folder(
            data_dir / "train",
            valid_pct=valid_pct,
            seed=seed,
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
            bs=bs,
        )

    if classes:
        dls.vocab = classes
    return dls


# -----------------------------
# Model / Training
# -----------------------------
def build_learner(
    dls: ImageDataLoaders,
    arch: str = "convnext_tiny",
    lr: float = 3e-3,
    pretrained: bool = True,
    n_out: Optional[int] = None,
):
    """
    Builds a timm-based learner (ConvNeXt by default).
    """
    if n_out is None:
        n_out = len(dls.vocab)

    learn = timm_learner(
        dls,
        arch=arch,
        pretrained=pretrained,
        metrics=[
            accuracy,
            error_rate,
            F1Score(average="macro"),
            Precision(average="macro"),
            Recall(average="macro"),
        ],
        n_out=n_out,
    )
    return learn


def train(
    data_dir: str,
    csv: Optional[str],
    arch: str,
    epochs: int,
    lr: float,
    bs: int,
    img_size: int,
    seed: int,
    valid_pct: float,
    pretrained: bool,
    classes_str: Optional[str],
):
    classes = _parse_classes(classes_str)
    dls = build_dataloaders(
        Path(data_dir),
        csv_path=Path(csv) if csv else None,
        img_size=img_size,
        bs=bs,
        classes=classes,
        valid_pct=valid_pct,
        seed=seed,
    )

    learn = build_learner(dls, arch=arch, lr=lr, pretrained=pretrained, n_out=len(classes))
    learn.model_dir = MODEL_DIR

    cbs = [
        SaveModelCallback(monitor="accuracy", fname="best"),
        EarlyStoppingCallback(monitor="accuracy", patience=5),
    ]

    learn.fine_tune(epochs, base_lr=lr, cbs=cbs)

    # Export pickle (for fastai load_learner) and safe weights (.pth)
    export_path = MODEL_DIR / "ConvNext.pkl"
    learn.export(export_path)
    learn.save("last")  # models/last.pth
    print(f"[OK] Exported: {export_path}")
    print(f"[OK] Saved weights: {MODEL_DIR / 'best.pth'} (best) and {MODEL_DIR / 'last.pth'} (last)")

    # Validation report
    preds, targs = learn.get_preds(with_decoded=True)
    dec = targs
    probs = preds
    y_true = targs.numpy()
    y_pred = dec.numpy()
    report = classification_report(
        y_true, y_pred, target_names=dls.vocab, output_dict=True, zero_division=0
    )
    with open(MODEL_DIR / "val_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Wrote validation report to {MODEL_DIR / 'val_report.json'}")

    # Confusion matrix
    cm = sk_confusion_matrix(y_true, y_pred)
    np.savetxt(MODEL_DIR / "val_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    print(f"[OK] Wrote confusion matrix to {MODEL_DIR / 'val_confusion_matrix.csv'}")


def evaluate(
    data_dir: str,
    csv: Optional[str],
    ckpt: Optional[str],
    img_size: int,
    bs: int,
    classes_str: Optional[str],
    seed: int,
    valid_pct: float,
):
    """
    Evaluate a saved model (best by default) on valid/ or test/ if present.
    """
    classes = _parse_classes(classes_str)
    dls = build_dataloaders(
        Path(data_dir),
        csv_path=Path(csv) if csv else None,
        img_size=img_size,
        bs=bs,
        classes=classes,
        valid_pct=valid_pct,
        seed=seed,
    )

    # Prefer loading fastai export if available, else weights
    export_path = MODEL_DIR / "ConvNext.pkl"
    if export_path.exists():
        learn = load_learner(export_path)
    else:
        # build from weights
        learn = build_learner(dls, arch="convnext_tiny", pretrained=False, n_out=len(classes))
        weight_stub = ckpt or "best"
        learn.load(weight_stub)

    # If test set exists, evaluate on it; else use valid
    has_valid, has_test = _infer_has_valid_test(Path(data_dir))
    dl = None
    split_name = "valid"
    if has_test:
        # Build test dataloader
        test_files = get_image_files(Path(data_dir) / "test")
        dl = learn.dls.test_dl(test_files)
        split_name = "test"
    else:
        dl = learn.dls.valid

    preds, targs = learn.get_preds(dl=dl, with_decoded=True)
    y_true = targs.numpy()
    y_pred = preds.argmax(dim=1).numpy()
    report = classification_report(
        y_true, y_pred, target_names=learn.dls.vocab, output_dict=True, zero_division=0
    )
    out_json = MODEL_DIR / f"{split_name}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    cm = sk_confusion_matrix(y_true, y_pred)
    np.savetxt(MODEL_DIR / f"{split_name}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    print(f"[OK] {split_name.capitalize()} report: {out_json}")
    print(f"[OK] {split_name.capitalize()} confusion matrix: {MODEL_DIR / f'{split_name}_confusion_matrix.csv'}")



# ----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="VisionInfantNet pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Common
    def add_data_args(p):
        p.add_argument("--data_dir", type=str, required=True, help="Root data directory")
        p.add_argument("--csv", type=str, default=None, help="CSV with columns: filepath,label")
        p.add_argument("--img_size", type=int, default=224)
        p.add_argument("--bs", type=int, default=32)
        p.add_argument("--classes", type=str, default=None, help="Comma-separated class names")
        p.add_argument("--seed", type=int, default=2025)
        p.add_argument("--valid_pct", type=float, default=0.1)

    # train
    p_train = sub.add_parser("train", help="Train a model")
    add_data_args(p_train)
    p_train.add_argument("--arch", type=str, default="convnext_tiny", help="timm arch (e.g., convnext_tiny, convnext_small, resnet50)")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--pretrained", action="store_true", help="Use pretrained weights (default False)")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a saved model")
    add_data_args(p_eval)
    p_eval.add_argument("--ckpt", type=str, default=None, help="Weight name (e.g., best, last) if .pkl not present")

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            data_dir=args.data_dir,
            csv=args.csv,
            arch=args.arch,
            epochs=args.epochs,
            lr=args.lr,
            bs=args.bs,
            img_size=args.img_size,
            seed=args.seed,
            valid_pct=args.valid_pct,
            pretrained=bool(args.pretrained),
            classes_str=args.classes,
        )
    elif args.cmd == "eval":
        evaluate(
            data_dir=args.data_dir,
            csv=args.csv,
            ckpt=args.ckpt,
            img_size=args.img_size,
            bs=args.bs,
            classes_str=args.classes,
            seed=args.seed,
            valid_pct=args.valid_pct,
        )
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    from io import BytesIO
    main()
