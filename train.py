"""
Train a YOLOv8 model on the Roboflow "Weapon 2" dataset (classes: guns, knife).

Usage:
    python train.py                         # defaults: dataset in ./dataset, 50 epochs
    python train.py --epochs 100 --batch 16
    python train.py --dataset "D:/data/weapon-2" --model yolov8s.pt

The best weights are copied to  models/best.pt  (used by detect_gui.py).
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def build_fixed_yaml(dataset_dir: Path) -> Path:
    """Roboflow's data.yaml uses '../train/images' paths which often break.
    We write a new yaml with an absolute dataset path and correct sub-folders."""
    src = dataset_dir / "data.yaml"
    if not src.exists():
        raise SystemExit(f"[ERROR] data.yaml not found in: {dataset_dir}\n"
                         "Put the extracted dataset (train/ valid/ test/ data.yaml) inside it.")

    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    def pick(*names):
        for n in names:
            if (dataset_dir / n / "images").exists():
                return f"{n}/images"
        return None

    train = pick("train")
    val = pick("valid", "val", "test")
    test = pick("test")
    if train is None or val is None:
        raise SystemExit("[ERROR] Could not find train/images and valid/images (or test/images) inside the dataset folder.")

    out = {
        "path": str(dataset_dir.resolve()),
        "train": train,
        "val": val,
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    if test:
        out["test"] = test

    fixed = dataset_dir / "data_fixed.yaml"
    fixed.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return fixed


def main():
    p = argparse.ArgumentParser(description="Train YOLOv8 on the weapon dataset")
    p.add_argument("--dataset", default=str(ROOT / "dataset"), help="folder containing data.yaml, train/, valid/, test/")
    p.add_argument("--model", default="yolov8n.pt", help="base model: yolov8n.pt (fast) / yolov8s.pt / yolov8m.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="lower this (8, 4) if you run out of memory; -1 = auto")
    p.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' for CPU; default = auto")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    data_yaml = build_fixed_yaml(Path(args.dataset))
    print(f"[INFO] Using dataset config: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs"),
        name="weapon",
        exist_ok=True,
    )

    best = ROOT / "runs" / "weapon" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit("[ERROR] Training finished but best.pt was not found.")

    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "best.pt")
    print("\n[DONE] Best model saved to: models/best.pt")
    print("       Now run:  python detect_gui.py")


if __name__ == "__main__":
    main()
