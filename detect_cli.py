"""
CLI Weapon Detector.
Runs YOLOv8 model inference on images and saves annotated results.

Usage:
    python detect_cli.py
    python detect_cli.py --source dataset/test/images --conf 0.3
    python detect_cli.py --source path/to/image.jpg
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "best.pt"
if not DEFAULT_MODEL.exists() and (ROOT / "best.pt").exists():
    DEFAULT_MODEL = ROOT / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Weapon Detection CLI")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to best.pt weights")
    parser.add_argument("--source", default=str(ROOT / "dataset" / "test" / "images"), help="Image file or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--save-dir", default=str(ROOT / "output_detections"), help="Directory to save annotated images")
    parser.add_argument("--limit", type=int, default=10, help="Max images to process if source is a folder (0 for all)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"[ERROR] Model weights not found at: {model_path}")

    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f"[ERROR] Source not found at: {source_path}")

    if source_path.is_file():
        image_files = [source_path]
    else:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [f for f in source_path.iterdir() if f.suffix.lower() in valid_exts]
        if args.limit > 0:
            image_files = image_files[:args.limit]

    if not image_files:
        raise SystemExit(f"[ERROR] No valid image files found in {source_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model from {model_path}...")
    model = YOLO(str(model_path))
    print(f"[INFO] Classes detected by model: {model.names}")
    print(f"[INFO] Processing {len(image_files)} image(s) with confidence >= {args.conf}...\n")

    weapons_found = 0
    for idx, img_path in enumerate(image_files, 1):
        res = model.predict(source=str(img_path), conf=args.conf, verbose=False)[0]
        dets = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            name = res.names[cls_id]
            dets.append(f"{name} ({conf_val * 100:.1f}%)")

        out_img_path = save_dir / f"detected_{img_path.name}"
        res.save(filename=str(out_img_path))

        status = f"ALERT: Detected {', '.join(dets)}" if dets else "No weapons detected"
        if dets:
            weapons_found += 1
        print(f"[{idx}/{len(image_files)}] {img_path.name} -> {status}")

    print(f"\n[DONE] Finished detection on {len(image_files)} image(s).")
    print(f"       Weapons detected in {weapons_found} image(s).")
    print(f"       Annotated results saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
