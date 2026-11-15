# Weapon Detector (YOLOv8)

A simple project that:

1. **Trains** a YOLOv8 object-detection model on the Roboflow *Weapon 2* dataset (classes: `guns`, `knife`).
2. Opens a small **desktop app** where you pick photos from your PC. The app draws boxes around detected weapons, shows **how certain** the model is (confidence %), and shows an **alert** when a weapon is found.

> This is a learning/demo project. Do not rely on it as a real security system: accuracy depends on the dataset and training, and it will make mistakes (false alarms and missed weapons).

---

## Project structure

```
weapon-detector/
├── dataset/            <- you put the downloaded dataset here
│   ├── data.yaml
│   ├── train/images, train/labels
│   ├── valid/images, valid/labels
│   └── test/images,  test/labels
├── train.py            <- trains the model
├── detect_gui.py       <- app to select photos and detect
├── requirements.txt
├── models/best.pt      <- created after training
└── README.md
```

## 1. Requirements

- Python **3.9 – 3.12**
- A GPU (NVIDIA) is strongly recommended for training, but CPU works (slow).
- Linux only: tkinter may need installing: `sudo apt install python3-tk`

## 2. Setup

Open a terminal inside the `weapon-detector` folder.

```bash
# create a virtual environment (recommended)
python -m venv venv

# activate it
#   Windows:
venv\Scripts\activate
#   Linux / macOS:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

**NVIDIA GPU users (Windows/Linux):** if training says it is using CPU, install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ , then re-run training. You can check with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Add the dataset

1. Download the dataset from Roboflow in **YOLOv8** format (you already did this).
2. Extract the zip into the `dataset/` folder so that `dataset/data.yaml`, `dataset/train/`, `dataset/valid/` and `dataset/test/` exist.

`train.py` automatically creates a corrected `dataset/data_fixed.yaml` (Roboflow's default relative paths often break), so you do not need to edit `data.yaml`.

## 4. Train

```bash
python train.py
```

Useful options:

| Option | Meaning | Default |
|---|---|---|
| `--epochs 100` | number of training rounds (more = usually better, slower) | 50 |
| `--batch 8` | batch size; lower it if you get out-of-memory errors | 16 |
| `--model yolov8s.pt` | bigger base model = more accurate but slower (`yolov8n.pt` is the smallest) | yolov8n.pt |
| `--device 0` / `--device cpu` | force GPU or CPU | auto |
| `--dataset "D:/path"` | dataset in another folder | ./dataset |

Examples:

```bash
python train.py --epochs 100 --batch 16
python train.py --device cpu --epochs 20 --batch 8      # quick test on CPU
```

Training time (roughly, ~4,000 images): a few minutes to ~1 hour on a GPU for 50 epochs; many hours on CPU. If you have no GPU, train for free on **Google Colab** (Runtime → Change runtime type → GPU): upload the project and dataset, run `pip install ultralytics`, `python train.py`, then download `models/best.pt` and put it in the same folder on your PC.

When finished you get:
- `models/best.pt` - the trained model used by the app
- `runs/weapon/` - training graphs, confusion matrix, and validation results (check `results.png`)

## 5. Run the detector app

```bash
python detect_gui.py
```

1. Click **Select image(s)...** and choose one or more photos.
2. The app shows each photo with boxes around detected weapons.
3. The banner turns **red** with `ALERT: WEAPON DETECTED` and the highest confidence, or **green** with `No weapon detected`.
4. The text box lists every detection with its certainty (e.g. `guns  certainty: 87.3%`).
5. A pop-up warning (and a beep on Windows) appears if any selected photo contains a weapon.
6. Use **Prev / Next** to browse multiple photos.

**Confidence threshold slider:** detections below this value are ignored. Lower it (e.g. 0.15) to catch more weapons but get more false alarms; raise it (e.g. 0.5) for fewer false alarms but more misses. Move the slider *before* selecting photos.

## Troubleshooting

- **`Model not found`** - run `python train.py` first (it creates `models/best.pt`).
- **`data.yaml not found`** - the dataset must be extracted directly inside `dataset/`.
- **Out of memory during training** - use `--batch 8` or `--batch 4`.
- **Windows: errors about workers/multiprocessing** - add `--workers 0`.
- **Poor results** - train longer (`--epochs 100`), use `--model yolov8s.pt`, and check `runs/weapon/results.png` and the confusion matrix. The dataset's quality is the main limit.

## Dataset license

*Weapon 2* by a Roboflow user (joao-assalim-xmovq), license **CC BY 4.0**:
https://universe.roboflow.com/joao-assalim-xmovq/weapon-2 — credit the author if you publish your work.
