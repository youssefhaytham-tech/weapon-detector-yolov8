# Weapon Detector (YOLOv8)

Deep-learning project that detects **guns** and **knives** in photos using a YOLOv8 object-detection model. It includes a desktop app and a command-line tool. Built for the Artificial Intelligence course (ELTE 322).

> ⚠️ **Educational demo only.** This is not a real security system. It makes false alarms and misses weapons. Never use it for real decisions.

## Features
- Trains YOLOv8 on the Roboflow *Weapon 2* dataset (classes: `guns`, `knife`)
- **Desktop app** (CustomTkinter): pick photos, see boxes, confidence ring, scan summary, alert pop-up and optional alarm beep
- **Confidence slider** that filters results instantly without re-running the model
- **CLI tool** for single images or whole folders
- Annotated results saved automatically to `output_detections/`

## Project structure
```
weapon-detector/
├── dataset/            # Roboflow dataset (not included)
├── models/best.pt      # trained weights
├── train.py            # train the model
├── detect_gui.py       # desktop app
├── detect_cli.py       # command-line detector
├── run_gui.bat         # Windows launcher
├── requirements.txt
└── README.md
```

## Setup
Requires Python 3.9–3.12.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```
Linux only: `sudo apt install python3-tk`

## Dataset
Download **Weapon 2** from Roboflow in **YOLOv8** format and extract it into `dataset/`:
https://universe.roboflow.com/joao-assalim-xmovq/weapon-2

`train.py` creates a corrected `data_fixed.yaml` automatically.

## Train
```bash
python train.py
python train.py --epochs 100 --batch 16 --model yolov8s.pt
python train.py --device cpu --epochs 20 --batch 8
```
The best weights are saved to `models/best.pt`.

## Run
**Desktop app**
```bash
python detect_gui.py
```
Press **Ctrl+O** to open photos, use the arrow keys to browse, and adjust *Min. confidence* (default 25%).

**Command line**
```bash
python detect_cli.py --source dataset/test/images --conf 0.3
python detect_cli.py --source path/to/image.jpg
```

## Results
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| guns | – | – | – | – |
| knife | – | – | – | – |

*(Fill in from `runs/weapon/results.csv`.)*

## Screenshots
![GUI](docs/gui.png)

## Limitations
Accuracy depends on the dataset; unusual angles and look-alike objects (tools, phones, toys) can cause errors. Works on still images only.

## Credits & license
- Dataset: *Weapon 2* by joao-assalim-xmovq, **CC BY 4.0**
- Model: [Ultralytics YOLOv8](https://docs.ultralytics.com)
- Code license: MIT (add a `LICENSE` file)
