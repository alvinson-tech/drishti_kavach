# 🚆 Drishti Kavach

**Drishti Kavach** is a real-time railway track safety system that combines semantic rail segmentation with obstacle detection to identify on-track hazards and trigger proximity-based alerts.

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| Rail segmentation | BiSeNetV2 (fine-tuned on RailSem19 + custom hardware dataset) |
| Object detection | YOLO11m |
| Deployment runtime | OpenCV, PyTorch |
| Monitoring dashboard | Streamlit |

---

## ✨ Key Features

- Real-time rail track segmentation and obstacle detection
- On-track / off-track classification with **KAVACH ALERT** system
- Multi-track labeling and proximity threshold visualization
- Hardware capture, dataset labeling, and fine-tuning pipeline
- Live Streamlit dashboard with metrics, alert state, and detection log
- Session report and snapshot export

---

## 🛠️ Setup (for new collaborators)

### 1. Clone the repository

```bash
git clone https://github.com/<org>/drishti-kavach.git
cd drishti-kavach
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the `weights/` folder and add model weights

The model weight files are **not included in the repository** (too large for GitHub) and must be obtained from the project lead.

First, create the folder:

```bash
mkdir weights
```

Then place the following files inside it (request these from the team):

| File | Purpose |
|---|---|
| `weights/bisenet_railsem19.pth` | BiSeNetV2 pretrained on RailSem19 |
| `weights/bisenet_hardware.pth` | BiSeNetV2 fine-tuned on hardware dataset |
| `weights/bisenet_hardware_last.pth` | Last checkpoint from hardware fine-tuning |

> **YOLO weights (`yolo11m.pt`)** — this file is downloaded **automatically** by `ultralytics` the first time you run the program, as long as you have an internet connection. You do not need to manually place it.

### 5. Auto-created folders (nothing to do)

The following folders and files are **created automatically** when you run the program for the first time. You do not need to create them manually:

| Folder / File | Created by |
|---|---|
| `snapshots/` | `main.py` / `main_hardware.py` on startup |
| `captures/` | `main.py` / `main_hardware.py` on startup |
| `static/session_data.json` | Written at runtime by the detection pipeline |
| `yolo11m.pt` | Downloaded automatically by `ultralytics` |

### 6. Run the system

Start the main detection pipeline:

```bash
python main.py
```

Start the Streamlit dashboard in a second terminal:

```bash
streamlit run dashboard.py
```

Open the dashboard at `http://localhost:8501`.

**OpenCV window controls:**
- `S` — save a snapshot
- `Q` — quit

---

## 📁 Repository Layout

```
drishti_kavach/
│
├── main.py                      # Primary detection pipeline (webcam / image)
├── main_hardware.py             # Hardware-specific deployment script
├── train_hardware.py            # BiSeNetV2 fine-tuning on hardware dataset
├── capture_hardware.py          # Hardware image capture utility
├── label_hardware.py            # Interactive segmentation labeling tool
├── dashboard.py                 # Streamlit monitoring dashboard
├── requirements.txt             # Python dependencies
├── README.md
├── .gitignore
│
├── docs/                        # System documentation (7 parts)
│   ├── 01_project_overview.md
│   ├── 02_datasets.md
│   ├── 03_bisenetv2.md
│   ├── 04_yolo11m.md
│   ├── 05_main_program.md
│   ├── 06_hardware_pipeline.md
│   └── 07_dashboard.md
│
├── test_images/                 # Sample railway test images (1.jpg – 9.jpg)
│
├── hardware_captures/           # Raw images captured from hardware prototype
│
├── hardware_dataset/            # Labeled segmentation dataset (hardware fine-tuning)
│   ├── images/
│   └── masks/
│
├── static/                      # Runtime JSON state for dashboard
│   └── session_data.json        # ← auto-generated at runtime, gitignored
│
└── models/
    └── rail_marking/            # Vendored BiSeNetV2 source (upstream: xmba15/rail_marking)
        ├── cfg/
        │   ├── bisenetv2_cfg.py          # Model config (num classes, input size, etc.)
        │   └── bisenetv2_ego_cfg.py
        └── rail_marking/                 # Core Python package
            ├── segmentation/
            │   ├── deploy/
            │   │   └── railtrack_segmentation_handler.py   # Inference handler
            │   ├── models/
            │   │   ├── bisenetv2.py      # BiSeNetV2 model architecture
            │   │   └── ohem_ce_loss.py   # Loss function
            │   ├── data_loader/          # Dataset loaders (used by train_hardware.py)
            │   └── trainer/              # Training loop
            ├── core/                     # RS19 class constants
            └── utils/                    # Shared utilities

# ── Gitignored (not in this repo) ─────────────────────────────────────────────
# weights/              → model .pth files — request from project lead
# yolo11m.pt            → auto-downloaded by ultralytics on first run
# venv/                 → Python virtual environment
# downloaded_datasets/  → RailSem19 dataset (7.4 GB)
# snapshots/            → auto-created at runtime
# captures/             → auto-created at runtime
# static/*.json         → auto-generated at runtime
```

---

## 📄 Documentation

Detailed documentation covering architecture, datasets, training pipelines, and deployment is available in the [`docs/`](docs/) directory.

---

## 🙏 Acknowledgements

The BiSeNetV2 rail segmentation backbone used in this project is based on the open-source work by **[@xmba15](https://github.com/xmba15)**:

> **rail_marking** — [https://github.com/xmba15/rail_marking](https://github.com/xmba15/rail_marking)

The source code has been vendored into `models/rail_marking/` and the model weights have been fine-tuned on a custom hardware dataset for this project.

---

## 📜 License

Private repository — all rights reserved.