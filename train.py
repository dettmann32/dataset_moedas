import os
import subprocess
import sys

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

YOLOV9_DIR = os.path.expanduser("~/yolov9")
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(YOLOV9_DIR):
    print(f"YOLOv9 not found at {YOLOV9_DIR}")
    print("Cloning YOLOv9...")
    subprocess.run(
        ["git", "clone", "https://github.com/WongKinYiu/yolov9", YOLOV9_DIR],
        check=True
    )
    subprocess.run(
        ["pip", "install", "-r", os.path.join(YOLOV9_DIR, "requirements.txt")],
        check=True
    )

data_yaml = os.path.join(DATASET_DIR, "data.yaml")

cmd = [
    sys.executable, os.path.join(YOLOV9_DIR, "train.py"),
    "--workers", "8",
    "--device", "cpu",
    "--batch-size", "16",
    "--data", data_yaml,
    "--img", "640",
    "--cfg", os.path.join(YOLOV9_DIR, "models/detect/yolov9-s.yaml"),
    "--weights", "",
    "--name", "detector_moedas",
    "--hyp", os.path.join(YOLOV9_DIR, "data/hyps/hyp.scratch-high.yaml"),
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
