from google.colab import drive

drive.mount('/content/drive')

import os

MODEL_DIR = "/content/drive/MyDrive/AthletePose3D_Project/models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Model directory:", MODEL_DIR)
print(os.listdir(MODEL_DIR))

import os
import gdown

# Permanent project location
MODEL_DIR = "/content/drive/MyDrive/AthletePose3D_Project/models"
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "TCPFormer_ap3d_81.pth.tr"
)

# Only download if it doesn't already exist
if os.path.exists(CHECKPOINT_PATH):
    print("Checkpoint already exists.")
    print(CHECKPOINT_PATH)
    print(f"Size: {os.path.getsize(CHECKPOINT_PATH) / (1024**2):.2f} MB")
else:
    print("Downloading TCPFormer checkpoint...")

    gdown.download(
        id="1H4tq_oHQkGbYqZxS_5SgOqNicRYA1zsI",
        output=CHECKPOINT_PATH,
        quiet=False
    )

    print("\nDownload complete.")
    print(CHECKPOINT_PATH)
    print(f"Size: {os.path.getsize(CHECKPOINT_PATH) / (1024**2):.2f} MB")

import torch
import os

CHECKPOINT_PATH = "/content/drive/MyDrive/AthletePose3D_Project/models/TCPFormer_ap3d_81.pth.tr"

print("Checkpoint exists:", os.path.exists(CHECKPOINT_PATH))
print(f"Checkpoint size: {os.path.getsize(CHECKPOINT_PATH) / (1024**2):.2f} MB")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint loaded successfully.")
print("Checkpoint keys:", checkpoint.keys())

# Remove the "module." prefix added by DataParallel

state_dict = checkpoint["model"]

state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}

model.load_state_dict(state_dict, strict=True)

model = model.cuda()
model.eval()

print("TCPFormer checkpoint loaded successfully.")
print("Device:", next(model.parameters()).device)
print("Model is ready for inference.")

import os

matches = []

for name in os.listdir("/content/drive/MyDrive"):
    if "AthletePose3D" in name:
        matches.append(name)

print("Matching folders/files:")
for name in matches:
    print(repr(name))

import os
import shutil

DRIVE_TCPFORMER = "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"
LOCAL_TCPFORMER = "/content/TCPFormer"

if not os.path.exists(LOCAL_TCPFORMER):
    shutil.copytree(DRIVE_TCPFORMER, LOCAL_TCPFORMER)
    print("TCPFormer repository restored.")
else:
    print("TCPFormer repository already exists.")

print("Path:", LOCAL_TCPFORMER)

import os

for root, dirs, files in os.walk("/content/TCPFormer/data"):
    for f in files:
        if f.endswith(".py"):
            print(os.path.join(root, f))

# Show the exact H36M preprocessing used by TCPFormer

for filepath in [
    "/content/TCPFormer/data/preprocess/h36m.py",
    "/content/TCPFormer/data/reader/h36m.py"
]:
    print("\n" + "=" * 80)
    print(filepath)
    print("=" * 80)

    with open(filepath, "r") as f:
        print(f.read())

import os

matches = []

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for f in files:
        if f == "data.zip":
            matches.append(os.path.join(root, f))

print("data.zip files found:", len(matches))

for p in matches:
    print(p)

%cd /content

!git clone https://github.com/calvinyeungck/AthletePose3D.git

import os

REPO_DIR = "/content/AthletePose3D"

print("Repository exists:", os.path.exists(REPO_DIR))

if os.path.exists(REPO_DIR):
    print("Files:")
    print(os.listdir(REPO_DIR))

from pathlib import Path

REPO_DIR = Path("/content/AthletePose3D")

for file in [
    REPO_DIR / "pose_2d" / "README.md",
    REPO_DIR / "pose_2d" / "ap2d.py",
    REPO_DIR / "pose_2d" / "moganet_b_ap2d_384x288.py",
]:
    print("\n" + "="*80)
    print(file)
    print("="*80)
    print(file.read_text(errors="ignore")[:12000])

%cd /content/AthletePose3D

!find . -maxdepth 3 -type f | head -100

%cd /content/AthletePose3D

!cat README.md

!cat pose_2d/README.md

!cat pose_3d/README.md

import zipfile
import pickle
import io
import numpy as np
import matplotlib.pyplot as plt

zip_path = '/content/drive/MyDrive/AthletePose3D /pose_3d.zip'

file_name = 'pose_3d_v3/frame_81/train/00000000.pkl'

with zipfile.ZipFile(zip_path, 'r') as z:
    data = z.read(file_name)
    sample = pickle.load(io.BytesIO(data))

pose_3d = sample['data_label']

print("Pose shape:", pose_3d.shape)

from mpl_toolkits.mplot3d import Axes3D

frame = pose_3d[0]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    frame[:, 0],
    frame[:, 1],
    frame[:, 2]
)

for i in range(17):
    ax.text(
        frame[i, 0],
        frame[i, 1],
        frame[i, 2],
        str(i)
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

%cd /content/AthletePose3D

!cat utils/mapping.py

%cd /content/AthletePose3D

!cat utils/rig.json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

frame = pose_3d[0]

# Human3.6M skeleton connections
connections = [
    (0, 1), (1, 2), (2, 3),       # Right leg
    (0, 4), (4, 5), (5, 6),       # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Body/head

    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 14), (14, 15), (15, 16)   # Right arm
]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot joints
ax.scatter(
    frame[:, 0],
    frame[:, 1],
    frame[:, 2],
    s=50
)

# Connect joints
for start, end in connections:
    ax.plot(
        [frame[start, 0], frame[end, 0]],
        [frame[start, 1], frame[end, 1]],
        [frame[start, 2], frame[end, 2]]
    )

# Label joints
for i, (x, y, z) in enumerate(frame):
    ax.text(x, y, z, str(i))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("AthletePose3D - 3D Human Skeleton")

plt.show()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

connections = [
    (0, 1), (1, 2), (2, 3),       # Right leg
    (0, 4), (4, 5), (5, 6),       # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Body/head
    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 14), (14, 15), (15, 16)   # Right arm
]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame_number):

    ax.clear()

    frame = pose_3d[frame_number]

    # Draw joints
    ax.scatter(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        s=50
    )

    # Draw skeleton
    for start, end in connections:
        ax.plot(
            [frame[start, 0], frame[end, 0]],
            [frame[start, 1], frame[end, 1]],
            [frame[start, 2], frame[end, 2]]
        )

    # Labels
    for i, (x, y, z) in enumerate(frame):
        ax.text(x, y, z, str(i))

    ax.set_xlim(
        pose_3d[:, :, 0].min(),
        pose_3d[:, :, 0].max()
    )

    ax.set_ylim(
        pose_3d[:, :, 1].min(),
        pose_3d[:, :, 1].max()
    )

    ax.set_zlim(
        pose_3d[:, :, 2].min(),
        pose_3d[:, :, 2].max()
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        f"AthletePose3D — Frame {frame_number + 1}/81"
    )

animation = FuncAnimation(
    fig,
    update,
    frames=81,
    interval=80
)

HTML(animation.to_jshtml())

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

connections = [
    (0, 1), (1, 2), (2, 3),       # Right leg
    (0, 4), (4, 5), (5, 6),       # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Body/head
    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 14), (14, 15), (15, 16)   # Right arm
]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame_number):

    ax.clear()

    frame = pose_3d[frame_number]

    # Draw joints
    ax.scatter(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        s=50
    )

    # Draw skeleton
    for start, end in connections:
        ax.plot(
            [frame[start, 0], frame[end, 0]],
            [frame[start, 1], frame[end, 1]],
            [frame[start, 2], frame[end, 2]]
        )

    # Labels
    for i, (x, y, z) in enumerate(frame):
        ax.text(x, y, z, str(i))

    ax.set_xlim(
        pose_3d[:, :, 0].min(),
        pose_3d[:, :, 0].max()
    )

    ax.set_ylim(
        pose_3d[:, :, 1].min(),
        pose_3d[:, :, 1].max()
    )

    ax.set_zlim(
        pose_3d[:, :, 2].min(),
        pose_3d[:, :, 2].max()
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        f"AthletePose3D — Frame {frame_number + 1}/81"
    )

animation = FuncAnimation(
    fig,
    update,
    frames=81,
    interval=80
)

HTML(animation.to_jshtml())

import zipfile
import json

zip_path = '/content/drive/MyDrive/AthletePose3D /data.zip'

with zipfile.ZipFile(zip_path, 'r') as z:

    json_files = [
        f for f in z.namelist()
        if f.endswith('.json')
    ]

    print("Number of JSON files:", len(json_files))
    print("\nFirst 10:")

    for f in json_files[:10]:
        print(f)

with zipfile.ZipFile(zip_path, 'r') as z:

    file_name = json_files[0]

    content = z.read(file_name).decode('utf-8')
    metadata = json.loads(content)

print(json.dumps(metadata, indent=2)[:5000])

import zipfile
import json

zip_path = '/content/drive/MyDrive/AthletePose3D /data.zip'

file_name = 'data/test_set/S1/Axel_10_cam_1.json'

with zipfile.ZipFile(zip_path, 'r') as z:
    content = z.read(file_name).decode('utf-8')
    metadata = json.loads(content)

print("Type:", type(metadata))

if isinstance(metadata, dict):
    print("\nKeys:")
    for key in metadata.keys():
        print(" -", key)

print("\nMetadata:")
print(json.dumps(metadata, indent=2)[:5000])

import zipfile

zip_path = '/content/drive/MyDrive/AthletePose3D /data.zip'

with zipfile.ZipFile(zip_path, 'r') as z:
    files = z.namelist()

print("Total files:", len(files))

print("\nTop-level structure:")

seen = set()

for f in files:
    parts = f.split('/')

    if len(parts) >= 3:
        path = '/'.join(parts[:3])

        if path not in seen:
            seen.add(path)
            print(path)

import zipfile
import os

zip_path = '/content/drive/MyDrive/AthletePose3D /data.zip'

with zipfile.ZipFile(zip_path, 'r') as z:
    json_files = [
        f for f in z.namelist()
        if f.endswith('.json')
    ]

# Extract source video names directly from filenames
source_names = set()

for f in json_files:
    name = os.path.basename(f)

    # Example:
    # Axel_10_cam_1.json
    # -> Axel_10
    parts = name.replace('.json', '').split('_cam_')

    if len(parts) == 2:
        source_names.add(parts[0])

source_names = sorted(source_names)

print("Unique motion/video names:", len(source_names))

for name in source_names:
    print(name)

import zipfile

zip_path = '/content/drive/MyDrive/AthletePose3D /data.zip'

with zipfile.ZipFile(zip_path, 'r') as z:

    running_files = [
        f for f in z.namelist()
        if 'Running_0' in f
    ]

print("Files for Running_0:")
for f in running_files:
    print(f)

import zipfile
import numpy as np
import io

zip_path = '/content/drive/MyDrive/AthletePose3D /data.zip'

file_name = 'data/train_set/S3/Running_0_cam_1_h36m.npy'

with zipfile.ZipFile(zip_path, 'r') as z:

    data = z.read(file_name)

    pose = np.load(io.BytesIO(data), allow_pickle=True)

print("Type:", type(pose))
print("Shape:", pose.shape)
print("Dtype:", pose.dtype)

print(pose[:2])

import zipfile
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

zip_path = '/content/drive/MyDrive/AthletePose3D /data.zip'

file_name = 'data/train_set/S3/Running_0_cam_1_h36m.npy'

with zipfile.ZipFile(zip_path, 'r') as z:
    data = z.read(file_name)

pose_running = np.load(
    io.BytesIO(data),
    allow_pickle=True
)

print("Running sequence shape:", pose_running.shape)

connections = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame_number):

    ax.clear()

    frame = pose_running[frame_number]

    # joints
    ax.scatter(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        s=40
    )

    # skeleton
    for start, end in connections:
        ax.plot(
            [frame[start, 0], frame[end, 0]],
            [frame[start, 1], frame[end, 1]],
            [frame[start, 2], frame[end, 2]]
        )

    # Keep same scale throughout animation
    ax.set_xlim(
        pose_running[:, :, 0].min(),
        pose_running[:, :, 0].max()
    )

    ax.set_ylim(
        pose_running[:, :, 1].min(),
        pose_running[:, :, 1].max()
    )

    ax.set_zlim(
        pose_running[:, :, 2].min(),
        pose_running[:, :, 2].max()
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        f"Running_0 — Frame {frame_number + 1}/{len(pose_running)}"
    )

animation = FuncAnimation(
    fig,
    update,
    frames=len(pose_running),
    interval=50
)

HTML(animation.to_jshtml())

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Use every 2nd frame to keep the animation small
pose_anim = pose_running[::2]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()

    pose = pose_anim[frame]

    # Plot joints
    ax.scatter(
        pose[:, 0],
        pose[:, 1],
        pose[:, 2],
        s=30
    )

    # Connect joints
    for i, j in connections:
        ax.plot(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            [pose[i, 2], pose[j, 2]]
        )

    ax.set_xlim(
        pose_running[:, :, 0].min(),
        pose_running[:, :, 0].max()
    )
    ax.set_ylim(
        pose_running[:, :, 1].min(),
        pose_running[:, :, 1].max()
    )
    ax.set_zlim(
        pose_running[:, :, 2].min(),
        pose_running[:, :, 2].max()
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Running - Frame {frame * 2}")

ani = FuncAnimation(
    fig,
    update,
    frames=len(pose_anim),
    interval=50
)

plt.close(fig)

HTML(ani.to_jshtml())

import os

test_path = "/content/drive/MyDrive/colab_write_test.txt"

try:
    with open(test_path, "w") as f:
        f.write("test")

    print("WRITE ACCESS: YES")
    os.remove(test_path)

except Exception as e:
    print("WRITE ACCESS: NO")
    print(type(e).__name__, e)

import os

# Existing dataset — READ ONLY
DATASET_DIR = "/content/drive/MyDrive/AthletePose3D "

# Our own permanent project folder
PROJECT_DIR = "/content/drive/MyDrive/AthletePose3D_Project"

folders = [
    "videos",
    "poses_2d",
    "poses_3d",
    "features",
    "references",
    "results",
    "models"
]

os.makedirs(PROJECT_DIR, exist_ok=True)

for folder in folders:
    os.makedirs(os.path.join(PROJECT_DIR, folder), exist_ok=True)

print("PROJECT CREATED SUCCESSFULLY")
print()
print("Dataset :", DATASET_DIR)
print("Project :", PROJECT_DIR)
print()

for folder in folders:
    print("✓", folder)

import os

print(os.listdir(PROJECT_DIR))

import os

REPO_DIR = "/content/AthletePose3D"

print("Repository exists:", os.path.exists(REPO_DIR))

print("\nRepository contents:")
for item in os.listdir(REPO_DIR):
    print(" ", item)

print("\n2D pose folder:")
for item in os.listdir(os.path.join(REPO_DIR, "pose_2d")):
    print(" ", item)

print("\nAvailable Python files:")
for root, dirs, files in os.walk(REPO_DIR):
    for file in files:
        if file.endswith(".py"):
            print(os.path.relpath(os.path.join(root, file), REPO_DIR))

import torch
import os

MODEL_PATH = os.path.join(
    DATASET_DIR,
    "model_params",
    "TCPFormer_ap3d_81.pth.tr"
)

print("Model exists:", os.path.exists(MODEL_PATH))
print("Model path:", MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(" ", key)
else:
    print("\nCheckpoint object:")
    print(checkpoint)

# Inspect the important parts of the official AthletePose3D 3D pipeline

from pathlib import Path

files_to_check = [
    "/content/AthletePose3D/pose_3d/datareader_ap3d.py",
    "/content/AthletePose3D/pose_3d/convert_ap3d.py",
    "/content/AthletePose3D/utils/demo.py"
]

for file_path in files_to_check:
    print("\n" + "=" * 80)
    print(Path(file_path).name)
    print("=" * 80)

    with open(file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        print(f"{i:4}: {line.rstrip()}")

import torch
import os

AP2D_PATH = os.path.join(
    DATASET_DIR,
    "model_params",
    "moganet_b_ap2d_384x288.pth"
)

print("2D model exists:", os.path.exists(AP2D_PATH))
print("Path:", AP2D_PATH)

checkpoint_2d = torch.load(
    AP2D_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint type:", type(checkpoint_2d))

if isinstance(checkpoint_2d, dict):
    print("\nCheckpoint keys:")
    for key in checkpoint_2d.keys():
        print(" ", key)

import importlib.util

packages = [
    "torch",
    "torchvision",
    "cv2",
    "mmcv",
    "mmengine",
    "mmpose",
    "mmdet"
]

print("Package availability:\n")

for package in packages:
    print(
        f"{package:12} :",
        "INSTALLED" if importlib.util.find_spec(package) else "NOT INSTALLED"
    )

import torch

print("Python version:")
import sys
print(sys.version)

print("\nPyTorch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

DATASET_DIR = "/content/drive/MyDrive/AthletePose3D "
PROJECT_DIR = "/content/drive/MyDrive/AthletePose3D_Project"
REPO_DIR = "/content/AthletePose3D"

print("Paths restored.")

!pip install -q -U openmim
!mim install "mmengine"
!mim install "mmcv"
!pip install -q "mmpose"

import subprocess

result = subprocess.run(
    ["/usr/bin/python3.12", "--version"],
    capture_output=True,
    text=True
)

print(result.stdout)

import subprocess
import sys

result = subprocess.run(
    [
        "/usr/bin/python3.12",
        "-m",
        "ensurepip",
        "--upgrade"
    ],
    capture_output=True,
    text=True
)

print("STDOUT:")
print(result.stdout)

print("STDERR:")
print(result.stderr)

import subprocess

result = subprocess.run(
    [
        "/usr/bin/python3.12",
        "-m",
        "pip",
        "--version"
    ],
    capture_output=True,
    text=True
)
print(result.stdout)
print(result.stderr)

import os

print(os.path.exists("/content/AthletePose3D"))
print(os.listdir("/content")[:20])

import zipfile
import os

ZIP_PATH = "/content/drive/MyDrive/AthletePose3D /data.zip"
OUT_DIR = "/content/drive/MyDrive/AthletePose3D_Project/references/Running_0"

os.makedirs(OUT_DIR, exist_ok=True)

target = "data/train_set/S3/Running_0_cam_1_h36m.npy"

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    print("File exists in ZIP:", target in z.namelist())

    if target in z.namelist():
        z.extract(target, OUT_DIR)

print("\nExtracted files:")
for root, dirs, files in os.walk(OUT_DIR):
    for f in files:
        print(os.path.join(root, f))

import numpy as np
import os

running_path = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "references/Running_0/data/train_set/S3/Running_0_cam_1_h36m.npy"
)

pose_running = np.load(running_path)

print("Shape:", pose_running.shape)
print("dtype:", pose_running.dtype)

import zipfile
import os

ZIP_PATH = "/content/drive/MyDrive/AthletePose3D /data.zip"
OUT_DIR = "/content/drive/MyDrive/AthletePose3D_Project/references/Running_0_2d"

os.makedirs(OUT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    matches = [
        f for f in z.namelist()
        if "Running_0_cam_1" in f
        and f.endswith(".npy")
    ]

print("Matching files:")
for f in matches:
    print(f)

import zipfile
import os
import numpy as np

ZIP_PATH = "/content/drive/MyDrive/AthletePose3D /data.zip"
OUT_DIR = "/content/drive/MyDrive/AthletePose3D_Project/references/Running_0_2d"

os.makedirs(OUT_DIR, exist_ok=True)

target = "data/train_set/S3/Running_0_cam_1_coco.npy"

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extract(target, OUT_DIR)

print("Extracted successfully.")

import numpy as np
import os

path_2d = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "references/Running_0_2d/data/train_set/S3/"
    "Running_0_cam_1_coco.npy"
)

pose_2d = np.load(path_2d)

print("Shape:", pose_2d.shape)
print("dtype:", pose_2d.dtype)
print("First frame:")
print(pose_2d[0])

import os

REPO_DIR = "/content/AthletePose3D"

print("pose_3d files:")
for f in os.listdir(os.path.join(REPO_DIR, "pose_3d")):
    print(f)

import os

for root, dirs, files in os.walk("/content/AthletePose3D"):
    for f in files:
        if f.endswith(".py") and ("model" in f.lower() or "tcp" in f.lower()):
            print(os.path.join(root, f))

import os

REPO_DIR = "/content/AthletePose3D"

print("Searching repository for TCPFormer...")

for root, dirs, files in os.walk(REPO_DIR):
    for f in files:
        if f.endswith((".py", ".yaml", ".yml")):
            path = os.path.join(root, f)

            try:
                text = open(path, "r", errors="ignore").read()

                if "TCPFormer" in text or "tcpformer" in text.lower():
                    print("\nFOUND:", path)

                    # Print the relevant lines
                    for i, line in enumerate(text.splitlines(), 1):
                        if "TCPFormer" in line or "tcpformer" in line.lower():
                            print(f"{i}: {line}")
            except:
                pass

print(open("/content/AthletePose3D/pose_3d/README.md", errors="ignore").read())

import os

TCP_DRIVE = "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"

if os.path.exists(TCP_DRIVE):
    print("TCPFormer already exists in Drive.")
    print(os.listdir(TCP_DRIVE))
else:
    !git clone https://github.com/AsukaCamellia/TCPFormer.git "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"
    print("TCPFormer cloned to Drive.")

import os

TCP_DRIVE = "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"

print("Exists:", os.path.exists(TCP_DRIVE))
print("Contents:")
print(os.listdir(TCP_DRIVE))

import os

TCP_DIR = "/content/TCPFormer"

print("TCPFormer exists:", os.path.exists(TCP_DIR))
print(os.listdir(TCP_DIR))

import os
import shutil

TCP_DRIVE = "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"
TCP_DIR = "/content/TCPFormer"

if os.path.exists(TCP_DRIVE):
    print("TCPFormer already exists in Drive.")
else:
    shutil.copytree(TCP_DIR, TCP_DRIVE)
    print("TCPFormer copied permanently to Drive.")

print("Drive copy exists:", os.path.exists(TCP_DRIVE))

TCP_DRIVE = "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"

import os
import shutil

TCP_DRIVE = "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"
TCP_DIR = "/content/TCPFormer"

if not os.path.exists(TCP_DIR):
    shutil.copytree(TCP_DRIVE, TCP_DIR)
    print("TCPFormer restored from Drive.")
else:
    print("TCPFormer already available.")

print("Ready:", os.path.exists(TCP_DIR))

import os

TCP_DIR = "/content/TCPFormer"

print("MODEL FILES")
print("=" * 60)

for root, dirs, files in os.walk(os.path.join(TCP_DIR, "model")):
    for f in files:
        print(os.path.join(root, f))

print("\nCONFIG FILES")
print("=" * 60)

for root, dirs, files in os.walk(os.path.join(TCP_DIR, "configs")):
    for f in files:
        print(os.path.join(root, f))

CONFIG_PATH = "/content/TCPFormer/configs/h36m/TCPFormer_h36m_81.yaml"

print(open(CONFIG_PATH, "r").read())

MODEL_PATH = "/content/TCPFormer/model/Model.py"

text = open(MODEL_PATH, "r", errors="ignore").read()

start = text.find("class MemoryInducedTransformer")
print(text[start:])

import re

TRAIN_PATH = "/content/TCPFormer/train.py"

text = open(TRAIN_PATH, "r", errors="ignore").read()

for i, line in enumerate(text.splitlines(), 1):
    if any(x in line.lower() for x in [
        "torch.load",
        "load_state_dict",
        "memoryinducedtransformer",
        "model =",
        "checkpoint"
    ]):
        print(f"{i}: {line}")

for i, line in enumerate(text.splitlines(), 1):
    if any(x in line.lower() for x in [
        "yaml",
        "config",
        "cfg",
        "model_name"
    ]):
        print(f"{i}: {line}")

import os

for root, dirs, files in os.walk("/content/TCPFormer"):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            text = open(path, "r", errors="ignore").read()

            if "def load_model_TCPFormer" in text:
                print("\nFOUND:", path)
                start = text.find("def load_model_TCPFormer")
                print(text[start:start+5000])

import os
import sys
import torch
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------
TCP_DIR = "/content/TCPFormer"

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/AthletePose3D "
    "/model_params/TCPFormer_ap3d_81.pth.tr"
)

CONFIG_PATH = (
    TCP_DIR +
    "/configs/h36m/TCPFormer_h36m_81.yaml"
)

# --------------------------------------------------
# Add TCPFormer to Python path
# --------------------------------------------------
if TCP_DIR not in sys.path:
    sys.path.insert(0, TCP_DIR)

# --------------------------------------------------
# Load config
# --------------------------------------------------
from utils.tools import get_config
from utils.learning import load_model_TCPFormer

args = get_config(CONFIG_PATH)

print("Model:", args.model_name)
print("Frames:", args.n_frames)
print("Joints:", args.num_joints)
print("Input channels:", args.dim_in)

# --------------------------------------------------
# Create model
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model_TCPFormer(args)
model = model.to(device)

print("\nModel created.")
print("Device:", device)

# --------------------------------------------------
# Load pretrained AthletePose3D checkpoint
# --------------------------------------------------
print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

print("Checkpoint keys:", checkpoint.keys())

model.load_state_dict(
    checkpoint["model"],
    strict=True
)

model.eval()

print("\nTCPFormer checkpoint loaded successfully.")

import os

PROJECT_DIR = "/content/drive/MyDrive/AthletePose3D_Project"

for root, dirs, files in os.walk(PROJECT_DIR):
    level = root.replace(PROJECT_DIR, "").count(os.sep)

    if level <= 3:
        print("\n" + root)
        for f in files[:20]:
            print("   ", f)

import os
import shutil

# Remove the old empty/stale mountpoint
if os.path.exists("/content/drive"):
    shutil.rmtree("/content/drive")

# Mount Google Drive again
from google.colab import drive
drive.mount("/content/drive")

import os

print("=== My Drive ===")
for item in os.listdir("/content/drive/MyDrive"):
    print(item)

import os

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

folders = [
    "code/preprocessing",
    "code/pose_estimation",
    "code/pose_3d",
    "code/features",
    "code/technique_analysis",
    "code/llm",

    "data/videos",
    "data/poses_2d",
    "data/poses_3d",
    "data/features",
    "data/references",

    "results/visualizations",
    "results/technique_reports",
    "results/experiments",

    "models/pretrained",
    "models/trained",

    "notebooks",

    "config",
]

for folder in folders:
    path = os.path.join(PROJECT, folder)
    os.makedirs(path, exist_ok=True)

print("Project structure created successfully.")
print("\nLocation:")
print(PROJECT)

import os

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

for root, dirs, files in os.walk(PROJECT):
    level = root.replace(PROJECT, "").count(os.sep)

    if level > 2:
        dirs[:] = []
        continue

    indent = "    " * level
    print(f"{indent}{os.path.basename(root)}/")

import zipfile
import os

DATA_ZIP = "/content/drive/MyDrive/AthletePose3D /data.zip"

with zipfile.ZipFile(DATA_ZIP, "r") as z:
    running_files = [
        name for name in z.namelist()
        if "Running_0" in name
    ]

print("Number of Running_0 files:", len(running_files))

print("\nFirst 30 files:")
for name in running_files[:30]:
    print(name)

import os
import zipfile

ORIGINAL = "/content/drive/MyDrive/AthletePose3D "
PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

DATA_ZIP = os.path.join(ORIGINAL, "data.zip")
OUTPUT_DIR = os.path.join(PROJECT, "data", "references", "Running_0", "S3")

os.makedirs(OUTPUT_DIR, exist_ok=True)

files_to_extract = [
    "data/train_set/S3/Running_0_cam_1.json",
    "data/train_set/S3/Running_0_cam_1.mp4",
    "data/train_set/S3/Running_0_cam_1.npy",
    "data/train_set/S3/Running_0_cam_1_coco.npy",
    "data/train_set/S3/Running_0_cam_1_h36m.npy",
]

with zipfile.ZipFile(DATA_ZIP, "r") as z:
    for file in files_to_extract:
        print("Extracting:", file)
        z.extract(file, OUTPUT_DIR)

print("\nExtraction complete.")

import os
import shutil
import numpy as np

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

# Extracted location
EXTRACTED = os.path.join(
    PROJECT,
    "data",
    "references",
    "Running_0",
    "S3",
    "data",
    "train_set",
    "S3"
)

# Clean permanent location
RUNNING = os.path.join(
    PROJECT,
    "data",
    "references",
    "Running_0",
    "S3",
    "cam_1"
)

os.makedirs(RUNNING, exist_ok=True)

files = [
    "Running_0_cam_1.json",
    "Running_0_cam_1.mp4",
    "Running_0_cam_1.npy",
    "Running_0_cam_1_coco.npy",
    "Running_0_cam_1_h36m.npy",
]

for filename in files:
    src = os.path.join(EXTRACTED, filename)
    dst = os.path.join(RUNNING, filename)

    if os.path.exists(dst):
        print("Already exists:", filename)
    elif os.path.exists(src):
        shutil.move(src, dst)
        print("Moved:", filename)
    else:
        print("MISSING:", filename)

# Remove empty extraction folders if possible
try:
    shutil.rmtree(
        os.path.join(
            PROJECT,
            "data",
            "references",
            "Running_0",
            "S3",
            "data"
        )
    )
except:
    pass

print("\n=== Final files ===")

for filename in os.listdir(RUNNING):
    path = os.path.join(RUNNING, filename)
    size_mb = os.path.getsize(path) / (1024**2)
    print(f"{filename:35s} {size_mb:.2f} MB")

import numpy as np
import os

RUNNING = "/content/drive/MyDrive/AthletePose3D_Project/data/references/Running_0/S3/cam_1"

for filename in [
    "Running_0_cam_1.npy",
    "Running_0_cam_1_coco.npy",
    "Running_0_cam_1_h36m.npy"
]:
    path = os.path.join(RUNNING, filename)

    data = np.load(path)

    print(f"{filename}")
    print(f"  Shape : {data.shape}")
    print(f"  Dtype : {data.dtype}")
    print()

import os

REPO = "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"
CHECKPOINT = "/content/drive/MyDrive/AthletePose3D_Project/models/TCPFormer_ap3d_81.pth.tr"
CONFIG = os.path.join(REPO, "configs/h36m/TCPFormer_h36m_81.yaml")

print("TCPFormer repo:", os.path.exists(REPO))
print("Checkpoint:", os.path.exists(CHECKPOINT))
print("Config:", os.path.exists(CONFIG))

print("\nCheckpoint size:")
if os.path.exists(CHECKPOINT):
    print(f"{os.path.getsize(CHECKPOINT)/(1024**3):.2f} GB")

import sys
import os

REPO = "/content/drive/MyDrive/AthletePose3D_Project/TCPFormer"

if REPO not in sys.path:
    sys.path.insert(0, REPO)

print("TCPFormer added to Python path.")

import os
import shutil

ORIGINAL = "/content/drive/MyDrive/AthletePose3D "
PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

src = os.path.join(
    ORIGINAL,
    "model_params",
    "TCPFormer_ap3d_81.pth.tr"
)

dst = os.path.join(
    PROJECT,
    "models",
    "pretrained",
    "TCPFormer_ap3d_81.pth.tr"
)

os.makedirs(os.path.dirname(dst), exist_ok=True)

if os.path.exists(dst):
    print("Checkpoint already exists. Skipping copy.")
else:
    print("Copying checkpoint...")
    shutil.copy2(src, dst)
    print("Copy complete.")

print("\nCheckpoint:")
print(dst)
print(f"Size: {os.path.getsize(dst)/(1024**3):.2f} GB")

import os
import subprocess

TCP_REPO = "/content/drive/MyDrive/AthletePose3D_Project/code/TCPFormer"

if os.path.exists(os.path.join(TCP_REPO, "model", "Model.py")):
    print("TCPFormer repository already exists. Skipping clone.")
else:
    print("Cloning TCPFormer...")
    subprocess.run([
        "git", "clone",
        "https://github.com/AsukaCamellia/TCPFormer.git",
        TCP_REPO
    ], check=True)
    print("Clone complete.")

print("\nTCPFormer location:")
print(TCP_REPO)

import os

TCP_REPO = "/content/drive/MyDrive/AthletePose3D_Project/code/TCPFormer"

required = [
    "model/Model.py",
    "utils/learning.py",
    "configs/h36m/TCPFormer_h36m_81.yaml",
    "train.py",
]

for file in required:
    path = os.path.join(TCP_REPO, file)
    print(f"{'OK' if os.path.exists(path) else 'MISSING'}  {file}")

import sys
import os
import torch
import yaml
from types import SimpleNamespace

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

TCP_REPO = os.path.join(PROJECT, "code", "TCPFormer")
CHECKPOINT_PATH = os.path.join(
    PROJECT,
    "models",
    "pretrained",
    "TCPFormer_ap3d_81.pth.tr"
)
CONFIG_PATH = os.path.join(
    TCP_REPO,
    "configs",
    "h36m",
    "TCPFormer_h36m_81.yaml"
)

# Make TCPFormer importable
if TCP_REPO not in sys.path:
    sys.path.insert(0, TCP_REPO)

from model.Model import MemoryInducedTransformer

# Read official configuration
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

args = SimpleNamespace(**config)

# Build exact architecture matching the checkpoint
model = MemoryInducedTransformer(
    n_layers=args.n_layers,
    dim_in=args.dim_in,
    dim_feat=args.dim_feat,
    dim_rep=args.dim_rep,
    dim_out=args.dim_out,
    mlp_ratio=args.mlp_ratio,
    act_layer=torch.nn.GELU,
    attn_drop=args.attn_drop,
    drop=args.drop,
    drop_path=args.drop_path,
    use_layer_scale=args.use_layer_scale,
    layer_scale_init_value=args.layer_scale_init_value,
    use_adaptive_fusion=args.use_adaptive_fusion,
    num_heads=args.num_heads,
    qkv_bias=args.qkv_bias,
    qkv_scale=args.qkv_scale,
    hierarchical=args.hierarchical,
    num_joints=args.num_joints,
    use_temporal_similarity=args.use_temporal_similarity,
    temporal_connection_len=args.temporal_connection_len,
    use_tcn=args.use_tcn,
    graph_only=args.graph_only,
    neighbour_num=args.neighbour_num,
    n_frames=args.n_frames
)

# Load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

state_dict = checkpoint["model"]

# Remove DataParallel "module." prefix
state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}

model.load_state_dict(state_dict, strict=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

print("================================")
print("TCPFormer loaded successfully")
print("================================")
print("Device :", device)
print("Frames :", args.n_frames)
print("Joints :", args.num_joints)
print("Input  :", args.dim_in)
print("Output :", args.dim_out)

import os
import subprocess

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"
AP3D_REPO = os.path.join(PROJECT, "code", "AthletePose3D")

# Remove incomplete clone if one was created
if os.path.exists(AP3D_REPO):
    import shutil
    shutil.rmtree(AP3D_REPO)

print("Cloning official AthletePose3D repository...")

subprocess.run([
    "git", "clone",
    "https://github.com/calvinyeungck/AthletePose3D.git",
    AP3D_REPO
], check=True)

print("Clone complete.")
print(AP3D_REPO)

import os

AP3D_REPO = "/content/drive/MyDrive/AthletePose3D_Project/code/AthletePose3D"

files = [
    "utils/mapping.py",
    "pose_3d/convert_ap3d.py",
    "pose_3d/datareader_ap3d.py"
]

for relative_path in files:
    path = os.path.join(AP3D_REPO, relative_path)

    print("\n" + "=" * 80)
    print(relative_path)
    print("=" * 80)

    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    # Show the first 250 lines
    for i, line in enumerate(lines[:250], start=1):
        print(f"{i:3}: {line}", end="")

import cv2
import os
import numpy as np

RUNNING = "/content/drive/MyDrive/AthletePose3D_Project/data/references/Running_0/S3/cam_1"

VIDEO = os.path.join(RUNNING, "Running_0_cam_1.mp4")
POSE2D = os.path.join(RUNNING, "Running_0_cam_1_coco.npy")

cap = cv2.VideoCapture(VIDEO)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

pose_2d = np.load(POSE2D)

print("Video width :", width)
print("Video height:", height)
print("Video frames:", frames)
print("Video FPS   :", fps)

print("\n2D pose shape:", pose_2d.shape)

import cv2
import os
import numpy as np

RUNNING = "/content/drive/MyDrive/AthletePose3D_Project/data/references/Running_0/S3/cam_1"

VIDEO = os.path.join(RUNNING, "Running_0_cam_1.mp4")
POSE2D = os.path.join(RUNNING, "Running_0_cam_1_coco.npy")

cap = cv2.VideoCapture(VIDEO)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

pose_2d = np.load(POSE2D)

print("Video width :", width)
print("Video height:", height)
print("Video frames:", frames)
print("Video FPS   :", fps)

print("\n2D pose shape:", pose_2d.shape)

import numpy as np
import os

# Load COCO 2D pose
pose_2d = np.load(POSE2D).astype(np.float32)

# Normalize exactly according to AthletePose3D/TCPFormer preprocessing
pose_norm = pose_2d.copy()

pose_norm[:, :, 0] = pose_norm[:, :, 0] / width * 2 - 1
pose_norm[:, :, 1] = pose_norm[:, :, 1] / width * 2 - (height / width)

# Confidence channel
confidence = np.ones(
    (pose_norm.shape[0], pose_norm.shape[1], 1),
    dtype=np.float32
)

# [153,17,2] → [153,17,3]
tcp_input = np.concatenate(
    [pose_norm, confidence],
    axis=2
)

print("Original:", pose_2d.shape)
print("TCPFormer input:", tcp_input.shape)
print("dtype:", tcp_input.dtype)

print("\nFirst frame:")
print(tcp_input[0])

window = tcp_input[:81]

print("Window shape:", window.shape)

import torch

x = torch.from_numpy(window).unsqueeze(0).to(device)

print("Input tensor:", x.shape)

with torch.no_grad():
    prediction = model(x)

print("Prediction shape:", prediction.shape)

import numpy as np
import torch
import os

RUNNING = "/content/drive/MyDrive/AthletePose3D_Project/data/references/Running_0/S3/cam_1"

# Load reference 3D pose
reference_3d = np.load(
    os.path.join(RUNNING, "Running_0_cam_1_h36m.npy")
).astype(np.float32)

# Convert TCPFormer prediction to NumPy
predicted_3d = prediction[0].detach().cpu().numpy()

print("Reference shape :", reference_3d.shape)
print("Prediction shape:", predicted_3d.shape)

import numpy as np
import os

RUNNING = "/content/drive/MyDrive/AthletePose3D_Project/data/references/Running_0/S3/cam_1"

pose_2d = np.load(
    os.path.join(RUNNING, "Running_0_cam_1_coco.npy")
).astype(np.float32)

nan_count = np.isnan(pose_2d).sum()
total_values = pose_2d.size

print("2D pose shape:", pose_2d.shape)
print("NaN values:", nan_count)
print("Total values:", total_values)
print("NaN percentage:", 100 * nan_count / total_values)

import numpy as np

def interpolate_missing_pose(pose):
    """
    Temporally interpolate missing 2D joint coordinates.
    pose shape: [frames, joints, 2]
    """
    pose = pose.copy()

    n_frames, n_joints, n_coords = pose.shape
    frame_indices = np.arange(n_frames)

    for j in range(n_joints):
        for c in range(n_coords):

            values = pose[:, j, c]
            valid = np.isfinite(values)

            if valid.sum() == 0:
                # No information for this joint at all
                values[:] = 0
            elif valid.sum() == 1:
                # Only one valid point
                values[:] = values[valid][0]
            else:
                values[~valid] = np.interp(
                    frame_indices[~valid],
                    frame_indices[valid],
                    values[valid]
                )

            pose[:, j, c] = values

    return pose


pose_2d_clean = interpolate_missing_pose(pose_2d)

print("Original NaNs :", np.isnan(pose_2d).sum())
print("Cleaned NaNs  :", np.isnan(pose_2d_clean).sum())
print("Shape         :", pose_2d_clean.shape)

pose_norm = pose_2d_clean.copy()

pose_norm[:, :, 0] = (
    pose_norm[:, :, 0] / width * 2 - 1
)

pose_norm[:, :, 1] = (
    pose_norm[:, :, 1] / width * 2
    - (height / width)
)

confidence = np.ones(
    (pose_norm.shape[0], pose_norm.shape[1], 1),
    dtype=np.float32
)

tcp_input_clean = np.concatenate(
    [pose_norm, confidence],
    axis=2
)

window_clean = tcp_input_clean[:81]

print("TCPFormer input:", window_clean.shape)
print("NaNs:", np.isnan(window_clean).sum())

x = torch.from_numpy(window_clean).unsqueeze(0).to(device)

with torch.no_grad():
    prediction_clean = model(x)

predicted_3d = prediction_clean[0].detach().cpu().numpy()

print("Prediction:", predicted_3d.shape)
print("NaNs:", np.isnan(predicted_3d).sum())
print("Infs:", np.isinf(predicted_3d).sum())

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np

# AthletePose3D H36M joint connections
connections = [
    (0,1), (1,2), (2,3),       # Right leg
    (0,4), (4,5), (5,6),       # Left leg
    (0,7), (7,8), (8,9), (9,10), # Spine/head
    (8,11), (11,12), (12,13),  # Left arm
    (8,14), (14,15), (15,16)   # Right arm
]

pose = predicted_3d.copy()

# Root-relative
pose = pose - pose[:, 0:1, :]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="3d")

def update(frame):
    ax.clear()

    points = pose[frame]

    # Draw joints
    ax.scatter(
        points[:,0],
        points[:,1],
        points[:,2],
        s=20
    )

    # Draw skeleton
    for a, b in connections:
        ax.plot(
            [points[a,0], points[b,0]],
            [points[a,1], points[b,1]],
            [points[a,2], points[b,2]]
        )

    ax.set_title(f"TCPFormer Prediction — Frame {frame}")

    # Keep same scale
    all_points = pose.reshape(-1, 3)
    max_range = np.max(np.ptp(all_points, axis=0)) / 2

    center = np.mean(all_points, axis=0)

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

ani = FuncAnimation(
    fig,
    update,
    frames=81,
    interval=80
)

plt.close(fig)

HTML(ani.to_jshtml())

import os

AP3D_REPO = "/content/drive/MyDrive/AthletePose3D_Project/code/AthletePose3D"

for root, dirs, files in os.walk(AP3D_REPO):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)

            try:
                text = open(path, "r", errors="ignore").read()
            except:
                continue

            if "coco" in text.lower() and "h36m" in text.lower():
                print("\nFOUND:", path)

                for i, line in enumerate(text.splitlines(), 1):
                    if "coco" in line.lower() or "h36m" in line.lower():
                        print(f"{i}: {line}")

import numpy as np
import torch
import os

# ============================================================
# 1. OFFICIAL ATHLETEPOSE3D COCO -> H36M MAPPING
# ============================================================

coco_to_h36m_mapping = {
    0: [12, 11],
    1: 12,
    2: 14,
    3: 16,
    4: 11,
    5: 13,
    6: 15,
    7: [6, 5, 12, 11],
    8: [6, 5],
    9: [6, 5, 0, 0],
    10: 0,
    11: 5,
    12: 7,
    13: 9,
    14: 6,
    15: 8,
    16: 10
}


def coco_to_h36m(pose):
    """
    Convert COCO 17-joint pose to AthletePose3D/H36M 17-joint format.
    Input:  (T, 17, C)
    Output: (T, 17, C)
    """
    h36m_pose = np.zeros_like(pose)

    for i in range(pose.shape[0]):
        for h36m_idx, coco_idx in coco_to_h36m_mapping.items():

            if isinstance(coco_idx, list):
                h36m_pose[i, h36m_idx] = np.mean(
                    pose[i, coco_idx], axis=0
                )
            else:
                h36m_pose[i, h36m_idx] = pose[i, coco_idx]

    return h36m_pose


# ============================================================
# 2. CONVERT CLEAN COCO POSE -> H36M
# ============================================================

pose_2d_h36m = coco_to_h36m(pose_2d_clean)

print("Original COCO:", pose_2d_clean.shape)
print("Converted H36M:", pose_2d_h36m.shape)
print("NaNs:", np.isnan(pose_2d_h36m).sum())


# ============================================================
# 3. NORMALIZE EXACTLY AS ATHLETEPOSE3D
# ============================================================

pose_norm = pose_2d_h36m.astype(np.float32).copy()

pose_norm[:, :, 0] = (
    pose_norm[:, :, 0] / width * 2 - 1
)

pose_norm[:, :, 1] = (
    pose_norm[:, :, 1] / width * 2
    - (height / width)
)


# ============================================================
# 4. ADD CONFIDENCE CHANNEL
# ============================================================

confidence = np.ones(
    (pose_norm.shape[0], pose_norm.shape[1], 1),
    dtype=np.float32
)

tcp_input = np.concatenate(
    [pose_norm, confidence],
    axis=2
)

print("TCPFormer input:", tcp_input.shape)
print("NaNs:", np.isnan(tcp_input).sum())


# ============================================================
# 5. FIRST 81 FRAMES
# ============================================================

window = tcp_input[:81]

x = torch.from_numpy(window).unsqueeze(0).to(device)


# ============================================================
# 6. TCPFORMER INFERENCE
# ============================================================

model.eval()

with torch.no_grad():
    prediction = model(x)

predicted_3d = prediction.squeeze(0).cpu().numpy()

print("Prediction:", predicted_3d.shape)
print("NaNs:", np.isnan(predicted_3d).sum())
print("Infs:", np.isinf(predicted_3d).sum())


# ============================================================
# 7. SAVE CORRECTED OUTPUT TO GOOGLE DRIVE
# ============================================================

POSE3D_DIR = PROJECT + "/data/poses_3d"
os.makedirs(POSE3D_DIR, exist_ok=True)

output_path = POSE3D_DIR + "/Running_0_cam_1_TCPFormer_3D.npy"

np.save(output_path, predicted_3d)

print("\nSaved:")
print(output_path)

import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

connections = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16)
]

pose = predicted_3d.copy()

# Root-relative visualization
pose = pose - pose[:, 0:1, :]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")

def update(frame):

    ax.clear()

    p = pose[frame]

    # joints
    ax.scatter(
        p[:,0],
        p[:,1],
        p[:,2],
        s=30
    )

    # skeleton
    for a, b in connections:
        ax.plot(
            [p[a,0], p[b,0]],
            [p[a,1], p[b,1]],
            [p[a,2], p[b,2]]
        )

    ax.set_title(f"TCPFormer Corrected — Frame {frame}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # fixed limits so animation doesn't jump
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)

    ax.view_init(
        elev=15,
        azim=-70
    )

ani = FuncAnimation(
    fig,
    update,
    frames=len(pose),
    interval=50
)

plt.close(fig)

HTML(ani.to_jshtml())

import numpy as np

pred = np.load(
    "/content/drive/MyDrive/AthletePose3D_Project/data/poses_3d/Running_0_cam_1_TCPFormer_3D.npy"
)

gt = np.load(
    "/content/drive/MyDrive/AthletePose3D_Project/data/references/Running_0/S3/cam_1/Running_0_cam_1_h36m.npy"
)

print("========== TCPFORMER ==========")
print("Shape:", pred.shape)
print("X:", pred[:,:,0].min(), pred[:,:,0].max())
print("Y:", pred[:,:,1].min(), pred[:,:,1].max())
print("Z:", pred[:,:,2].min(), pred[:,:,2].max())

print("\n========== GROUND TRUTH ==========")
print("Shape:", gt.shape)
print("X:", gt[:,:,0].min(), gt[:,:,0].max())
print("Y:", gt[:,:,1].min(), gt[:,:,1].max())
print("Z:", gt[:,:,2].min(), gt[:,:,2].max())

import matplotlib.pyplot as plt

frame = 34

p = pred[frame].copy()

# Root-relative
p = p - p[0]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")

# Plot joints
ax.scatter(
    p[:,0],
    p[:,1],
    p[:,2],
    s=40
)

# Plot skeleton
for a, b in connections:
    ax.plot(
        [p[a,0], p[b,0]],
        [p[a,1], p[b,1]],
        [p[a,2], p[b,2]]
    )

# AUTO SCALE
mins = p.min(axis=0)
maxs = p.max(axis=0)

center = (mins + maxs) / 2
radius = max(maxs - mins) / 2

ax.set_xlim(center[0]-radius, center[0]+radius)
ax.set_ylim(center[1]-radius, center[1]+radius)
ax.set_zlim(center[2]-radius, center[2]+radius)

ax.set_title(f"TCPFormer Prediction — Frame {frame}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# LOAD SAVED TCPFORMER PREDICTION
# ------------------------------------------------------------
pred = np.load(
    "/content/drive/MyDrive/AthletePose3D_Project/data/poses_3d/"
    "Running_0_cam_1_TCPFormer_3D.npy"
)

# ------------------------------------------------------------
# LOAD ATHLETEPOSE3D GROUND TRUTH
# ------------------------------------------------------------
gt = np.load(
    "/content/drive/MyDrive/AthletePose3D_Project/data/references/"
    "Running_0/S3/cam_1/Running_0_cam_1_h36m.npy"
)

# Use same number of frames
n = min(len(pred), len(gt))

pred = pred[:n]
gt = gt[:n]

print("Prediction:", pred.shape)
print("Ground truth:", gt.shape)

# ------------------------------------------------------------
# ROOT-RELATIVE
# ------------------------------------------------------------
pred_rel = pred - pred[:, 0:1, :]
gt_rel = gt - gt[:, 0:1, :]

# ------------------------------------------------------------
# MPJPE
# ------------------------------------------------------------
error = np.linalg.norm(
    pred_rel - gt_rel,
    axis=2
)

mpjpe = error.mean()

print("\n==============================")
print("ROOT-RELATIVE MPJPE")
print("==============================")
print(f"{mpjpe:.6f}")

frame = 34

p = pred_rel[frame]
g = gt_rel[frame]

fig = plt.figure(figsize=(14, 6))

# ============================================================
# TCPFORMER
# ============================================================
ax1 = fig.add_subplot(121, projection="3d")

ax1.scatter(
    p[:,0], p[:,1], p[:,2],
    s=40
)

for a, b in connections:
    ax1.plot(
        [p[a,0], p[b,0]],
        [p[a,1], p[b,1]],
        [p[a,2], p[b,2]]
    )

ax1.set_title(f"TCPFormer — Frame {frame}")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")


# ============================================================
# GROUND TRUTH
# ============================================================
ax2 = fig.add_subplot(122, projection="3d")

ax2.scatter(
    g[:,0], g[:,1], g[:,2],
    s=40
)

for a, b in connections:
    ax2.plot(
        [g[a,0], g[b,0]],
        [g[a,1], g[b,1]],
        [g[a,2], g[b,2]]
    )

ax2.set_title(f"AthletePose3D Ground Truth — Frame {frame}")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.tight_layout()
plt.show()

import numpy as np
import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

POSE3D_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/data/poses_3d/"
    "Running_0_cam_1_TCPFormer_3D.npy"
)

FEATURE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/data/features"
)

os.makedirs(FEATURE_DIR, exist_ok=True)


# ============================================================
# LOAD 3D POSE
# ============================================================

pose = np.load(POSE3D_PATH)

print("3D pose shape:", pose.shape)


# ============================================================
# H36M JOINT INDICES
# ============================================================

PELVIS = 0

RIGHT_HIP = 1
RIGHT_KNEE = 2
RIGHT_ANKLE = 3

LEFT_HIP = 4
LEFT_KNEE = 5
LEFT_ANKLE = 6

SPINE = 7
CLAVICLE = 8
NECK = 9
HEAD = 10

LEFT_SHOULDER = 11
LEFT_ELBOW = 12
LEFT_WRIST = 13

RIGHT_SHOULDER = 14
RIGHT_ELBOW = 15
RIGHT_WRIST = 16


# ============================================================
# ANGLE FUNCTION
# ============================================================

def joint_angle(a, b, c):
    """
    Angle ABC.
    a, b, c -> (..., 3)
    """

    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba, axis=-1)
    bc_norm = np.linalg.norm(bc, axis=-1)

    denominator = ba_norm * bc_norm

    cosine = np.sum(ba * bc, axis=-1) / np.maximum(
        denominator, 1e-8
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# ============================================================
# JOINT ANGLES
# ============================================================

right_knee_angle = joint_angle(
    pose[:, RIGHT_HIP],
    pose[:, RIGHT_KNEE],
    pose[:, RIGHT_ANKLE]
)

left_knee_angle = joint_angle(
    pose[:, LEFT_HIP],
    pose[:, LEFT_KNEE],
    pose[:, LEFT_ANKLE]
)


right_hip_angle = joint_angle(
    pose[:, PELVIS],
    pose[:, RIGHT_HIP],
    pose[:, RIGHT_KNEE]
)

left_hip_angle = joint_angle(
    pose[:, PELVIS],
    pose[:, LEFT_HIP],
    pose[:, LEFT_KNEE]
)


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


# ============================================================
# BODY / LEG FEATURES
# ============================================================

hip_width = distance(
    pose[:, LEFT_HIP],
    pose[:, RIGHT_HIP]
)

shoulder_width = distance(
    pose[:, LEFT_SHOULDER],
    pose[:, RIGHT_SHOULDER]
)

right_leg_length = (
    distance(pose[:, RIGHT_HIP], pose[:, RIGHT_KNEE])
    +
    distance(pose[:, RIGHT_KNEE], pose[:, RIGHT_ANKLE])
)

left_leg_length = (
    distance(pose[:, LEFT_HIP], pose[:, LEFT_KNEE])
    +
    distance(pose[:, LEFT_KNEE], pose[:, LEFT_ANKLE])
)


knee_separation = distance(
    pose[:, LEFT_KNEE],
    pose[:, RIGHT_KNEE]
)

ankle_separation = distance(
    pose[:, LEFT_ANKLE],
    pose[:, RIGHT_ANKLE]
)


# ============================================================
# PELVIS-ROOTED JOINT POSITIONS
# ============================================================

root = pose[:, PELVIS:PELVIS+1, :]

relative_pose = pose - root


# ============================================================
# JOINT VELOCITIES
# ============================================================

velocity = np.zeros_like(pose)

velocity[1:] = pose[1:] - pose[:-1]

right_ankle_speed = np.linalg.norm(
    velocity[:, RIGHT_ANKLE],
    axis=1
)

left_ankle_speed = np.linalg.norm(
    velocity[:, LEFT_ANKLE],
    axis=1
)

right_knee_speed = np.linalg.norm(
    velocity[:, RIGHT_KNEE],
    axis=1
)

left_knee_speed = np.linalg.norm(
    velocity[:, LEFT_KNEE],
    axis=1
)


# ============================================================
# LEFT-RIGHT SYMMETRY
# ============================================================

knee_angle_difference = np.abs(
    right_knee_angle - left_knee_angle
)

hip_angle_difference = np.abs(
    right_hip_angle - left_hip_angle
)

ankle_speed_difference = np.abs(
    right_ankle_speed - left_ankle_speed
)


# ============================================================
# CREATE FEATURE TABLE
# ============================================================

features = pd.DataFrame({

    "frame": np.arange(len(pose)),

    "right_knee_angle": right_knee_angle,
    "left_knee_angle": left_knee_angle,

    "right_hip_angle": right_hip_angle,
    "left_hip_angle": left_hip_angle,

    "knee_angle_difference": knee_angle_difference,
    "hip_angle_difference": hip_angle_difference,

    "hip_width": hip_width,
    "shoulder_width": shoulder_width,

    "right_leg_length": right_leg_length,
    "left_leg_length": left_leg_length,

    "knee_separation": knee_separation,
    "ankle_separation": ankle_separation,

    "right_ankle_speed": right_ankle_speed,
    "left_ankle_speed": left_ankle_speed,

    "right_knee_speed": right_knee_speed,
    "left_knee_speed": left_knee_speed,

    "ankle_speed_difference": ankle_speed_difference
})


# ============================================================
# SAVE
# ============================================================

FEATURE_PATH = (
    FEATURE_DIR +
    "/Running_0_cam_1_TCPFormer_features.csv"
)

features.to_csv(FEATURE_PATH, index=False)

print("\nFeatures extracted:", features.shape)
print("\nSaved to:")
print(FEATURE_PATH)

print("\nFirst 5 frames:")
display(features.head())

import numpy as np
import pandas as pd
import os

# ============================================================
# PATHS
# ============================================================

GT_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/references/Running_0/S3/cam_1/"
    "Running_0_cam_1_h36m.npy"
)

FEATURE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/features"
)

GT_FEATURE_PATH = (
    FEATURE_DIR +
    "/Running_0_cam_1_GroundTruth_features.csv"
)

os.makedirs(FEATURE_DIR, exist_ok=True)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

gt_pose = np.load(GT_PATH)

print("Ground truth shape:", gt_pose.shape)


# ============================================================
# USE FIRST 81 FRAMES
# ============================================================

gt_pose = gt_pose[:81]

print("Using:", gt_pose.shape)


# ============================================================
# JOINT INDICES
# ============================================================

PELVIS = 0

RIGHT_HIP = 1
RIGHT_KNEE = 2
RIGHT_ANKLE = 3

LEFT_HIP = 4
LEFT_KNEE = 5
LEFT_ANKLE = 6

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 14


# ============================================================
# FUNCTIONS
# ============================================================

def joint_angle(a, b, c):

    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba, axis=-1)
    bc_norm = np.linalg.norm(bc, axis=-1)

    cosine = np.sum(ba * bc, axis=-1) / np.maximum(
        ba_norm * bc_norm,
        1e-8
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def distance(a, b):

    return np.linalg.norm(a - b, axis=-1)


# ============================================================
# ANGLES
# ============================================================

right_knee_angle = joint_angle(
    gt_pose[:, RIGHT_HIP],
    gt_pose[:, RIGHT_KNEE],
    gt_pose[:, RIGHT_ANKLE]
)

left_knee_angle = joint_angle(
    gt_pose[:, LEFT_HIP],
    gt_pose[:, LEFT_KNEE],
    gt_pose[:, LEFT_ANKLE]
)

right_hip_angle = joint_angle(
    gt_pose[:, PELVIS],
    gt_pose[:, RIGHT_HIP],
    gt_pose[:, RIGHT_KNEE]
)

left_hip_angle = joint_angle(
    gt_pose[:, PELVIS],
    gt_pose[:, LEFT_HIP],
    gt_pose[:, LEFT_KNEE]
)


# ============================================================
# DISTANCES
# ============================================================

hip_width = distance(
    gt_pose[:, LEFT_HIP],
    gt_pose[:, RIGHT_HIP]
)

shoulder_width = distance(
    gt_pose[:, LEFT_SHOULDER],
    gt_pose[:, RIGHT_SHOULDER]
)

right_leg_length = (
    distance(gt_pose[:, RIGHT_HIP], gt_pose[:, RIGHT_KNEE])
    +
    distance(gt_pose[:, RIGHT_KNEE], gt_pose[:, RIGHT_ANKLE])
)

left_leg_length = (
    distance(gt_pose[:, LEFT_HIP], gt_pose[:, LEFT_KNEE])
    +
    distance(gt_pose[:, LEFT_KNEE], gt_pose[:, LEFT_ANKLE])
)

knee_separation = distance(
    gt_pose[:, LEFT_KNEE],
    gt_pose[:, RIGHT_KNEE]
)

ankle_separation = distance(
    gt_pose[:, LEFT_ANKLE],
    gt_pose[:, RIGHT_ANKLE]
)


# ============================================================
# VELOCITIES
# ============================================================

velocity = np.zeros_like(gt_pose)

velocity[1:] = gt_pose[1:] - gt_pose[:-1]

right_ankle_speed = np.linalg.norm(
    velocity[:, RIGHT_ANKLE],
    axis=1
)

left_ankle_speed = np.linalg.norm(
    velocity[:, LEFT_ANKLE],
    axis=1
)

right_knee_speed = np.linalg.norm(
    velocity[:, RIGHT_KNEE],
    axis=1
)

left_knee_speed = np.linalg.norm(
    velocity[:, LEFT_KNEE],
    axis=1
)


# ============================================================
# SYMMETRY
# ============================================================

knee_angle_difference = np.abs(
    right_knee_angle - left_knee_angle
)

hip_angle_difference = np.abs(
    right_hip_angle - left_hip_angle
)

ankle_speed_difference = np.abs(
    right_ankle_speed - left_ankle_speed
)


# ============================================================
# FEATURE TABLE
# ============================================================

gt_features = pd.DataFrame({

    "frame": np.arange(len(gt_pose)),

    "right_knee_angle": right_knee_angle,
    "left_knee_angle": left_knee_angle,

    "right_hip_angle": right_hip_angle,
    "left_hip_angle": left_hip_angle,

    "knee_angle_difference": knee_angle_difference,
    "hip_angle_difference": hip_angle_difference,

    "hip_width": hip_width,
    "shoulder_width": shoulder_width,

    "right_leg_length": right_leg_length,
    "left_leg_length": left_leg_length,

    "knee_separation": knee_separation,
    "ankle_separation": ankle_separation,

    "right_ankle_speed": right_ankle_speed,
    "left_ankle_speed": left_ankle_speed,

    "right_knee_speed": right_knee_speed,
    "left_knee_speed": left_knee_speed,

    "ankle_speed_difference": ankle_speed_difference
})


# ============================================================
# SAVE
# ============================================================

gt_features.to_csv(
    GT_FEATURE_PATH,
    index=False
)

print("\nReference features:", gt_features.shape)

print("\nSaved to:")
print(GT_FEATURE_PATH)

display(gt_features.head())

import pandas as pd
import numpy as np
import os

FEATURE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/features"
)

PRED_PATH = FEATURE_DIR + "/Running_0_cam_1_TCPFormer_features.csv"
GT_PATH = FEATURE_DIR + "/Running_0_cam_1_GroundTruth_features.csv"

NORMALIZED_PRED_PATH = (
    FEATURE_DIR +
    "/Running_0_cam_1_TCPFormer_features_normalized.csv"
)

NORMALIZED_GT_PATH = (
    FEATURE_DIR +
    "/Running_0_cam_1_GroundTruth_features_normalized.csv"
)


# ============================================================
# LOAD
# ============================================================

pred = pd.read_csv(PRED_PATH)
gt = pd.read_csv(GT_PATH)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_features(df):

    df = df.copy()

    # Average leg length = body scale
    body_scale = (
        df["right_leg_length"] +
        df["left_leg_length"]
    ) / 2

    body_scale = body_scale.replace(0, np.nan)

    # --------------------------------------------------------
    # NORMALIZED DISTANCES
    # --------------------------------------------------------

    df["hip_width_norm"] = (
        df["hip_width"] / body_scale
    )

    df["shoulder_width_norm"] = (
        df["shoulder_width"] / body_scale
    )

    df["knee_separation_norm"] = (
        df["knee_separation"] / body_scale
    )

    df["ankle_separation_norm"] = (
        df["ankle_separation"] / body_scale
    )

    # --------------------------------------------------------
    # NORMALIZED LEG LENGTH
    # --------------------------------------------------------

    df["right_leg_length_norm"] = (
        df["right_leg_length"] / body_scale
    )

    df["left_leg_length_norm"] = (
        df["left_leg_length"] / body_scale
    )

    # --------------------------------------------------------
    # NORMALIZED SPEEDS
    # --------------------------------------------------------

    df["right_ankle_speed_norm"] = (
        df["right_ankle_speed"] / body_scale
    )

    df["left_ankle_speed_norm"] = (
        df["left_ankle_speed"] / body_scale
    )

    df["right_knee_speed_norm"] = (
        df["right_knee_speed"] / body_scale
    )

    df["left_knee_speed_norm"] = (
        df["left_knee_speed"] / body_scale
    )

    df["ankle_speed_difference_norm"] = (
        df["ankle_speed_difference"] / body_scale
    )

    return df


# ============================================================
# NORMALIZE
# ============================================================

pred_norm = normalize_features(pred)
gt_norm = normalize_features(gt)


# ============================================================
# SAVE
# ============================================================

pred_norm.to_csv(
    NORMALIZED_PRED_PATH,
    index=False
)

gt_norm.to_csv(
    NORMALIZED_GT_PATH,
    index=False
)


print("Prediction normalized:")
print(NORMALIZED_PRED_PATH)

print("\nGround truth normalized:")
print(NORMALIZED_GT_PATH)

print("\nNormalized prediction:")
display(pred_norm.head())

import os

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

USER_DATA = PROJECT + "/data/user_video"

os.makedirs(USER_DATA, exist_ok=True)
os.makedirs(USER_DATA + "/frames", exist_ok=True)
os.makedirs(USER_DATA + "/poses_2d", exist_ok=True)
os.makedirs(USER_DATA + "/poses_3d", exist_ok=True)
os.makedirs(USER_DATA + "/features", exist_ok=True)
os.makedirs(USER_DATA + "/results", exist_ok=True)

print("User-video workspace ready:")
print(USER_DATA)

!pip install -q -U openmim
!mim install -q "mmengine>=0.10.0"
!mim install -q "mmcv>=2.0.0"
!pip install -q "mmpose>=1.3.2"

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA:", torch.version.cuda)

!pip install -q -U ultralytics

import ultralytics
print("Ultralytics:", ultralytics.__version__)

!pip install -q -U openmim
!mim install "mmengine>=0.9.0"
!mim install "mmcv>=2.0.1"

!git clone -q https://github.com/open-mmlab/mmpose.git /content/mmpose
%cd /content/mmpose
!pip install -q -e .

import os

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

USER_DATA = PROJECT + "/data/user_video"

FRAME_DIR = USER_DATA + "/frames"
POSE2D_DIR = USER_DATA + "/poses_2d"
POSE3D_DIR = USER_DATA + "/poses_3d"
FEATURE_DIR = USER_DATA + "/features"
RESULT_DIR = USER_DATA + "/results"

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(POSE2D_DIR, exist_ok=True)
os.makedirs(POSE3D_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("User-video workspace ready:")
print(USER_DATA)

from google.colab import files

uploaded = files.upload()

video_name = list(uploaded.keys())[0]

print("Uploaded video:")
print(video_name)

import os
import shutil

video_path = os.path.join(USER_DATA, video_name)

if os.path.exists(video_path):
    print("Video already exists in Drive.")
else:
    shutil.copy2(video_name, video_path)
    print("Video copied to Drive.")

print("Video path:")
print(video_path)

import cv2

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()

print("Video:", video_name)
print("FPS:", fps)
print("Total frames:", total_frames)
print("Resolution:", width, "x", height)
print("Duration:", round(total_frames / fps, 2), "seconds")

import cv2
import os

existing_frames = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.endswith(".jpg")
])

if len(existing_frames) > 0:

    print("Frames already exist.")
    print("Number of frames:", len(existing_frames))

else:

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(
            FRAME_DIR,
            f"frame_{frame_idx:05d}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        frame_idx += 1

    cap.release()

    print("Frames extracted:", frame_idx)

print("Saved in:")
print(FRAME_DIR)

import matplotlib.pyplot as plt
import cv2
import os

frames = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.endswith(".jpg")
])

indices = [
    0,
    len(frames)//4,
    len(frames)//2,
    3*len(frames)//4,
    len(frames)-1
]

fig, axes = plt.subplots(
    1,
    len(indices),
    figsize=(18, 4)
)

for ax, idx in zip(axes, indices):

    img = cv2.imread(
        os.path.join(FRAME_DIR, frames[idx])
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    ax.imshow(img)
    ax.set_title(f"Frame {idx}")
    ax.axis("off")

plt.tight_layout()
plt.show()

!pip install -q -U ultralytics

from ultralytics import YOLO
import torch

device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolo26n-pose.pt")

print("Pose model loaded")
print("Device:", device)

test_frame = os.path.join(
    FRAME_DIR,
    "frame_00000.jpg"
)

results = model(
    test_frame,
    imgsz=1280,
    conf=0.05,
    device=device,
    verbose=False
)

result = results[0]

print("Number of people detected:", len(result.keypoints))
print("Keypoint shape:", result.keypoints.xy.shape)

annotated = result.plot()

import matplotlib.pyplot as plt
import cv2

annotated_rgb = cv2.cvtColor(
    annotated,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10, 7))
plt.imshow(annotated_rgb)
plt.axis("off")
plt.title("YOLO Pose Detection - Frame 0")
plt.show()

import numpy as np

COCO_RAW_PATH = POSE2D_DIR + "/user_video_COCO17_raw.npy"

coco_raw = np.load(COCO_RAW_PATH)

print("Raw shape:", coco_raw.shape)

print("\nValid detections per joint:")

joint_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]

for j, name in enumerate(joint_names):

    valid_frames = np.isfinite(coco_raw[:, j, 0]).sum()

    print(
        f"{j:2d} {name:15s}: "
        f"{valid_frames}/{len(coco_raw)}"
    )

print("\nFirst frame:")
print(coco_raw[0])

print("\nFrame 72:")
print(coco_raw[72])

print("\nFrame 143:")
print(coco_raw[143])

import os
import numpy as np
from tqdm import tqdm

COCO_RAW_PATH = POSE2D_DIR + "/user_video_COCO17_raw.npy"
COCO_PATH = POSE2D_DIR + "/user_video_COCO17.npy"
CONF_PATH = POSE2D_DIR + "/user_video_COCO17_confidence.npy"

frames = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.endswith(".jpg")
])

all_keypoints = []
all_confidence = []

print("Processing", len(frames), "frames...")

for frame_name in tqdm(frames):

    frame_path = os.path.join(FRAME_DIR, frame_name)

    results = model.predict(
        source=frame_path,
        imgsz=1280,
        conf=0.05,
        device=device,
        verbose=False
    )

    result = results[0]

    # Check actual number of detected persons
    num_people = result.keypoints.xy.shape[0]

    if num_people == 0:

        keypoints = np.full(
            (17, 2),
            np.nan,
            dtype=np.float32
        )

        confidence = np.full(
            (17,),
            np.nan,
            dtype=np.float32
        )

    else:

        # Choose largest detected person
        if result.boxes is not None and result.boxes.xyxy.shape[0] > 1:

            boxes = result.boxes.xyxy.cpu().numpy()

            areas = (
                (boxes[:, 2] - boxes[:, 0]) *
                (boxes[:, 3] - boxes[:, 1])
            )

            person_idx = int(np.argmax(areas))

        else:

            person_idx = 0

        # COCO-17 x,y
        keypoints = (
            result.keypoints.xy[person_idx]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Keypoint confidence
        if result.keypoints.conf is not None:

            confidence = (
                result.keypoints.conf[person_idx]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        else:

            confidence = np.ones(
                17,
                dtype=np.float32
            )

    all_keypoints.append(keypoints)
    all_confidence.append(confidence)


# Convert to arrays
coco_raw = np.asarray(
    all_keypoints,
    dtype=np.float32
)

coco_conf = np.asarray(
    all_confidence,
    dtype=np.float32
)

# Save raw results
np.save(COCO_RAW_PATH, coco_raw)
np.save(CONF_PATH, coco_conf)

print("\nRaw COCO-17 saved.")
print("Shape:", coco_raw.shape)
print("Confidence shape:", coco_conf.shape)

# --------------------------------------------------
# Interpolate missing frames
# --------------------------------------------------

coco_keypoints = coco_raw.copy()

for joint in range(17):

    for coord in range(2):

        values = coco_keypoints[:, joint, coord]

        valid = np.isfinite(values)

        if valid.sum() >= 2:

            coco_keypoints[:, joint, coord] = np.interp(
                np.arange(len(values)),
                np.where(valid)[0],
                values[valid]
            )

        elif valid.sum() == 1:

            coco_keypoints[:, joint, coord] = values[valid][0]

        else:

            print(
                f"Warning: joint {joint}, "
                f"coordinate {coord} has no detections."
            )


# Save final interpolated pose
np.save(
    COCO_PATH,
    coco_keypoints
)

print("\nFinal COCO-17 saved.")
print("Shape:", coco_keypoints.shape)
print(
    "NaN count:",
    np.isnan(coco_keypoints).sum()
)

import os
import numpy as np

test_frame = os.path.join(
    FRAME_DIR,
    "frame_00000.jpg"
)

print("Testing:", test_frame)
print("Exists:", os.path.exists(test_frame))

# Method A — the method that worked previously
results_a = model(
    test_frame,
    imgsz=1280,
    conf=0.05,
    device=device,
    verbose=False
)

result_a = results_a[0]

print("\nMETHOD A")
print("Boxes:", result_a.boxes.xyxy.shape)
print("Keypoints:", result_a.keypoints.xy.shape)


# Method B — the method used in the full loop
results_b = model.predict(
    source=test_frame,
    imgsz=1280,
    conf=0.05,
    device=device,
    verbose=False
)

result_b = results_b[0]

print("\nMETHOD B")
print("Boxes:", result_b.boxes.xyxy.shape)
print("Keypoints:", result_b.keypoints.xy.shape)

from ultralytics import YOLO
import torch

device = 0 if torch.cuda.is_available() else "cpu"

# Completely reload the pretrained pose model
model = YOLO("yolo26n-pose.pt")

print("Model:", model.model.__class__.__name__)
print("Device:", device)

import os

test_frame = os.path.join(
    FRAME_DIR,
    "frame_00000.jpg"
)

results = model.predict(
    source=test_frame,
    imgsz=1920,
    conf=0.001,
    iou=0.7,
    max_det=10,
    device=device,
    verbose=True
)

result = results[0]

print("\nBoxes:", result.boxes.xyxy.shape)
print("Keypoints:", result.keypoints.xy.shape)

if result.boxes.xyxy.shape[0] > 0:
    print("Detection confidence:")
    print(result.boxes.conf.cpu().numpy())

import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread(test_frame)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

boxes = result.boxes.xyxy.cpu().numpy()
scores = result.boxes.conf.cpu().numpy()
keypoints = result.keypoints.xy.cpu().numpy()

plt.figure(figsize=(14, 9))
plt.imshow(img_rgb)

for i, (box, score) in enumerate(zip(boxes, scores)):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1
    area = width * height

    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2
        )
    )

    plt.text(
        x1,
        y1,
        f"{i}: {score:.3f} | area={area:.0f}",
        fontsize=9
    )

plt.axis("off")
plt.title("All YOLO Pose Detections")
plt.show()

areas = []

for box in boxes:

    x1, y1, x2, y2 = box

    area = (x2 - x1) * (y2 - y1)

    areas.append(area)

areas = np.array(areas)

largest_idx = np.argmax(areas)

print("Largest detection:", largest_idx)
print("Confidence:", scores[largest_idx])
print("Area:", areas[largest_idx])
print("Box:", boxes[largest_idx])

import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread(test_frame)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

boxes = result.boxes.xyxy.cpu().numpy()
scores = result.boxes.conf.cpu().numpy()

plt.figure(figsize=(14, 9))
plt.imshow(img_rgb)

for i, (box, score) in enumerate(zip(boxes, scores)):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1
    area = width * height

    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2
        )
    )

    plt.text(
        x1,
        y1 - 5,
        f"ID {i} | conf={score:.3f} | area={area:.0f}",
        fontsize=10
    )

plt.axis("off")
plt.title("YOLO detections — identify the actual runner")
plt.show()

import os

DATASET_DIR = "/content/drive/MyDrive/AthletePose3D "
PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

AP2D_CHECKPOINT = os.path.join(
    DATASET_DIR,
    "model_params",
    "moganet_b_ap2d_384x288.pth"
)

print("Checkpoint:")
print(AP2D_CHECKPOINT)

print("\nExists:", os.path.exists(AP2D_CHECKPOINT))

if os.path.exists(AP2D_CHECKPOINT):
    print(
        "Size:",
        round(os.path.getsize(AP2D_CHECKPOINT) / (1024**3), 2),
        "GB"
    )

import shutil
import os

PROJECT_AP2D = os.path.join(
    PROJECT,
    "models",
    "pretrained",
    "moganet_b_ap2d_384x288.pth"
)

os.makedirs(
    os.path.dirname(PROJECT_AP2D),
    exist_ok=True
)

if os.path.exists(PROJECT_AP2D):

    print("Project checkpoint already exists.")

else:

    shutil.copy2(
        AP2D_CHECKPOINT,
        PROJECT_AP2D
    )

    print("Checkpoint copied to project.")

print(PROJECT_AP2D)

import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

import os

REPO_DIR = "/content/drive/MyDrive/AthletePose3D_Project/code/AthletePose3D"

print("AthletePose3D repo:", os.path.exists(REPO_DIR))

print("\npose_2d files:")

for f in os.listdir(os.path.join(REPO_DIR, "pose_2d")):
    print(" ", f)

print("\nSearching for MogaNet implementation/config references...")

for root, dirs, files in os.walk(REPO_DIR):

    for file in files:

        if file.endswith(".py"):

            path = os.path.join(root, file)

            try:
                text = open(
                    path,
                    "r",
                    errors="ignore"
                ).read()

                if "MogaNet" in text or "MogaNet_feat" in text:

                    print("\nFOUND:", os.path.relpath(path, REPO_DIR))

                    for line in text.splitlines():

                        if "MogaNet" in line or "moganet" in line:
                            print(" ", line[:200])

            except:
                pass

import os

MOGANET_DIR = "/content/MogaNet"

if os.path.exists(MOGANET_DIR):
    print("MogaNet repository already exists.")
else:
    !git clone -q https://github.com/Westlake-AI/MogaNet.git /content/MogaNet
    print("MogaNet cloned successfully.")

print("Path:", MOGANET_DIR)

import os

POSE_REPO = "/content/MogaNet/pose_estimation"

print("Pose estimation directory exists:",
      os.path.exists(POSE_REPO))

if os.path.exists(POSE_REPO):

    print("\nContents:")

    for item in os.listdir(POSE_REPO):
        print(" ", item)

import os

for root, dirs, files in os.walk("/content/MogaNet/pose_estimation"):

    for file in files:

        if file.endswith(".py") and (
            "moganet" in file.lower() or
            "mmpose" in root.lower()
        ):

            print(
                os.path.relpath(
                    os.path.join(root, file),
                    "/content/MogaNet/pose_estimation"
                )
            )

import os

ROOT = "/content/MogaNet"

matches = []

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)

            try:
                text = open(path, "r", errors="ignore").read()

                if "class MogaNet" in text or "MogaNet_feat" in text:
                    matches.append(path)

            except:
                pass

print("MogaNet implementation files found:")

for path in matches:
    print(os.path.relpath(path, ROOT))

import os

POSE_DIR = "/content/MogaNet/pose_estimation"

files_to_check = [
    os.path.join(POSE_DIR, "README.md"),
    os.path.join(
        POSE_DIR,
        "configs/body/2d_kpt_sview_rgb_img/"
        "topdown_heatmap/coco/moganet_b_coco_384x288.py"
    ),
]

for path in files_to_check:

    print("\n" + "=" * 80)
    print(os.path.relpath(path, "/content/MogaNet"))
    print("=" * 80)

    if os.path.exists(path):

        text = open(path, "r", errors="ignore").read()

        for line in text.splitlines():

            if any(keyword in line.lower() for keyword in [
                "mmpose",
                "mmcv",
                "mmengine",
                "moganet",
                "checkpoint",
                "model",
                "install",
                "version"
            ]):
                print(line)

    else:
        print("FILE NOT FOUND")

import torch
import os

CKPT = "/content/drive/MyDrive/AthletePose3D_Project/models/pretrained/moganet_b_ap2d_384x288.pth"

print("Checkpoint exists:", os.path.exists(CKPT))
print("Size (GB):", round(os.path.getsize(CKPT) / (1024**3), 3))

checkpoint = torch.load(
    CKPT,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint type:")
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nCheckpoint keys:")
    print(checkpoint.keys())

    for key, value in checkpoint.items():
        if hasattr(value, "keys"):
            print(f"\n{key} keys:")
            print(list(value.keys())[:20])
        else:
            print(f"\n{key}: {type(value)}")

CONFIG = "/content/MogaNet/pose_estimation/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/moganet_b_coco_384x288.py"

text = open(CONFIG, "r", errors="ignore").read()

print(text)

import importlib.util

packages = [
    "mmpose",
    "mmcv",
    "mmengine",
    "mmdet"
]

for package in packages:
    spec = importlib.util.find_spec(package)

    print(
        f"{package:10s} :",
        "INSTALLED" if spec else "NOT INSTALLED"
    )

    if spec:
        try:
            module = __import__(package)
            print(
                " " * 13,
                "version:",
                getattr(module, "__version__", "unknown")
            )
        except Exception as e:
            print(
                " " * 13,
                "import error:",
                str(e)[:200]
            )

!pip install -q "mmpose==0.29.0"

import os

REPO = "/content/drive/MyDrive/AthletePose3D_Project/code/AthletePose3D"

print("2D pose files:\n")

for root, dirs, files in os.walk(os.path.join(REPO, "pose_2d")):
    for f in files:
        print(os.path.relpath(os.path.join(root, f), REPO))

AP2D = "/content/drive/MyDrive/AthletePose3D_Project/code/AthletePose3D/pose_2d/ap2d.py"

print(open(AP2D, "r", errors="ignore").read())

!pip install -q mediapipe

import os
import urllib.request

MODEL_DIR = "/content/mediapipe_models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pose_landmarker_full.task"
)

if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded.")
else:
    print("Model already exists.")

print(MODEL_PATH)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(options)

print("Pose Landmarker loaded successfully.")

import os

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

print("Searching for extracted video frames...\n")

found = []

for root, dirs, files in os.walk(PROJECT):

    image_files = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if image_files:
        found.append((root, len(image_files)))

for path, count in found:
    print(f"{count:4d} images -> {path}")

import os

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

print("Searching for video files...\n")

for root, dirs, files in os.walk(PROJECT):
    for f in files:
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"{size_mb:.2f} MB -> {path}")

import os
import shutil

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"
USER_DATA = PROJECT + "/data/user_video"

os.makedirs(USER_DATA + "/frames", exist_ok=True)
os.makedirs(USER_DATA + "/poses_2d", exist_ok=True)
os.makedirs(USER_DATA + "/poses_3d", exist_ok=True)
os.makedirs(USER_DATA + "/features", exist_ok=True)
os.makedirs(USER_DATA + "/results", exist_ok=True)

from google.colab import files

uploaded = files.upload()

video_name = list(uploaded.keys())[0]

video_path = os.path.join(USER_DATA, video_name)

shutil.copy2(video_name, video_path)

print("Video saved permanently:")
print(video_path)

import cv2
import os

VIDEO_PATH = "/content/drive/MyDrive/AthletePose3D_Project/data/user_video/human_running_video.mp4"
FRAME_DIR = "/content/drive/MyDrive/AthletePose3D_Project/data/user_video/frames"

os.makedirs(FRAME_DIR, exist_ok=True)

existing_frames = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith(".jpg")
])

if len(existing_frames) > 0:

    print("Frames already exist.")
    print("Number of frames:", len(existing_frames))

else:

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("Total frames:", total_frames)

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(
            FRAME_DIR,
            f"frame_{frame_idx:05d}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        frame_idx += 1

    cap.release()

    print("Extracted frames:", frame_idx)

print("Frame folder:")
print(FRAME_DIR)

import cv2
import os
import mediapipe as mp

FRAME_DIR = "/content/drive/MyDrive/AthletePose3D_Project/data/user_video/frames"

frame_files = sorted([
    os.path.join(FRAME_DIR, f)
    for f in os.listdir(FRAME_DIR)
    if f.lower().endswith(".jpg")
])

print("Frames found:", len(frame_files))
print("Testing frame:", frame_files[0])

# Read frame
frame = cv2.imread(frame_files[0])

# BGR → RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# MediaPipe image
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)

# Run pose estimation
result = landmarker.detect(mp_image)

print("\nDetected poses:", len(result.pose_landmarks))

if len(result.pose_landmarks) > 0:

    landmarks = result.pose_landmarks[0]

    print("Landmarks detected:", len(landmarks))

    for i, lm in enumerate(landmarks):
        print(
            f"{i:2d}: "
            f"x={lm.x:.4f}, "
            f"y={lm.y:.4f}, "
            f"visibility={lm.visibility:.4f}"
        )

else:
    print("NO PERSON DETECTED")

import os
import cv2
import numpy as np

FRAME_DIR = "/content/drive/MyDrive/AthletePose3D_Project/data/user_video/frames"
POSE_DIR = "/content/drive/MyDrive/AthletePose3D_Project/data/user_video/poses_2d"

OUTPUT_PATH = os.path.join(
    POSE_DIR,
    "human_running_video_coco.npy"
)

os.makedirs(POSE_DIR, exist_ok=True)

# --------------------------------------------------
# Skip if already generated
# --------------------------------------------------

if os.path.exists(OUTPUT_PATH):

    poses_2d = np.load(OUTPUT_PATH)

    print("2D pose file already exists.")
    print("Path :", OUTPUT_PATH)
    print("Shape:", poses_2d.shape)

else:

    frame_files = sorted([
        os.path.join(FRAME_DIR, f)
        for f in os.listdir(FRAME_DIR)
        if f.lower().endswith(".jpg")
    ])

    print("Total frames:", len(frame_files))

    # --------------------------------------------------
    # MediaPipe → COCO-17 mapping
    #
    # COCO:
    # 0  nose
    # 1  left_eye
    # 2  right_eye
    # 3  left_ear
    # 4  right_ear
    # 5  left_shoulder
    # 6  right_shoulder
    # 7  left_elbow
    # 8  right_elbow
    # 9  left_wrist
    # 10 right_wrist
    # 11 left_hip
    # 12 right_hip
    # 13 left_knee
    # 14 right_knee
    # 15 left_ankle
    # 16 right_ankle
    # --------------------------------------------------

    mp_to_coco = [
        0,   # nose
        2,   # left_eye
        5,   # right_eye
        7,   # left_ear
        8,   # right_ear
        11,  # left_shoulder
        12,  # right_shoulder
        13,  # left_elbow
        14,  # right_elbow
        15,  # left_wrist
        16,  # right_wrist
        23,  # left_hip
        24,  # right_hip
        25,  # left_knee
        26,  # right_knee
        27,  # left_ankle
        28   # right_ankle
    ]

    poses_2d = np.full(
        (len(frame_files), 17, 2),
        np.nan,
        dtype=np.float32
    )

    for frame_idx, frame_path in enumerate(frame_files):

        frame = cv2.imread(frame_path)

        if frame is None:
            print("Could not read:", frame_path)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if len(result.pose_landmarks) == 0:
            continue

        landmarks = result.pose_landmarks[0]

        for coco_idx, mp_idx in enumerate(mp_to_coco):

            lm = landmarks[mp_idx]

            poses_2d[frame_idx, coco_idx, 0] = lm.x
            poses_2d[frame_idx, coco_idx, 1] = lm.y

        if (frame_idx + 1) % 20 == 0:
            print(
                f"Processed {frame_idx + 1}/{len(frame_files)} frames"
            )

    np.save(OUTPUT_PATH, poses_2d)

    print("\n2D pose extraction complete.")

print("\nOutput:")
print(OUTPUT_PATH)
print("Shape:", poses_2d.shape)

print("\nNaN count:", np.isnan(poses_2d).sum())

import os
import numpy as np

POSE_2D_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_2d/human_running_video_coco.npy"
)

POSE_INPUT_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_3d"
)

OUTPUT_PATH = os.path.join(
    POSE_INPUT_DIR,
    "human_running_video_TCPFormer_input.npy"
)

os.makedirs(POSE_INPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Skip if already prepared
# --------------------------------------------------

if os.path.exists(OUTPUT_PATH):

    tcp_input = np.load(OUTPUT_PATH)

    print("TCPFormer input already exists.")
    print("Path :", OUTPUT_PATH)
    print("Shape:", tcp_input.shape)

else:

    coco_2d = np.load(POSE_2D_PATH)

    print("Original COCO shape:", coco_2d.shape)

    # --------------------------------------------------
    # Official AthletePose3D COCO → H36M mapping
    # --------------------------------------------------

    coco_to_h36m_mapping = {
        0: [12, 11],
        1: 12,
        2: 14,
        3: 16,
        4: 11,
        5: 13,
        6: 15,
        7: [6, 5, 12, 11],
        8: [6, 5],
        9: [6, 5, 0, 0],
        10: 0,
        11: 5,
        12: 7,
        13: 9,
        14: 6,
        15: 8,
        16: 10
    }

    num_frames = coco_2d.shape[0]

    h36m_2d = np.zeros(
        (num_frames, 17, 2),
        dtype=np.float32
    )

    for i in range(num_frames):

        for h36m_idx, coco_idx in coco_to_h36m_mapping.items():

            if isinstance(coco_idx, list):
                h36m_2d[i, h36m_idx] = np.mean(
                    coco_2d[i, coco_idx],
                    axis=0
                )
            else:
                h36m_2d[i, h36m_idx] = coco_2d[
                    i, coco_idx
                ]

    print("H36M 2D shape:", h36m_2d.shape)

    # --------------------------------------------------
    # Normalize exactly in TCPFormer style
    #
    # AthletePose3D:
    # x = x / image_width * 2 - 1
    # y = y / image_width * 2 - image_height/image_width
    #
    # Our MediaPipe coordinates are already normalized
    # to [0,1], so convert them to the same [-1,1]
    # coordinate convention.
    # --------------------------------------------------

    # MediaPipe gives normalized x,y.
    #
    # Equivalent square normalization:
    # x -> 2*x - 1
    # y -> 2*y - 1

    h36m_norm = h36m_2d.copy()

    h36m_norm[:, :, 0] = (
        h36m_norm[:, :, 0] * 2.0 - 1.0
    )

    h36m_norm[:, :, 1] = (
        h36m_norm[:, :, 1] * 2.0 - 1.0
    )

    # --------------------------------------------------
    # Add confidence channel
    # --------------------------------------------------

    confidence = np.ones(
        (num_frames, 17, 1),
        dtype=np.float32
    )

    tcp_input_full = np.concatenate(
        [h36m_norm, confidence],
        axis=-1
    )

    print(
        "Full TCPFormer input:",
        tcp_input_full.shape
    )

    # --------------------------------------------------
    # Create 81-frame windows
    # --------------------------------------------------

    WINDOW = 81

    windows = []

    for start in range(
        0,
        num_frames - WINDOW + 1,
        WINDOW
    ):

        window = tcp_input_full[
            start:start + WINDOW
        ]

        windows.append(window)

    tcp_input = np.stack(windows)

    print(
        "TCPFormer windows:",
        tcp_input.shape
    )

    # Save
    np.save(
        OUTPUT_PATH,
        tcp_input
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

import os
import sys
import torch

TCPFORMER_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "code/TCPFormer"
)

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "models/pretrained/TCPFormer_ap3d_81.pth.tr"
)

if TCPFORMER_DIR not in sys.path:
    sys.path.insert(0, TCPFORMER_DIR)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Checkpoint exists:", os.path.exists(CHECKPOINT_PATH))

from model.Model import MemoryInducedTransformer

# --------------------------------------------------
# Official TCPFormer architecture
# --------------------------------------------------

model = MemoryInducedTransformer(
    n_layers=16,
    dim_in=3,
    dim_feat=128,
    dim_rep=512,
    dim_out=3,
    mlp_ratio=4,
    act_layer=torch.nn.GELU,
    attn_drop=0.0,
    drop=0.0,
    drop_path=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
    use_adaptive_fusion=True,
    num_heads=8,
    qkv_bias=False,
    qkv_scale=None,
    hierarchical=False,
    num_joints=17,
    use_temporal_similarity=True,
    temporal_connection_len=1,
    use_tcn=False,
    graph_only=False,
    neighbour_num=2,
    n_frames=81
)

# --------------------------------------------------
# Load pretrained weights
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

state_dict = {
    key.replace("module.", "", 1): value
    for key, value in checkpoint["model"].items()
}

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(device)
model.eval()

print("\nTCPFormer loaded successfully.")
print("Device :", device)
print("Frames :", 81)
print("Joints :", 17)
print("Input  :", 3)
print("Output :", 3)

import os
import numpy as np
import torch

INPUT_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_3d/"
    "human_running_video_TCPFormer_input.npy"
)

OUTPUT_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D.npy"
)

# --------------------------------------------------
# Skip if already generated
# --------------------------------------------------

if os.path.exists(OUTPUT_PATH):

    pose_3d = np.load(OUTPUT_PATH)

    print("3D pose already exists.")
    print("Path :", OUTPUT_PATH)
    print("Shape:", pose_3d.shape)

else:

    tcp_input = np.load(INPUT_PATH)

    print("Input shape:", tcp_input.shape)

    input_tensor = torch.from_numpy(
        tcp_input
    ).float().to(device)

    print("Tensor shape:", input_tensor.shape)

    # --------------------------------------------------
    # TCPFormer inference
    # --------------------------------------------------

    with torch.no_grad():

        prediction = model(input_tensor)

    # --------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------

    pose_3d = prediction.detach().cpu().numpy()

    print("Raw output shape:", pose_3d.shape)

    # --------------------------------------------------
    # Save permanently
    # --------------------------------------------------

    np.save(
        OUTPUT_PATH,
        pose_3d
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

# --------------------------------------------------
# Validation
# --------------------------------------------------

print("\nFinal 3D pose shape:", pose_3d.shape)
print("NaN count:", np.isnan(pose_3d).sum())
print("Inf count:", np.isinf(pose_3d).sum())

import os
import numpy as np

INPUT_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D.npy"
)

OUTPUT_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_combined.npy"
)

# --------------------------------------------------
# Skip if already exists
# --------------------------------------------------

if os.path.exists(OUTPUT_PATH):

    pose_3d_combined = np.load(OUTPUT_PATH)

    print("Combined 3D pose already exists.")
    print("Shape:", pose_3d_combined.shape)

else:

    pose_3d_windows = np.load(INPUT_PATH)

    print("Windowed input:", pose_3d_windows.shape)

    # (2, 81, 17, 3) → (162, 17, 3)
    pose_3d_combined = pose_3d_windows.reshape(
        -1,
        17,
        3
    )

    np.save(
        OUTPUT_PATH,
        pose_3d_combined
    )

    print("\nCombined 3D pose saved.")

print("\nPath:")
print(OUTPUT_PATH)

print("\nFinal shape:", pose_3d_combined.shape)
print("NaN count:", np.isnan(pose_3d_combined).sum())
print("Inf count:", np.isinf(pose_3d_combined).sum())

import os
import numpy as np
import pandas as pd

POSE_PATH = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_combined.npy"
)

FEATURE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/features"
)

OUTPUT_PATH = os.path.join(
    FEATURE_DIR,
    "human_running_video_features.csv"
)

os.makedirs(FEATURE_DIR, exist_ok=True)


# --------------------------------------------------
# Skip if already exists
# --------------------------------------------------

if os.path.exists(OUTPUT_PATH):

    features = pd.read_csv(OUTPUT_PATH)

    print("Features already exist.")
    print("Path:", OUTPUT_PATH)
    print("Shape:", features.shape)

else:

    pose = np.load(POSE_PATH)

    print("3D pose shape:", pose.shape)


    # --------------------------------------------------
    # Joint indices in AthletePose3D H36M order
    # --------------------------------------------------

    PELVIS = 0

    RIGHT_HIP = 1
    RIGHT_KNEE = 2
    RIGHT_ANKLE = 3

    LEFT_HIP = 4
    LEFT_KNEE = 5
    LEFT_ANKLE = 6

    SPINE = 7
    CLAVICLE = 8
    NECK = 9
    HEAD = 10

    LEFT_SHOULDER = 11
    LEFT_ELBOW = 12
    LEFT_WRIST = 13

    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16


    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------

    def angle_3points(a, b, c):
        """
        Angle ABC.
        """
        ba = a - b
        bc = c - b

        denominator = (
            np.linalg.norm(ba, axis=1)
            * np.linalg.norm(bc, axis=1)
        )

        cosine = np.sum(
            ba * bc,
            axis=1
        ) / np.maximum(
            denominator,
            1e-8
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return np.degrees(
            np.arccos(cosine)
        )


    def distance(a, b):
        return np.linalg.norm(
            a - b,
            axis=1
        )


    def frame_speed(points):
        """
        Frame-to-frame movement speed.
        """
        speed = np.zeros(
            len(points),
            dtype=np.float32
        )

        speed[1:] = np.linalg.norm(
            points[1:] - points[:-1],
            axis=1
        )

        return speed


    # --------------------------------------------------
    # Angles
    # --------------------------------------------------

    right_knee_angle = angle_3points(
        pose[:, RIGHT_HIP],
        pose[:, RIGHT_KNEE],
        pose[:, RIGHT_ANKLE]
    )

    left_knee_angle = angle_3points(
        pose[:, LEFT_HIP],
        pose[:, LEFT_KNEE],
        pose[:, LEFT_ANKLE]
    )


    right_hip_angle = angle_3points(
        pose[:, PELVIS],
        pose[:, RIGHT_HIP],
        pose[:, RIGHT_KNEE]
    )

    left_hip_angle = angle_3points(
        pose[:, PELVIS],
        pose[:, LEFT_HIP],
        pose[:, LEFT_KNEE]
    )


    # --------------------------------------------------
    # Symmetry
    # --------------------------------------------------

    knee_angle_difference = np.abs(
        right_knee_angle -
        left_knee_angle
    )

    hip_angle_difference = np.abs(
        right_hip_angle -
        left_hip_angle
    )


    # --------------------------------------------------
    # Body geometry
    # --------------------------------------------------

    hip_width = distance(
        pose[:, RIGHT_HIP],
        pose[:, LEFT_HIP]
    )

    shoulder_width = distance(
        pose[:, RIGHT_SHOULDER],
        pose[:, LEFT_SHOULDER]
    )

    right_leg_length = (
        distance(
            pose[:, RIGHT_HIP],
            pose[:, RIGHT_KNEE]
        )
        +
        distance(
            pose[:, RIGHT_KNEE],
            pose[:, RIGHT_ANKLE]
        )
    )

    left_leg_length = (
        distance(
            pose[:, LEFT_HIP],
            pose[:, LEFT_KNEE]
        )
        +
        distance(
            pose[:, LEFT_KNEE],
            pose[:, LEFT_ANKLE]
        )
    )

    knee_separation = distance(
        pose[:, RIGHT_KNEE],
        pose[:, LEFT_KNEE]
    )

    ankle_separation = distance(
        pose[:, RIGHT_ANKLE],
        pose[:, LEFT_ANKLE]
    )


    # --------------------------------------------------
    # Movement speeds
    # --------------------------------------------------

    right_ankle_speed = frame_speed(
        pose[:, RIGHT_ANKLE]
    )

    left_ankle_speed = frame_speed(
        pose[:, LEFT_ANKLE]
    )

    right_knee_speed = frame_speed(
        pose[:, RIGHT_KNEE]
    )

    left_knee_speed = frame_speed(
        pose[:, LEFT_KNEE]
    )

    ankle_speed_difference = np.abs(
        right_ankle_speed -
        left_ankle_speed
    )


    # --------------------------------------------------
    # Create dataframe
    # --------------------------------------------------

    features = pd.DataFrame({

        "frame": np.arange(
            len(pose)
        ),

        "right_knee_angle":
            right_knee_angle,

        "left_knee_angle":
            left_knee_angle,

        "right_hip_angle":
            right_hip_angle,

        "left_hip_angle":
            left_hip_angle,

        "knee_angle_difference":
            knee_angle_difference,

        "hip_angle_difference":
            hip_angle_difference,

        "hip_width":
            hip_width,

        "shoulder_width":
            shoulder_width,

        "right_leg_length":
            right_leg_length,

        "left_leg_length":
            left_leg_length,

        "knee_separation":
            knee_separation,

        "ankle_separation":
            ankle_separation,

        "right_ankle_speed":
            right_ankle_speed,

        "left_ankle_speed":
            left_ankle_speed,

        "right_knee_speed":
            right_knee_speed,

        "left_knee_speed":
            left_knee_speed,

        "ankle_speed_difference":
            ankle_speed_difference
    })


    # --------------------------------------------------
    # Normalize distance/speed features by leg length
    # --------------------------------------------------

    body_scale = (
        features["right_leg_length"]
        +
        features["left_leg_length"]
    ) / 2.0

    body_scale = body_scale.replace(
        0,
        np.nan
    )

    features["hip_width_norm"] = (
        features["hip_width"] /
        body_scale
    )

    features["shoulder_width_norm"] = (
        features["shoulder_width"] /
        body_scale
    )

    features["knee_separation_norm"] = (
        features["knee_separation"] /
        body_scale
    )

    features["ankle_separation_norm"] = (
        features["ankle_separation"] /
        body_scale
    )

    features["right_leg_length_norm"] = (
        features["right_leg_length"] /
        body_scale
    )

    features["left_leg_length_norm"] = (
        features["left_leg_length"] /
        body_scale
    )

    features["right_ankle_speed_norm"] = (
        features["right_ankle_speed"] /
        body_scale
    )

    features["left_ankle_speed_norm"] = (
        features["left_ankle_speed"] /
        body_scale
    )

    features["right_knee_speed_norm"] = (
        features["right_knee_speed"] /
        body_scale
    )

    features["left_knee_speed_norm"] = (
        features["left_knee_speed"] /
        body_scale
    )

    features["ankle_speed_difference_norm"] = (
        features["ankle_speed_difference"] /
        body_scale
    )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    features.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature extraction complete.")
    print("Saved:", OUTPUT_PATH)
    print("Shape:", features.shape)


# --------------------------------------------------
# Final validation
# --------------------------------------------------

print("\nColumns:")
print(list(features.columns))

print("\nNaN values:")
print(features.isna().sum().sum())

print("\nFirst 5 rows:")
print(features.head())

import os

REFERENCE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/references/RBDS"
)

os.makedirs(REFERENCE_DIR, exist_ok=True)

print("Reference folder:")
print(REFERENCE_DIR)

import requests
import json

ARTICLE_ID = 4543435

url = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"

response = requests.get(url, timeout=30)

print("Status:", response.status_code)

if response.status_code != 200:
    print(response.text)
    raise RuntimeError("Could not access the Figshare dataset.")

metadata = response.json()

print("\nTitle:")
print(metadata.get("title"))

print("\nDescription:")
print(metadata.get("description", "")[:500])

print("\nFiles:")
for f in metadata.get("files", []):
    print(
        f"Name: {f.get('name')}\n"
        f"Size: {f.get('size')} bytes\n"
        f"Download: {f.get('download_url')}\n"
    )

import os
import requests

REFERENCE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/references/RBDS/processed"
)

os.makedirs(REFERENCE_DIR, exist_ok=True)

# Official Figshare processed files
processed_files = {
    "RBDS004processed.txt":
        "https://ndownloader.figshare.com/files/7824949",

    "RBDS005processed.txt":
        "https://ndownloader.figshare.com/files/7824952",

    "RBDS006processed.txt":
        "https://ndownloader.figshare.com/files/7824955",

    "RBDS007processed.txt":
        "https://ndownloader.figshare.com/files/7824958",

    "RBDS008processed.txt":
        "https://ndownloader.figshare.com/files/7824961",

    "RBDS009processed.txt":
        "https://ndownloader.figshare.com/files/7824964",

    "RBDS010processed.txt":
        "https://ndownloader.figshare.com/files/7824967",

    "RBDS011processed.txt":
        "https://ndownloader.figshare.com/files/7824970",

    "RBDS012processed.txt":
        "https://ndownloader.figshare.com/files/7824973",

    "RBDS013processed.txt":
        "https://ndownloader.figshare.com/files/7824976",

    "RBDS014processed.txt":
        "https://ndownloader.figshare.com/files/7824979",

    "RBDS015processed.txt":
        "https://ndownloader.figshare.com/files/7824982",

    "RBDS016processed.txt":
        "https://ndownloader.figshare.com/files/7824985",

    "RBDS017processed.txt":
        "https://ndownloader.figshare.com/files/7824988",

    "RBDS018processed.txt":
        "https://ndownloader.figshare.com/files/7824991",

    "RBDS019processed.txt":
        "https://ndownloader.figshare.com/files/7824994",

    "RBDS020processed.txt":
        "https://ndownloader.figshare.com/files/7824997",

    "RBDS021processed.txt":
        "https://ndownloader.figshare.com/files/7825000",

    "RBDS022processed.txt":
        "https://ndownloader.figshare.com/files/7825003",

    "RBDS023processed.txt":
        "https://ndownloader.figshare.com/files/7825006",

    "RBDS024processed.txt":
        "https://ndownloader.figshare.com/files/7825009",

    "RBDS025processed.txt":
        "https://ndownloader.figshare.com/files/7825012",

    "RBDS026processed.txt":
        "https://ndownloader.figshare.com/files/7825015",

    "RBDS027processed.txt":
        "https://ndownloader.figshare.com/files/7825018",

    "RBDS028processed.txt":
        "https://ndownloader.figshare.com/files/7825021",

    "RBDS001processed.txt":
        "https://ndownloader.figshare.com/files/7825024"
}

for filename, url in processed_files.items():

    output_path = os.path.join(
        REFERENCE_DIR,
        filename
    )

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print("Already exists:", filename)
        continue

    print("Downloading:", filename)

    response = requests.get(
        url,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    with open(output_path, "wb") as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                f.write(chunk)

    size_kb = os.path.getsize(output_path) / 1024

    print(
        f"  Saved: {size_kb:.1f} KB"
    )

print("\nReference directory:")
print(REFERENCE_DIR)

print(
    "\nFiles downloaded:",
    len([
        f for f in os.listdir(REFERENCE_DIR)
        if f.endswith(".txt")
    ])
)

import os
import pandas as pd

REFERENCE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/references/RBDS/processed"
)

file_path = os.path.join(
    REFERENCE_DIR,
    "RBDS001processed.txt"
)

print("File:", file_path)
print("Size:", round(os.path.getsize(file_path) / 1024, 2), "KB")

# Show first 20 lines exactly as stored
with open(file_path, "r", errors="ignore") as f:
    for i in range(20):
        line = f.readline()
        if not line:
            break
        print(f"{i+1}: {line.rstrip()}")

import os
import pandas as pd
import numpy as np

REFERENCE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/references/RBDS/processed"
)

file_path = os.path.join(
    REFERENCE_DIR,
    "RBDS001processed.txt"
)

df = pd.read_csv(file_path, sep="\t")

print("Shape:", df.shape)

print("\nAngle columns:")
angle_cols = [
    c for c in df.columns
    if "Ang" in c
]

print(angle_cols)

print("\nPercGcycle:")
print(df["PercGcycle"].head())
print("...")
print(df["PercGcycle"].tail())

print("\nAngle ranges:")
for col in angle_cols:
    print(
        f"{col:15s} "
        f"min={df[col].min():8.2f} "
        f"max={df[col].max():8.2f}"
    )

import os

REFERENCE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/references/RBDS/processed"
)

print("RBDS processed files:")
files = sorted([
    f for f in os.listdir(REFERENCE_DIR)
    if f.endswith(".txt")
])

for f in files:
    print(f)

print("\nCount:", len(files))

import os
import pandas as pd
import numpy as np

REFERENCE_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/references/RBDS/processed"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/references/RBDS/reference_curves"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "RBDS_running_reference_curves.csv"
)

files = sorted([
    f for f in os.listdir(REFERENCE_DIR)
    if f.endswith(".txt")
])

# Exact RBDS joint names
joints = {
    "hip": "hip",
    "knee": "knee",
    "ankle": "ankle"
}

sides = {
    "right": "R",
    "left": "L"
}

speeds = {
    "2.5mps": "25",
    "3.5mps": "35",
    "4.5mps": "45"
}

records = []

for speed_label, speed_code in speeds.items():

    print(f"Processing speed: {speed_label}")

    for joint_name, joint_code in joints.items():

        for side_name, side_code in sides.items():

            for axis in ["X", "Y", "Z"]:

                # IMPORTANT: exact RBDS capitalization
                column = (
                    f"{side_code}"
                    f"{joint_code}"
                    f"Ang{axis}"
                    f"{speed_code}"
                )

                curves = []

                for filename in files:

                    path = os.path.join(
                        REFERENCE_DIR,
                        filename
                    )

                    df = pd.read_csv(
                        path,
                        sep="\t"
                    )

                    if column not in df.columns:
                        continue

                    values = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).to_numpy()

                    if len(values) == 101:
                        curves.append(values)

                if len(curves) == 0:
                    print("WARNING - no data:", column)
                    continue

                curves = np.asarray(curves)

                for cycle_percent in range(101):

                    values = curves[:, cycle_percent]
                    values = values[~np.isnan(values)]

                    if len(values) == 0:
                        continue

                    records.append({
                        "speed": speed_label,
                        "joint": joint_name,
                        "side": side_name,
                        "axis": axis,
                        "gait_cycle_percent": cycle_percent,
                        "n_trials": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values, ddof=1),
                        "p05": np.percentile(values, 5),
                        "p95": np.percentile(values, 95)
                    })

reference_df = pd.DataFrame(records)

# Overwrite the empty file
reference_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===================================")
print("REFERENCE DATASET CREATED")
print("===================================")
print("Shape:", reference_df.shape)
print("Saved:", OUTPUT_FILE)

print("\nSpeeds:")
print(reference_df["speed"].unique())

print("\nJoints:")
print(reference_df["joint"].unique())

print("\nAxes:")
print(reference_df["axis"].unique())

print("\nSample:")
print(reference_df.head(10))

import os
import pandas as pd
import numpy as np

USER_FEATURE_FILE = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/features/"
    "human_running_video_features.csv"
)

df = pd.read_csv(USER_FEATURE_FILE)

print("Shape:", df.shape)

print("\nFrames:")
print(df["frame"].min(), "to", df["frame"].max())

print("\nKey running signals:")

signals = [
    "right_knee_angle",
    "left_knee_angle",
    "right_ankle_speed",
    "left_ankle_speed",
    "knee_angle_difference"
]

for col in signals:
    print(
        f"{col:30s}",
        f"min={df[col].min():.3f}",
        f"max={df[col].max():.3f}",
        f"mean={df[col].mean():.3f}"
    )

print("\nFirst 10 rows:")
print(
    df[
        ["frame"] + signals
    ].head(10)
)

import pandas as pd
import matplotlib.pyplot as plt

USER_FEATURE_FILE = (
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/features/"
    "human_running_video_features.csv"
)

df = pd.read_csv(USER_FEATURE_FILE)

plt.figure(figsize=(14, 5))
plt.plot(
    df["frame"],
    df["right_ankle_speed"],
    label="Right ankle speed"
)
plt.plot(
    df["frame"],
    df["left_ankle_speed"],
    label="Left ankle speed"
)

plt.xlabel("Frame")
plt.ylabel("Normalized movement")
plt.title("User Running Movement — Ankle Motion")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(
    df["frame"],
    df["right_knee_angle"],
    label="Right knee angle"
)
plt.plot(
    df["frame"],
    df["left_knee_angle"],
    label="Left knee angle"
)

plt.xlabel("Frame")
plt.ylabel("Angle (degrees)")
plt.title("User Running Movement — Knee Angles")
plt.legend()
plt.grid(True)
plt.show()

import os
import numpy as np
import torch

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

COCO_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

EXISTING_3D_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_combined.npy"
)

FULL_3D_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

# --------------------------------------------------
# 1. Check if already completed
# --------------------------------------------------

if os.path.exists(FULL_3D_PATH):
    full_pose = np.load(FULL_3D_PATH)

    print("Full 240-frame 3D pose already exists.")
    print("Shape:", full_pose.shape)

else:

    # --------------------------------------------------
    # 2. Load existing 162-frame result
    # --------------------------------------------------

    existing_pose = np.load(EXISTING_3D_PATH)

    print("Existing pose:", existing_pose.shape)

    # Expected: (162, 17, 3)
    assert existing_pose.shape[0] == 162

    # --------------------------------------------------
    # 3. Load complete 2D COCO pose
    # --------------------------------------------------

    coco = np.load(COCO_PATH)

    print("COCO pose:", coco.shape)

    # Expected: (240, 17, 2)
    assert coco.shape == (240, 17, 2)

    # --------------------------------------------------
    # 4. Final overlapping 81-frame window
    #    Frames 159 ... 239
    # --------------------------------------------------

    window_2d = coco[159:240].copy()

    # --------------------------------------------------
    # 5. Handle possible missing values
    # --------------------------------------------------

    for joint in range(17):

        for coord in range(2):

            values = window_2d[:, joint, coord]

            if np.any(np.isnan(values)):

                valid = ~np.isnan(values)

                if valid.sum() >= 2:
                    window_2d[:, joint, coord] = np.interp(
                        np.arange(len(values)),
                        np.where(valid)[0],
                        values[valid]
                    )

                elif valid.sum() == 1:
                    window_2d[:, joint, coord] = values[valid][0]

                else:
                    window_2d[:, joint, coord] = 0.5

    # --------------------------------------------------
    # 6. Official COCO -> H36M mapping
    # --------------------------------------------------

    coco_to_h36m_mapping = {
        0: [12, 11],
        1: 12,
        2: 14,
        3: 16,
        4: 11,
        5: 13,
        6: 15,
        7: [6, 5, 12, 11],
        8: [6, 5],
        9: [6, 5, 0, 0],
        10: 0,
        11: 5,
        12: 7,
        13: 9,
        14: 6,
        15: 8,
        16: 10
    }

    h36m_2d = np.zeros(
        (81, 17, 2),
        dtype=np.float32
    )

    for h36m_idx, coco_idx in coco_to_h36m_mapping.items():

        if isinstance(coco_idx, list):
            h36m_2d[:, h36m_idx] = np.mean(
                window_2d[:, coco_idx],
                axis=1
            )
        else:
            h36m_2d[:, h36m_idx] = (
                window_2d[:, coco_idx]
            )

    # --------------------------------------------------
    # 7. Normalize exactly as our existing user pipeline
    # --------------------------------------------------

    h36m_input = np.zeros(
        (81, 17, 3),
        dtype=np.float32
    )

    h36m_input[:, :, 0] = (
        h36m_2d[:, :, 0] * 2.0 - 1.0
    )

    h36m_input[:, :, 1] = (
        h36m_2d[:, :, 1] * 2.0 - 1.0
    )

    # Confidence
    h36m_input[:, :, 2] = 1.0

    input_tensor = torch.from_numpy(
        h36m_input
    ).unsqueeze(0).float()

    # --------------------------------------------------
    # 8. Reload TCPFormer only if necessary
    # --------------------------------------------------

    if "model" not in globals():

        print("Reloading TCPFormer...")

        import sys

        TCPFORMER_DIR = (
            PROJECT +
            "/code/TCPFormer"
        )

        if TCPFORMER_DIR not in sys.path:
            sys.path.insert(0, TCPFORMER_DIR)

        from model.Model import MemoryInducedTransformer

        model = MemoryInducedTransformer(
            n_layers=16,
            dim_in=3,
            dim_feat=128,
            dim_rep=512,
            dim_out=3,
            mlp_ratio=4,
            act_layer=torch.nn.GELU,
            attn_drop=0.0,
            drop=0.0,
            drop_path=0.0,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
            use_adaptive_fusion=True,
            num_heads=8,
            qkv_bias=False,
            qkv_scale=None,
            hierarchical=False,
            num_joints=17,
            use_temporal_similarity=True,
            temporal_connection_len=1,
            use_tcn=False,
            graph_only=False,
            neighbour_num=2,
            n_frames=81
        )

        checkpoint_path = (
            PROJECT +
            "/models/pretrained/"
            "TCPFormer_ap3d_81.pth.tr"
        )

        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False
        )

        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint["model"].items()
        }

        model.load_state_dict(
            state_dict,
            strict=True
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        model = model.to(device)
        model.eval()

    else:
        device = next(model.parameters()).device

    # --------------------------------------------------
    # 9. TCPFormer inference
    # --------------------------------------------------

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    prediction = prediction.detach().cpu().numpy()

    print("Final window prediction:", prediction.shape)

    # --------------------------------------------------
    # 10. Keep only NEW frames
    #
    # Window:
    #   159 ... 239
    #
    # Existing:
    #   0 ... 161
    #
    # New frames:
    #   162 ... 239
    #
    # Therefore discard first 3 predictions.
    # --------------------------------------------------

    new_frames = prediction[0, 3:81]

    print("New frames:", new_frames.shape)

    # --------------------------------------------------
    # 11. Combine
    # --------------------------------------------------

    full_pose = np.concatenate(
        [
            existing_pose,
            new_frames
        ],
        axis=0
    )

    print("Complete pose:", full_pose.shape)

    assert full_pose.shape == (240, 17, 3)

    # --------------------------------------------------
    # 12. Save permanently
    # --------------------------------------------------

    np.save(
        FULL_3D_PATH,
        full_pose.astype(np.float32)
    )

    print("\nSaved complete 240-frame 3D pose:")
    print(FULL_3D_PATH)

import os
import numpy as np
import pandas as pd

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

OUTPUT_PATH = (
    PROJECT +
    "/data/user_video/features/"
    "human_running_video_features_full.csv"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# --------------------------------------------------
# Skip if already created
# --------------------------------------------------

if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 1000:

    features_df = pd.read_csv(OUTPUT_PATH)

    print("Full feature file already exists.")
    print("Shape:", features_df.shape)
    print("Path:", OUTPUT_PATH)

else:

    pose = np.load(POSE_PATH)

    print("3D pose shape:", pose.shape)

    assert pose.shape == (240, 17, 3)

    # --------------------------------------------------
    # AthletePose3D H36M joint indices
    # --------------------------------------------------

    PELVIS = 0
    RIGHT_HIP = 1
    RIGHT_KNEE = 2
    RIGHT_ANKLE = 3

    LEFT_HIP = 4
    LEFT_KNEE = 5
    LEFT_ANKLE = 6

    SPINE = 7
    CLAVICLE = 8

    LEFT_SHOULDER = 11
    LEFT_ELBOW = 12
    LEFT_WRIST = 13

    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16

    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------

    def angle_3d(a, b, c):
        """
        Angle ABC in degrees.
        """
        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba, axis=1)
        norm_bc = np.linalg.norm(bc, axis=1)

        denominator = norm_ba * norm_bc

        cosine = np.sum(
            ba * bc,
            axis=1
        ) / np.maximum(denominator, 1e-8)

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return np.degrees(
            np.arccos(cosine)
        )

    def distance(a, b):
        return np.linalg.norm(
            a - b,
            axis=1
        )

    def frame_speed(points):
        """
        Frame-to-frame displacement.
        """
        speed = np.zeros(
            len(points),
            dtype=np.float32
        )

        speed[1:] = np.linalg.norm(
            points[1:] - points[:-1],
            axis=1
        )

        return speed

    # --------------------------------------------------
    # Extract joints
    # --------------------------------------------------

    pelvis = pose[:, PELVIS]

    rhip = pose[:, RIGHT_HIP]
    rknee = pose[:, RIGHT_KNEE]
    rankle = pose[:, RIGHT_ANKLE]

    lhip = pose[:, LEFT_HIP]
    lknee = pose[:, LEFT_KNEE]
    lankle = pose[:, LEFT_ANKLE]

    lshoulder = pose[:, LEFT_SHOULDER]
    rshoulder = pose[:, RIGHT_SHOULDER]

    # --------------------------------------------------
    # Joint angles
    # --------------------------------------------------

    right_knee_angle = angle_3d(
        rhip,
        rknee,
        rankle
    )

    left_knee_angle = angle_3d(
        lhip,
        lknee,
        lankle
    )

    right_hip_angle = angle_3d(
        pelvis,
        rhip,
        rknee
    )

    left_hip_angle = angle_3d(
        pelvis,
        lhip,
        lknee
    )

    # --------------------------------------------------
    # Geometric features
    # --------------------------------------------------

    hip_width = distance(
        rhip,
        lhip
    )

    shoulder_width = distance(
        rshoulder,
        lshoulder
    )

    right_leg_length = (
        distance(rhip, rknee) +
        distance(rknee, rankle)
    )

    left_leg_length = (
        distance(lhip, lknee) +
        distance(lknee, lankle)
    )

    knee_separation = distance(
        rknee,
        lknee
    )

    ankle_separation = distance(
        rankle,
        lankle
    )

    # --------------------------------------------------
    # Movement features
    # --------------------------------------------------

    right_ankle_speed = frame_speed(rankle)
    left_ankle_speed = frame_speed(lankle)

    right_knee_speed = frame_speed(rknee)
    left_knee_speed = frame_speed(lknee)

    ankle_speed_difference = np.abs(
        right_ankle_speed -
        left_ankle_speed
    )

    # --------------------------------------------------
    # Symmetry features
    # --------------------------------------------------

    knee_angle_difference = np.abs(
        right_knee_angle -
        left_knee_angle
    )

    hip_angle_difference = np.abs(
        right_hip_angle -
        left_hip_angle
    )

    # --------------------------------------------------
    # Body-scale normalization
    # --------------------------------------------------

    body_scale = (
        right_leg_length +
        left_leg_length
    ) / 2.0

    body_scale = np.maximum(
        body_scale,
        1e-8
    )

    hip_width_norm = (
        hip_width / body_scale
    )

    shoulder_width_norm = (
        shoulder_width / body_scale
    )

    knee_separation_norm = (
        knee_separation / body_scale
    )

    ankle_separation_norm = (
        ankle_separation / body_scale
    )

    right_leg_length_norm = (
        right_leg_length / body_scale
    )

    left_leg_length_norm = (
        left_leg_length / body_scale
    )

    right_ankle_speed_norm = (
        right_ankle_speed / body_scale
    )

    left_ankle_speed_norm = (
        left_ankle_speed / body_scale
    )

    right_knee_speed_norm = (
        right_knee_speed / body_scale
    )

    left_knee_speed_norm = (
        left_knee_speed / body_scale
    )

    ankle_speed_difference_norm = (
        ankle_speed_difference / body_scale
    )

    # --------------------------------------------------
    # Create dataframe
    # --------------------------------------------------

    features_df = pd.DataFrame({

        "frame": np.arange(len(pose)),

        "right_knee_angle":
            right_knee_angle,

        "left_knee_angle":
            left_knee_angle,

        "right_hip_angle":
            right_hip_angle,

        "left_hip_angle":
            left_hip_angle,

        "knee_angle_difference":
            knee_angle_difference,

        "hip_angle_difference":
            hip_angle_difference,

        "hip_width":
            hip_width,

        "shoulder_width":
            shoulder_width,

        "right_leg_length":
            right_leg_length,

        "left_leg_length":
            left_leg_length,

        "knee_separation":
            knee_separation,

        "ankle_separation":
            ankle_separation,

        "right_ankle_speed":
            right_ankle_speed,

        "left_ankle_speed":
            left_ankle_speed,

        "right_knee_speed":
            right_knee_speed,

        "left_knee_speed":
            left_knee_speed,

        "ankle_speed_difference":
            ankle_speed_difference,

        "hip_width_norm":
            hip_width_norm,

        "shoulder_width_norm":
            shoulder_width_norm,

        "knee_separation_norm":
            knee_separation_norm,

        "ankle_separation_norm":
            ankle_separation_norm,

        "right_leg_length_norm":
            right_leg_length_norm,

        "left_leg_length_norm":
            left_leg_length_norm,

        "right_ankle_speed_norm":
            right_ankle_speed_norm,

        "left_ankle_speed_norm":
            left_ankle_speed_norm,

        "right_knee_speed_norm":
            right_knee_speed_norm,

        "left_knee_speed_norm":
            left_knee_speed_norm,

        "ankle_speed_difference_norm":
            ankle_speed_difference_norm
    })

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    print("\nFeature shape:", features_df.shape)

    print(
        "NaN values:",
        features_df.isna().sum().sum()
    )

    print(
        "Infinite values:",
        np.isinf(
            features_df.select_dtypes(
                include=[np.number]
            ).to_numpy()
        ).sum()
    )

    # --------------------------------------------------
    # Save permanently
    # --------------------------------------------------

    features_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved full 240-frame features:")
    print(OUTPUT_PATH)

print("\nFirst 5 rows:")
print(features_df.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

FEATURE_PATH = (
    PROJECT +
    "/data/user_video/features/"
    "human_running_video_features_full.csv"
)

df = pd.read_csv(FEATURE_PATH)

# Combined ankle movement
df["combined_ankle_speed"] = (
    df["right_ankle_speed_norm"] +
    df["left_ankle_speed_norm"]
) / 2

# Smooth the signal slightly to reduce frame-to-frame noise
df["ankle_movement_smooth"] = (
    df["combined_ankle_speed"]
    .rolling(window=5, center=True, min_periods=1)
    .mean()
)

plt.figure(figsize=(14, 5))

plt.plot(
    df["frame"],
    df["right_ankle_speed_norm"],
    alpha=0.4,
    label="Right ankle"
)

plt.plot(
    df["frame"],
    df["left_ankle_speed_norm"],
    alpha=0.4,
    label="Left ankle"
)

plt.plot(
    df["frame"],
    df["ankle_movement_smooth"],
    linewidth=2,
    label="Combined ankle movement"
)

plt.xlabel("Frame")
plt.ylabel("Normalized ankle movement")
plt.title("Full Video — Ankle Movement")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

pose = np.load(POSE_PATH)

print("Pose shape:", pose.shape)

# H36M joints
PELVIS = 0
RIGHT_ANKLE = 3
LEFT_ANKLE = 6

pelvis = pose[:, PELVIS]

right_ankle = pose[:, RIGHT_ANKLE] - pelvis
left_ankle = pose[:, LEFT_ANKLE] - pelvis

# Smooth trajectories slightly
def smooth(signal, window=7):
    kernel = np.ones(window) / window
    return np.convolve(
        signal,
        kernel,
        mode="same"
    )

# Plot each coordinate
coordinate_names = ["X", "Y", "Z"]

for axis, name in enumerate(coordinate_names):

    right = smooth(right_ankle[:, axis])
    left = smooth(left_ankle[:, axis])

    plt.figure(figsize=(14, 5))

    plt.plot(
        right,
        label=f"Right ankle relative {name}"
    )

    plt.plot(
        left,
        label=f"Left ankle relative {name}"
    )

    plt.xlabel("Frame")
    plt.ylabel("Relative position")
    plt.title(
        f"Ankle Trajectory Relative to Pelvis — {name} Axis"
    )

    plt.legend()
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

pose = np.load(POSE_PATH)

PELVIS = 0
RIGHT_ANKLE = 3
LEFT_ANKLE = 6

pelvis = pose[:, PELVIS]

right_ankle = pose[:, RIGHT_ANKLE] - pelvis
left_ankle = pose[:, LEFT_ANKLE] - pelvis

def smooth(signal, window=7):
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")

for axis, name in [(0, "X"), (1, "Y")]:

    plt.figure(figsize=(14, 5))

    plt.plot(
        smooth(right_ankle[:, axis]),
        label="Right ankle"
    )

    plt.plot(
        smooth(left_ankle[:, axis]),
        label="Left ankle"
    )

    plt.xlabel("Frame")
    plt.ylabel("Relative position")
    plt.title(
        f"Ankle Trajectory Relative to Pelvis — {name} Axis"
    )

    plt.legend()
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

pose = np.load(POSE_PATH)

PELVIS = 0
RIGHT_ANKLE = 3
LEFT_ANKLE = 6

pelvis = pose[:, PELVIS]

right_ankle = pose[:, RIGHT_ANKLE] - pelvis
left_ankle = pose[:, LEFT_ANKLE] - pelvis

def smooth(signal, window=7):
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")

plt.figure(figsize=(14, 5))

plt.plot(
    smooth(right_ankle[:, 0]),
    label="Right ankle"
)

plt.plot(
    smooth(left_ankle[:, 0]),
    label="Left ankle"
)

plt.xlabel("Frame")
plt.ylabel("Relative position")
plt.title("Ankle Trajectory Relative to Pelvis — X Axis")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

pose = np.load(POSE_PATH)

print("Pose shape:", pose.shape)

# H36M skeleton connections
connections = [
    (0, 1), (1, 2), (2, 3),          # right leg
    (0, 4), (4, 5), (5, 6),          # left leg
    (0, 7), (7, 8), (8, 9), (9, 10), # spine/head
    (8, 11), (11, 12), (12, 13),     # left arm
    (8, 14), (14, 15), (15, 16)      # right arm
]

frame_numbers = [20, 60, 100, 140, 180, 220]

fig = plt.figure(figsize=(18, 10))

for plot_idx, frame in enumerate(frame_numbers):

    ax = fig.add_subplot(
        2, 3,
        plot_idx + 1,
        projection="3d"
    )

    # Root-relative pose
    skeleton = pose[frame] - pose[frame, 0]

    # Plot joints
    ax.scatter(
        skeleton[:, 0],
        skeleton[:, 1],
        skeleton[:, 2],
        s=30
    )

    # Plot skeleton connections
    for a, b in connections:

        ax.plot(
            [skeleton[a, 0], skeleton[b, 0]],
            [skeleton[a, 1], skeleton[b, 1]],
            [skeleton[a, 2], skeleton[b, 2]]
        )

    ax.set_title(f"Frame {frame}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Equal-ish scale for all frames
    max_range = np.max(
        np.ptp(skeleton, axis=0)
    ) / 2

    center = np.mean(
        skeleton,
        axis=0
    )

    ax.set_xlim(
        center[0] - max_range,
        center[0] + max_range
    )

    ax.set_ylim(
        center[1] - max_range,
        center[1] + max_range
    )

    ax.set_zlim(
        center[2] - max_range,
        center[2] + max_range
    )

plt.suptitle(
    "TCPFormer 3D Pose Across Running Video",
    fontsize=16
)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

COCO_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

coco = np.load(COCO_PATH)

print("COCO pose shape:", coco.shape)

# COCO indices
RIGHT_KNEE = 14
LEFT_KNEE = 13
RIGHT_ANKLE = 16
LEFT_ANKLE = 15
RIGHT_HIP = 12
LEFT_HIP = 11

frames = np.arange(len(coco))

plt.figure(figsize=(14, 5))

plt.plot(
    frames,
    coco[:, RIGHT_KNEE, 1],
    label="Right knee Y"
)

plt.plot(
    frames,
    coco[:, LEFT_KNEE, 1],
    label="Left knee Y"
)

plt.plot(
    frames,
    coco[:, RIGHT_ANKLE, 1],
    label="Right ankle Y"
)

plt.plot(
    frames,
    coco[:, LEFT_ANKLE, 1],
    label="Left ankle Y"
)

plt.xlabel("Frame")
plt.ylabel("Normalized image Y")
plt.title("MediaPipe COCO 2D — Vertical Leg Motion")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(14, 5))

plt.plot(
    frames,
    coco[:, RIGHT_ANKLE, 0],
    label="Right ankle X"
)

plt.plot(
    frames,
    coco[:, LEFT_ANKLE, 0],
    label="Left ankle X"
)

plt.xlabel("Frame")
plt.ylabel("Normalized image X")
plt.title("MediaPipe COCO 2D — Horizontal Ankle Motion")
plt.legend()
plt.grid(True)
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

VIDEO_PATH = (
    PROJECT +
    "/data/user_video/"
    "human_running_video.mp4"
)

COCO_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

coco = np.load(COCO_PATH)

print("COCO shape:", coco.shape)

# COCO-17 connections
connections = [
    (0, 5), (0, 6),                 # nose -> shoulders
    (5, 7), (7, 9),                 # left arm
    (6, 8), (8, 10),                # right arm
    (5, 11), (6, 12),               # shoulders -> hips
    (11, 13), (13, 15),             # left leg
    (12, 14), (14, 16),             # right leg
    (11, 12),                       # hips
    (5, 6)                          # shoulders
]

frame_numbers = [20, 60, 100, 140, 180, 220]

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video:", width, "x", height)
print("FPS:", fps)

fig, axes = plt.subplots(
    2, 3,
    figsize=(18, 10)
)

for ax, frame_idx in zip(
    axes.ravel(),
    frame_numbers
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_idx
    )

    success, frame = cap.read()

    if not success:
        ax.set_title(
            f"Frame {frame_idx} - FAILED"
        )
        ax.axis("off")
        continue

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Draw skeleton
    for a, b in connections:

        x1 = int(coco[frame_idx, a, 0] * width)
        y1 = int(coco[frame_idx, a, 1] * height)

        x2 = int(coco[frame_idx, b, 0] * width)
        y2 = int(coco[frame_idx, b, 1] * height)

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=2
        )

    # Draw joints
    for joint in range(17):

        x = int(
            coco[frame_idx, joint, 0] * width
        )

        y = int(
            coco[frame_idx, joint, 1] * height
        )

        ax.scatter(
            x,
            y,
            s=30
        )

    ax.imshow(frame)

    ax.set_title(
        f"Frame {frame_idx}"
    )

    ax.axis("off")

cap.release()

plt.suptitle(
    "MediaPipe COCO-17 Pose Overlay",
    fontsize=16
)

plt.tight_layout()
plt.show()

import os
import json
import numpy as np
import pandas as pd

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

REPORT_DIR = (
    PROJECT +
    "/data/user_video/results"
)

os.makedirs(REPORT_DIR, exist_ok=True)

pose = np.load(POSE_PATH)

print("Pose shape:", pose.shape)

# ---------------------------------------------------------
# COCO joint indices
# ---------------------------------------------------------

LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

LEFT_HIP = 11
RIGHT_HIP = 12

# ---------------------------------------------------------
# Basic visibility check
# ---------------------------------------------------------

valid_mask = np.isfinite(pose).all(axis=2)

frame_validity = valid_mask.mean(axis=1)

valid_frame_ratio = frame_validity.mean()

# ---------------------------------------------------------
# Calculate joint motion
# ---------------------------------------------------------

def frame_motion(joint_a, joint_b):
    """
    Mean frame-to-frame displacement for a joint pair.
    """
    a = pose[:, joint_a]
    b = pose[:, joint_b]

    midpoint = (a + b) / 2

    displacement = np.linalg.norm(
        np.diff(midpoint, axis=0),
        axis=1
    )

    return displacement


ankle_motion = (
    frame_motion(LEFT_ANKLE, RIGHT_ANKLE)
)

knee_motion = (
    frame_motion(LEFT_KNEE, RIGHT_KNEE)
)

hip_motion = (
    frame_motion(LEFT_HIP, RIGHT_HIP)
)

# ---------------------------------------------------------
# Body scale
# ---------------------------------------------------------

left_hip = pose[:, LEFT_HIP]
right_hip = pose[:, RIGHT_HIP]

hip_width = np.linalg.norm(
    left_hip - right_hip,
    axis=1
)

valid_scale = hip_width[
    np.isfinite(hip_width) &
    (hip_width > 0)
]

median_body_scale = np.median(valid_scale)

# Normalize motion by body scale
ankle_motion_norm = (
    np.median(ankle_motion) /
    median_body_scale
)

knee_motion_norm = (
    np.median(knee_motion) /
    median_body_scale
)

hip_motion_norm = (
    np.median(hip_motion) /
    median_body_scale
)

# ---------------------------------------------------------
# Overall motion score
# ---------------------------------------------------------

motion_score = np.mean([
    ankle_motion_norm,
    knee_motion_norm,
    hip_motion_norm
])

# ---------------------------------------------------------
# Running video requirements
# ---------------------------------------------------------

MIN_VALID_FRAME_RATIO = 0.90

# Conservative initial threshold.
# We will calibrate this later using good running videos.
MIN_MOTION_SCORE = 0.005

has_valid_pose = (
    valid_frame_ratio >= MIN_VALID_FRAME_RATIO
)

has_enough_motion = (
    motion_score >= MIN_MOTION_SCORE
)

video_accepted = (
    has_valid_pose and
    has_enough_motion
)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

report = {
    "video": "human_running_video.mp4",
    "total_frames": int(len(pose)),
    "valid_frame_ratio": float(valid_frame_ratio),

    "ankle_motion_normalized":
        float(ankle_motion_norm),

    "knee_motion_normalized":
        float(knee_motion_norm),

    "hip_motion_normalized":
        float(hip_motion_norm),

    "overall_motion_score":
        float(motion_score),

    "minimum_motion_score":
        float(MIN_MOTION_SCORE),

    "video_accepted":
        bool(video_accepted)
}

report_path = (
    REPORT_DIR +
    "/video_quality_report.json"
)

with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("\n===================================")
print("VIDEO QUALITY CHECK")
print("===================================")

print(
    f"Valid frame ratio : "
    f"{valid_frame_ratio:.3f}"
)

print(
    f"Ankle motion      : "
    f"{ankle_motion_norm:.5f}"
)

print(
    f"Knee motion       : "
    f"{knee_motion_norm:.5f}"
)

print(
    f"Hip motion        : "
    f"{hip_motion_norm:.5f}"
)

print(
    f"Overall motion    : "
    f"{motion_score:.5f}"
)

print(
    f"Minimum required  : "
    f"{MIN_MOTION_SCORE:.5f}"
)

print("-----------------------------------")

if video_accepted:
    print("VIDEO ACCEPTED")
else:
    print("VIDEO REJECTED")
    print(
        "Reason: insufficient continuous "
        "running motion or pose visibility."
    )

print("-----------------------------------")
print("Saved:", report_path)

import numpy as np
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

pose = np.load(POSE_PATH)

# COCO indices
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# ---------------------------------------------------------
# Pelvis center
# ---------------------------------------------------------

pelvis = (
    pose[:, LEFT_HIP] +
    pose[:, RIGHT_HIP]
) / 2

# ---------------------------------------------------------
# Relative joint positions
# ---------------------------------------------------------

left_ankle_rel = (
    pose[:, LEFT_ANKLE] - pelvis
)

right_ankle_rel = (
    pose[:, RIGHT_ANKLE] - pelvis
)

left_knee_rel = (
    pose[:, LEFT_KNEE] - pelvis
)

right_knee_rel = (
    pose[:, RIGHT_KNEE] - pelvis
)

# ---------------------------------------------------------
# Normalize by hip width
# ---------------------------------------------------------

hip_width = np.linalg.norm(
    pose[:, LEFT_HIP] -
    pose[:, RIGHT_HIP],
    axis=1
)

scale = np.median(
    hip_width[
        np.isfinite(hip_width) &
        (hip_width > 0)
    ]
)

left_ankle_rel /= scale
right_ankle_rel /= scale
left_knee_rel /= scale
right_knee_rel /= scale

# ---------------------------------------------------------
# Calculate actual articulation movement
# ---------------------------------------------------------

ankle_distance = np.linalg.norm(
    left_ankle_rel - right_ankle_rel,
    axis=1
)

knee_distance = np.linalg.norm(
    left_knee_rel - right_knee_rel,
    axis=1
)

# ---------------------------------------------------------
# Plot ankle trajectories
# ---------------------------------------------------------

frames = np.arange(len(pose))

plt.figure(figsize=(14, 5))

plt.plot(
    frames,
    right_ankle_rel[:, 0],
    label="Right ankle X"
)

plt.plot(
    frames,
    left_ankle_rel[:, 0],
    label="Left ankle X"
)

plt.xlabel("Frame")
plt.ylabel("Relative X position")
plt.title("Ankle Movement Relative to Pelvis — X")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(14, 5))

plt.plot(
    frames,
    right_ankle_rel[:, 1],
    label="Right ankle Y"
)

plt.plot(
    frames,
    left_ankle_rel[:, 1],
    label="Left ankle Y"
)

plt.xlabel("Frame")
plt.ylabel("Relative Y position")
plt.title("Ankle Movement Relative to Pelvis — Y")
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# Print movement ranges
# ---------------------------------------------------------

print("===================================")
print("RELATIVE RUNNING MOTION")
print("===================================")

print(
    f"Right ankle X range: "
    f"{np.ptp(right_ankle_rel[:,0]):.4f}"
)

print(
    f"Left ankle X range : "
    f"{np.ptp(left_ankle_rel[:,0]):.4f}"
)

print(
    f"Right ankle Y range: "
    f"{np.ptp(right_ankle_rel[:,1]):.4f}"
)

print(
    f"Left ankle Y range : "
    f"{np.ptp(left_ankle_rel[:,1]):.4f}"
)

print(
    f"Ankle separation range: "
    f"{np.ptp(ankle_distance):.4f}"
)

print(
    f"Knee separation range: "
    f"{np.ptp(knee_distance):.4f}"
)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

RESULT_DIR = (
    PROJECT +
    "/data/user_video/results/gait_cycles"
)

os.makedirs(RESULT_DIR, exist_ok=True)

pose = np.load(POSE_PATH)

# ---------------------------------------------------------
# COCO indices
# ---------------------------------------------------------

LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# ---------------------------------------------------------
# Pelvis
# ---------------------------------------------------------

pelvis = (
    pose[:, LEFT_HIP] +
    pose[:, RIGHT_HIP]
) / 2

# ---------------------------------------------------------
# Ankle positions relative to pelvis
# ---------------------------------------------------------

left_ankle = (
    pose[:, LEFT_ANKLE] - pelvis
)

right_ankle = (
    pose[:, RIGHT_ANKLE] - pelvis
)

# ---------------------------------------------------------
# Body-scale normalization
# ---------------------------------------------------------

hip_width = np.linalg.norm(
    pose[:, LEFT_HIP] -
    pose[:, RIGHT_HIP],
    axis=1
)

scale = np.median(
    hip_width[
        np.isfinite(hip_width) &
        (hip_width > 0)
    ]
)

left_ankle /= scale
right_ankle /= scale

# ---------------------------------------------------------
# Combined ankle vertical signal
# ---------------------------------------------------------

ankle_y = (
    np.abs(left_ankle[:, 1]) +
    np.abs(right_ankle[:, 1])
) / 2

# Smooth signal
window = 15

if window >= len(ankle_y):
    window = len(ankle_y) - 1

if window % 2 == 0:
    window -= 1

ankle_y_smooth = savgol_filter(
    ankle_y,
    window_length=window,
    polyorder=2
)

# ---------------------------------------------------------
# Detect candidate gait events
# ---------------------------------------------------------

# Minimum distance between events.
# At 24 FPS this prevents many tiny fluctuations
# from being treated as separate strides.
min_distance = int(0.45 * 24)

peaks, peak_properties = find_peaks(
    ankle_y_smooth,
    distance=min_distance,
    prominence=0.05
)

troughs, trough_properties = find_peaks(
    -ankle_y_smooth,
    distance=min_distance,
    prominence=0.05
)

print("===================================")
print("GAIT EVENT DETECTION")
print("===================================")

print("Detected peaks   :", peaks)
print("Detected troughs :", troughs)

print(
    "\nPeak count   :",
    len(peaks)
)

print(
    "Trough count :",
    len(troughs)
)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

frames = np.arange(len(ankle_y_smooth))

plt.figure(figsize=(15, 6))

plt.plot(
    frames,
    ankle_y,
    alpha=0.35,
    label="Raw ankle signal"
)

plt.plot(
    frames,
    ankle_y_smooth,
    linewidth=2,
    label="Smoothed ankle signal"
)

plt.scatter(
    peaks,
    ankle_y_smooth[peaks],
    s=70,
    label="Candidate peaks"
)

plt.scatter(
    troughs,
    ankle_y_smooth[troughs],
    s=70,
    label="Candidate troughs"
)

plt.xlabel("Frame")
plt.ylabel("Pelvis-relative ankle position")
plt.title("Candidate Running Gait Events")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# Save events
# ---------------------------------------------------------

events_df = pd.DataFrame({
    "peak_frame": pd.Series(peaks),
    "trough_frame": pd.Series(troughs)
})

events_path = (
    RESULT_DIR +
    "/candidate_gait_events.csv"
)

events_df.to_csv(
    events_path,
    index=False
)

np.save(
    RESULT_DIR + "/ankle_signal_smooth.npy",
    ankle_y_smooth
)

print("\nSaved:")
print(events_path)
print(
    RESULT_DIR +
    "/ankle_signal_smooth.npy"
)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

RESULT_DIR = (
    PROJECT +
    "/data/user_video/results/gait_cycles"
)

os.makedirs(RESULT_DIR, exist_ok=True)

pose = np.load(POSE_PATH)

# ---------------------------------------------------------
# COCO joints
# ---------------------------------------------------------

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# ---------------------------------------------------------
# Pelvis
# ---------------------------------------------------------

pelvis = (
    pose[:, LEFT_HIP] +
    pose[:, RIGHT_HIP]
) / 2

# ---------------------------------------------------------
# Ankle positions relative to pelvis
# ---------------------------------------------------------

left_ankle = pose[:, LEFT_ANKLE] - pelvis
right_ankle = pose[:, RIGHT_ANKLE] - pelvis

# ---------------------------------------------------------
# Normalize by body scale
# ---------------------------------------------------------

hip_width = np.linalg.norm(
    pose[:, LEFT_HIP] -
    pose[:, RIGHT_HIP],
    axis=1
)

scale = np.median(
    hip_width[
        np.isfinite(hip_width) &
        (hip_width > 0)
    ]
)

left_ankle = left_ankle / scale
right_ankle = right_ankle / scale

# ---------------------------------------------------------
# Vertical signals
# ---------------------------------------------------------

left_y = left_ankle[:, 1]
right_y = right_ankle[:, 1]

# Smooth
window = 15

left_smooth = savgol_filter(
    left_y,
    window_length=window,
    polyorder=2
)

right_smooth = savgol_filter(
    right_y,
    window_length=window,
    polyorder=2
)

# ---------------------------------------------------------
# Adaptive prominence
# ---------------------------------------------------------

left_prominence = (
    0.20 *
    np.std(left_smooth)
)

right_prominence = (
    0.20 *
    np.std(right_smooth)
)

# At 24 FPS, prevent events closer than ~0.5 sec.
min_distance = int(0.45 * 24)

# ---------------------------------------------------------
# Detect peaks
# ---------------------------------------------------------

left_peaks, left_properties = find_peaks(
    left_smooth,
    distance=min_distance,
    prominence=left_prominence
)

right_peaks, right_properties = find_peaks(
    right_smooth,
    distance=min_distance,
    prominence=right_prominence
)

# ---------------------------------------------------------
# Detect troughs
# ---------------------------------------------------------

left_troughs, _ = find_peaks(
    -left_smooth,
    distance=min_distance,
    prominence=left_prominence
)

right_troughs, _ = find_peaks(
    -right_smooth,
    distance=min_distance,
    prominence=right_prominence
)

print("===================================")
print("SEPARATE FOOT GAIT EVENTS")
print("===================================")

print("\nLEFT FOOT")
print("Peaks  :", left_peaks)
print("Troughs:", left_troughs)

print("\nRIGHT FOOT")
print("Peaks  :", right_peaks)
print("Troughs:", right_troughs)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

frames = np.arange(len(left_smooth))

fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 9),
    sharex=True
)

# LEFT
axes[0].plot(
    frames,
    left_y,
    alpha=0.25,
    label="Left ankle raw"
)

axes[0].plot(
    frames,
    left_smooth,
    linewidth=2,
    label="Left ankle smoothed"
)

axes[0].scatter(
    left_peaks,
    left_smooth[left_peaks],
    s=70,
    label="Left peaks"
)

axes[0].scatter(
    left_troughs,
    left_smooth[left_troughs],
    s=70,
    label="Left troughs"
)

axes[0].set_title(
    "Left Ankle — Pelvis Relative Vertical Motion"
)

axes[0].set_ylabel("Relative Y")
axes[0].legend()
axes[0].grid(True)

# RIGHT
axes[1].plot(
    frames,
    right_y,
    alpha=0.25,
    label="Right ankle raw"
)

axes[1].plot(
    frames,
    right_smooth,
    linewidth=2,
    label="Right ankle smoothed"
)

axes[1].scatter(
    right_peaks,
    right_smooth[right_peaks],
    s=70,
    label="Right peaks"
)

axes[1].scatter(
    right_troughs,
    right_smooth[right_troughs],
    s=70,
    label="Right troughs"
)

axes[1].set_title(
    "Right Ankle — Pelvis Relative Vertical Motion"
)

axes[1].set_xlabel("Frame")
axes[1].set_ylabel("Relative Y")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

np.save(
    RESULT_DIR + "/left_ankle_relative_y.npy",
    left_smooth
)

np.save(
    RESULT_DIR + "/right_ankle_relative_y.npy",
    right_smooth
)

np.save(
    RESULT_DIR + "/left_ankle_peaks.npy",
    left_peaks
)

np.save(
    RESULT_DIR + "/right_ankle_peaks.npy",
    right_peaks
)

np.save(
    RESULT_DIR + "/left_ankle_troughs.npy",
    left_troughs
)

np.save(
    RESULT_DIR + "/right_ankle_troughs.npy",
    right_troughs
)

print("\nSaved separate foot gait signals and events.")

import os
import numpy as np
import pandas as pd

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

RESULT_DIR = (
    PROJECT +
    "/data/user_video/results/gait_cycles"
)

os.makedirs(RESULT_DIR, exist_ok=True)

# Load detected events
left_peaks = np.load(
    RESULT_DIR + "/left_ankle_peaks.npy"
)

right_peaks = np.load(
    RESULT_DIR + "/right_ankle_peaks.npy"
)

left_troughs = np.load(
    RESULT_DIR + "/left_ankle_troughs.npy"
)

right_troughs = np.load(
    RESULT_DIR + "/right_ankle_troughs.npy"
)

print("===================================")
print("GAIT CYCLE CANDIDATE ANALYSIS")
print("===================================")

print("Left peaks :", left_peaks)
print("Right peaks:", right_peaks)

# ---------------------------------------------------------
# Calculate intervals between same-foot events
# ---------------------------------------------------------

left_intervals = np.diff(left_peaks)
right_intervals = np.diff(right_peaks)

print("\nLeft peak intervals:")
print(left_intervals)

print("\nRight peak intervals:")
print(right_intervals)

# ---------------------------------------------------------
# Keep plausible running cycles
#
# Video = 24 FPS
#
# We reject extremely short intervals.
# Approximate acceptable range:
# 0.55 sec -> 1.60 sec
#
# At 24 FPS:
# 13 -> 38 frames
# ---------------------------------------------------------

MIN_CYCLE_FRAMES = int(0.55 * 24)
MAX_CYCLE_FRAMES = int(1.60 * 24)

print("\nAccepted interval range:")
print(
    MIN_CYCLE_FRAMES,
    "to",
    MAX_CYCLE_FRAMES,
    "frames"
)

# ---------------------------------------------------------
# Right-foot candidate cycles
# ---------------------------------------------------------

right_cycles = []

for i in range(len(right_peaks) - 1):

    start = int(right_peaks[i])
    end = int(right_peaks[i + 1])

    length = end - start

    if MIN_CYCLE_FRAMES <= length <= MAX_CYCLE_FRAMES:

        right_cycles.append({
            "cycle_id": len(right_cycles),
            "foot": "right",
            "start_frame": start,
            "end_frame": end,
            "length_frames": length,
            "duration_sec": length / 24.0
        })

# ---------------------------------------------------------
# Left-foot candidate cycles
# ---------------------------------------------------------

left_cycles = []

for i in range(len(left_peaks) - 1):

    start = int(left_peaks[i])
    end = int(left_peaks[i + 1])

    length = end - start

    if MIN_CYCLE_FRAMES <= length <= MAX_CYCLE_FRAMES:

        left_cycles.append({
            "cycle_id": len(left_cycles),
            "foot": "left",
            "start_frame": start,
            "end_frame": end,
            "length_frames": length,
            "duration_sec": length / 24.0
        })

# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------

cycles = right_cycles + left_cycles

cycles_df = pd.DataFrame(cycles)

print("\n===================================")
print("CANDIDATE CYCLES")
print("===================================")

print(cycles_df)

print(
    "\nRight candidate cycles:",
    len(right_cycles)
)

print(
    "Left candidate cycles:",
    len(left_cycles)
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_path = (
    RESULT_DIR +
    "/candidate_gait_cycles.csv"
)

cycles_df.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

RESULT_DIR = (
    PROJECT +
    "/data/user_video/results/gait_cycles"
)

candidate_path = (
    RESULT_DIR +
    "/candidate_gait_cycles.csv"
)

cycles_df = pd.read_csv(candidate_path)

# ---------------------------------------------------------
# Estimate dominant cycle duration
# ---------------------------------------------------------

lengths = cycles_df["length_frames"].values

median_length = np.median(lengths)

print("===================================")
print("STABLE GAIT CYCLE SELECTION")
print("===================================")

print("All candidate cycle lengths:")
print(lengths)

print("\nMedian cycle length:", median_length)

# ---------------------------------------------------------
# Accept cycles reasonably close to dominant duration
#
# Running cycles in this video appear around ~16-25 frames.
# We use a 30% tolerance around the median.
# ---------------------------------------------------------

lower_bound = median_length * 0.70
upper_bound = median_length * 1.30

print(
    "\nStable range:",
    round(lower_bound, 2),
    "to",
    round(upper_bound, 2),
    "frames"
)

stable_df = cycles_df[
    (cycles_df["length_frames"] >= lower_bound) &
    (cycles_df["length_frames"] <= upper_bound)
].copy()

# Re-number
stable_df["stable_cycle_id"] = np.arange(len(stable_df))

print("\n===================================")
print("STABLE CYCLES")
print("===================================")

print(
    stable_df[
        [
            "stable_cycle_id",
            "foot",
            "start_frame",
            "end_frame",
            "length_frames",
            "duration_sec"
        ]
    ].to_string(index=False)
)

print("\nTotal stable cycles:", len(stable_df))

print(
    "Stable right cycles:",
    len(stable_df[stable_df["foot"] == "right"])
)

print(
    "Stable left cycles:",
    len(stable_df[stable_df["foot"] == "left"])
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

stable_path = (
    RESULT_DIR +
    "/stable_gait_cycles.csv"
)

stable_df.to_csv(
    stable_path,
    index=False
)

print("\nSaved:")
print(stable_path)

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

plt.figure(figsize=(15, 5))

for _, row in stable_df.iterrows():

    start = int(row["start_frame"])
    end = int(row["end_frame"])

    if row["foot"] == "right":
        plt.axvspan(
            start,
            end,
            alpha=0.18
        )
    else:
        plt.axvspan(
            start,
            end,
            alpha=0.10
        )

    plt.text(
        (start + end) / 2,
        0.5,
        f'{row["foot"][0].upper()}{int(row["stable_cycle_id"])}',
        ha="center",
        va="center"
    )

plt.xlim(0, 240)
plt.yticks([])
plt.xlabel("Frame")
plt.title("Stable Candidate Running Gait Cycles")
plt.grid(True, axis="x")

plt.tight_layout()
plt.show()

import os
import numpy as np
import pandas as pd

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

POSE_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

FEATURE_PATH = (
    PROJECT +
    "/data/user_video/features/"
    "human_running_video_features_full.csv"
)

CYCLES_PATH = (
    PROJECT +
    "/data/user_video/results/gait_cycles/"
    "stable_gait_cycles.csv"
)

OUTPUT_DIR = (
    PROJECT +
    "/data/user_video/results/gait_cycles/"
    "normalized_101"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

pose = np.load(POSE_PATH)
features = pd.read_csv(FEATURE_PATH)
cycles = pd.read_csv(CYCLES_PATH)

print("===================================")
print("GAIT CYCLE NORMALIZATION")
print("===================================")

print("3D pose:", pose.shape)
print("Features:", features.shape)
print("Stable cycles:", len(cycles))

# ---------------------------------------------------------
# Helper: interpolate sequence to 101 points
# ---------------------------------------------------------

def interpolate_sequence(sequence, target_points=101):

    old_length = len(sequence)

    if old_length < 2:
        raise ValueError(
            "A gait cycle must contain at least 2 frames."
        )

    old_x = np.linspace(
        0,
        100,
        old_length
    )

    new_x = np.linspace(
        0,
        100,
        target_points
    )

    # -----------------------------------------------------
    # 1D
    # -----------------------------------------------------

    if sequence.ndim == 1:

        return np.interp(
            new_x,
            old_x,
            sequence
        )

    # -----------------------------------------------------
    # 2D
    # -----------------------------------------------------

    if sequence.ndim == 2:

        output = np.zeros(
            (target_points, sequence.shape[1]),
            dtype=np.float32
        )

        for j in range(sequence.shape[1]):

            output[:, j] = np.interp(
                new_x,
                old_x,
                sequence[:, j]
            )

        return output

    # -----------------------------------------------------
    # 3D
    # -----------------------------------------------------

    if sequence.ndim == 3:

        output = np.zeros(
            (
                target_points,
                sequence.shape[1],
                sequence.shape[2]
            ),
            dtype=np.float32
        )

        for j in range(sequence.shape[1]):

            for k in range(sequence.shape[2]):

                output[:, j, k] = np.interp(
                    new_x,
                    old_x,
                    sequence[:, j, k]
                )

        return output

    raise ValueError(
        f"Unsupported sequence dimensions: {sequence.ndim}"
    )


# ---------------------------------------------------------
# Process every stable gait cycle
# ---------------------------------------------------------

normalized_pose_cycles = []
normalized_feature_cycles = []

cycle_metadata = []

for _, row in cycles.iterrows():

    cycle_id = int(row["stable_cycle_id"])
    foot = row["foot"]

    start = int(row["start_frame"])
    end = int(row["end_frame"])

    # Include both start and end frames
    pose_sequence = pose[start:end + 1]

    feature_sequence = features.iloc[
        start:end + 1
    ].drop(
        columns=["frame"]
    ).values

    # -----------------------------------------------------
    # Normalize to 101 gait-phase points
    # -----------------------------------------------------

    pose_101 = interpolate_sequence(
        pose_sequence,
        target_points=101
    )

    feature_101 = interpolate_sequence(
        feature_sequence,
        target_points=101
    )

    normalized_pose_cycles.append(
        pose_101
    )

    normalized_feature_cycles.append(
        feature_101
    )

    cycle_metadata.append({
        "normalized_cycle_id": len(cycle_metadata),
        "original_stable_cycle_id": cycle_id,
        "foot": foot,
        "start_frame": start,
        "end_frame": end,
        "original_frames": len(pose_sequence),
        "normalized_points": 101
    })


# ---------------------------------------------------------
# Convert to arrays
# ---------------------------------------------------------

normalized_pose_cycles = np.stack(
    normalized_pose_cycles
)

normalized_feature_cycles = np.stack(
    normalized_feature_cycles
)

metadata_df = pd.DataFrame(
    cycle_metadata
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

POSE_OUTPUT = (
    OUTPUT_DIR +
    "/user_gait_cycles_101_pose.npy"
)

FEATURE_OUTPUT = (
    OUTPUT_DIR +
    "/user_gait_cycles_101_features.npy"
)

METADATA_OUTPUT = (
    OUTPUT_DIR +
    "/user_gait_cycles_101_metadata.csv"
)

np.save(
    POSE_OUTPUT,
    normalized_pose_cycles
)

np.save(
    FEATURE_OUTPUT,
    normalized_feature_cycles
)

metadata_df.to_csv(
    METADATA_OUTPUT,
    index=False
)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("\n===================================")
print("NORMALIZATION COMPLETE")
print("===================================")

print(
    "Normalized 3D pose shape:",
    normalized_pose_cycles.shape
)

print(
    "Normalized feature shape:",
    normalized_feature_cycles.shape
)

print(
    "Expected pose shape:",
    f"({len(cycles)}, 101, 17, 3)"
)

print(
    "Expected feature shape:",
    f"({len(cycles)}, 101, 28)"
)

print("\nCycle metadata:")
print(metadata_df.to_string(index=False))

print("\nSaved:")
print(POSE_OUTPUT)
print(FEATURE_OUTPUT)
print(METADATA_OUTPUT)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

NORMALIZED_DIR = (
    PROJECT +
    "/data/user_video/results/gait_cycles/"
    "normalized_101"
)

POSE_PATH = (
    NORMALIZED_DIR +
    "/user_gait_cycles_101_pose.npy"
)

FEATURE_PATH = (
    NORMALIZED_DIR +
    "/user_gait_cycles_101_features.npy"
)

META_PATH = (
    NORMALIZED_DIR +
    "/user_gait_cycles_101_metadata.csv"
)

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

pose_cycles = np.load(POSE_PATH)
feature_cycles = np.load(FEATURE_PATH)
metadata = pd.read_csv(META_PATH)

print("===================================")
print("NORMALIZED GAIT CYCLE VALIDATION")
print("===================================")

print("Pose:", pose_cycles.shape)
print("Features:", feature_cycles.shape)
print("Cycles:", len(metadata))

# ---------------------------------------------------------
# Feature column mapping
#
# Original feature CSV had:
# frame + 28 numerical features
# ---------------------------------------------------------

FEATURE_NAMES = [
    "right_knee_angle",
    "left_knee_angle",
    "right_hip_angle",
    "left_hip_angle",
    "knee_angle_difference",
    "hip_angle_difference",
    "hip_width",
    "shoulder_width",
    "right_leg_length",
    "left_leg_length",
    "knee_separation",
    "ankle_separation",
    "right_ankle_speed",
    "left_ankle_speed",
    "right_knee_speed",
    "left_knee_speed",
    "ankle_speed_difference",
    "hip_width_norm",
    "shoulder_width_norm",
    "knee_separation_norm",
    "ankle_separation_norm",
    "right_leg_length_norm",
    "left_leg_length_norm",
    "right_ankle_speed_norm",
    "left_ankle_speed_norm",
    "right_knee_speed_norm",
    "left_knee_speed_norm",
    "ankle_speed_difference_norm"
]

# ---------------------------------------------------------
# Find right / left cycles
# ---------------------------------------------------------

right_mask = (
    metadata["foot"].values == "right"
)

left_mask = (
    metadata["foot"].values == "left"
)

right_features = feature_cycles[
    right_mask
]

left_features = feature_cycles[
    left_mask
]

print("\nRight cycles:", len(right_features))
print("Left cycles :", len(left_features))

# ---------------------------------------------------------
# Feature indices
# ---------------------------------------------------------

right_knee_idx = FEATURE_NAMES.index(
    "right_knee_angle"
)

left_knee_idx = FEATURE_NAMES.index(
    "left_knee_angle"
)

# ---------------------------------------------------------
# Extract knee-angle curves
# ---------------------------------------------------------

right_knee = right_features[
    :, :,
    right_knee_idx
]

left_knee = left_features[
    :, :,
    left_knee_idx
]

gait_phase = np.linspace(
    0,
    100,
    101
)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

right_mean = np.mean(
    right_knee,
    axis=0
)

right_std = np.std(
    right_knee,
    axis=0
)

left_mean = np.mean(
    left_knee,
    axis=0
)

left_std = np.std(
    left_knee,
    axis=0
)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 10),
    sharex=True
)

# =========================================================
# RIGHT KNEE
# =========================================================

for cycle in right_knee:

    axes[0].plot(
        gait_phase,
        cycle,
        alpha=0.35
    )

axes[0].plot(
    gait_phase,
    right_mean,
    linewidth=3,
    label="Right knee mean"
)

axes[0].fill_between(
    gait_phase,
    right_mean - right_std,
    right_mean + right_std,
    alpha=0.20,
    label="±1 SD"
)

axes[0].set_title(
    "User Running — Right Knee Angle Across Normalized Gait Cycles"
)

axes[0].set_ylabel(
    "Knee angle (degrees)"
)

axes[0].legend()
axes[0].grid(True)

# =========================================================
# LEFT KNEE
# =========================================================

for cycle in left_knee:

    axes[1].plot(
        gait_phase,
        cycle,
        alpha=0.35
    )

axes[1].plot(
    gait_phase,
    left_mean,
    linewidth=3,
    label="Left knee mean"
)

axes[1].fill_between(
    gait_phase,
    left_mean - left_std,
    left_mean + left_std,
    alpha=0.20,
    label="±1 SD"
)

axes[1].set_title(
    "User Running — Left Knee Angle Across Normalized Gait Cycles"
)

axes[1].set_xlabel(
    "Gait cycle (%)"
)

axes[1].set_ylabel(
    "Knee angle (degrees)"
)

axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("\n===================================")
print("KNEE ANGLE SUMMARY")
print("===================================")

print(
    f"Right knee overall mean: "
    f"{np.mean(right_knee):.2f}°"
)

print(
    f"Right knee overall std : "
    f"{np.std(right_knee):.2f}°"
)

print(
    f"Left knee overall mean : "
    f"{np.mean(left_knee):.2f}°"
)

print(
    f"Left knee overall std  : "
    f"{np.std(left_knee):.2f}°"
)

print("\nValidation complete.")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

USER_FEATURE_PATH = (
    PROJECT +
    "/data/user_video/results/gait_cycles/"
    "normalized_101/user_gait_cycles_101_features.npy"
)

USER_META_PATH = (
    PROJECT +
    "/data/user_video/results/gait_cycles/"
    "normalized_101/user_gait_cycles_101_metadata.csv"
)

REFERENCE_PATH = (
    PROJECT +
    "/data/references/RBDS/reference_curves/"
    "RBDS_running_reference_curves.csv"
)

OUTPUT_DIR = (
    PROJECT +
    "/data/user_video/results/"
    "reference_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

user_features = np.load(USER_FEATURE_PATH)
metadata = pd.read_csv(USER_META_PATH)
reference = pd.read_csv(REFERENCE_PATH)

print("===================================")
print("USER vs RBDS KNEE FLEXION")
print("===================================")

print("User features:", user_features.shape)
print("Reference:", reference.shape)

# ---------------------------------------------------------
# User feature indices
# ---------------------------------------------------------

FEATURE_NAMES = [
    "right_knee_angle",
    "left_knee_angle",
    "right_hip_angle",
    "left_hip_angle",
    "knee_angle_difference",
    "hip_angle_difference",
    "hip_width",
    "shoulder_width",
    "right_leg_length",
    "left_leg_length",
    "knee_separation",
    "ankle_separation",
    "right_ankle_speed",
    "left_ankle_speed",
    "right_knee_speed",
    "left_knee_speed",
    "ankle_speed_difference",
    "hip_width_norm",
    "shoulder_width_norm",
    "knee_separation_norm",
    "ankle_separation_norm",
    "right_leg_length_norm",
    "left_leg_length_norm",
    "right_ankle_speed_norm",
    "left_ankle_speed_norm",
    "right_knee_speed_norm",
    "left_knee_speed_norm",
    "ankle_speed_difference_norm"
]

right_knee_idx = FEATURE_NAMES.index(
    "right_knee_angle"
)

left_knee_idx = FEATURE_NAMES.index(
    "left_knee_angle"
)

# ---------------------------------------------------------
# Separate user's right and left gait cycles
# ---------------------------------------------------------

right_mask = (
    metadata["foot"].values == "right"
)

left_mask = (
    metadata["foot"].values == "left"
)

user_right_knee_angle = user_features[
    right_mask,
    :,
    right_knee_idx
]

user_left_knee_angle = user_features[
    left_mask,
    :,
    left_knee_idx
]

# ---------------------------------------------------------
# Convert included knee angle to approximate flexion
#
# Straight knee ≈ 180 degrees
# Flexion ≈ 180 - included angle
# ---------------------------------------------------------

user_right_flexion = (
    180.0 - user_right_knee_angle
)

user_left_flexion = (
    180.0 - user_left_knee_angle
)

# Mean user curves
user_right_mean = np.mean(
    user_right_flexion,
    axis=0
)

user_left_mean = np.mean(
    user_left_flexion,
    axis=0
)

# ---------------------------------------------------------
# Gait phase
# ---------------------------------------------------------

phase = np.arange(101)

# ---------------------------------------------------------
# Compare each RBDS speed
# ---------------------------------------------------------

speeds = [
    "2.5mps",
    "3.5mps",
    "4.5mps"
]

comparison_rows = []

for speed in speeds:

    print("\nProcessing:", speed)

    # -----------------------------
    # Right knee
    # -----------------------------

    rbds_right = reference[
        (reference["speed"] == speed) &
        (reference["joint"] == "knee") &
        (reference["side"] == "right") &
        (reference["axis"] == "Z")
    ].sort_values(
        "gait_cycle_percent"
    )

    # -----------------------------
    # Left knee
    # -----------------------------

    rbds_left = reference[
        (reference["speed"] == speed) &
        (reference["joint"] == "knee") &
        (reference["side"] == "left") &
        (reference["axis"] == "Z")
    ].sort_values(
        "gait_cycle_percent"
    )

    if len(rbds_right) != 101:
        print(
            "WARNING: right reference has",
            len(rbds_right),
            "points"
        )

    if len(rbds_left) != 101:
        print(
            "WARNING: left reference has",
            len(rbds_left),
            "points"
        )

    # -----------------------------
    # Reference curves
    # -----------------------------

    right_mean = rbds_right["mean"].values
    right_std = rbds_right["std"].values
    right_p05 = rbds_right["p05"].values
    right_p95 = rbds_right["p95"].values

    left_mean = rbds_left["mean"].values
    left_std = rbds_left["std"].values
    left_p05 = rbds_left["p05"].values
    left_p95 = rbds_left["p95"].values

    # -----------------------------------------------------
    # Deviations
    # -----------------------------------------------------

    right_difference = (
        user_right_mean -
        right_mean
    )

    left_difference = (
        user_left_mean -
        left_mean
    )

    # -----------------------------------------------------
    # Outside 5th-95th percentile reference band
    # -----------------------------------------------------

    right_outside = (
        (user_right_mean < right_p05) |
        (user_right_mean > right_p95)
    )

    left_outside = (
        (user_left_mean < left_p05) |
        (user_left_mean > left_p95)
    )

    right_outside_percent = (
        np.mean(right_outside) * 100
    )

    left_outside_percent = (
        np.mean(left_outside) * 100
    )

    # -----------------------------------------------------
    # RMSE
    # -----------------------------------------------------

    right_rmse = np.sqrt(
        np.mean(
            right_difference ** 2
        )
    )

    left_rmse = np.sqrt(
        np.mean(
            left_difference ** 2
        )
    )

    # -----------------------------------------------------
    # Mean absolute deviation
    # -----------------------------------------------------

    right_mad = np.mean(
        np.abs(right_difference)
    )

    left_mad = np.mean(
        np.abs(left_difference)
    )

    print(
        f"Right knee: "
        f"RMSE={right_rmse:.2f}°, "
        f"MAD={right_mad:.2f}°, "
        f"outside={right_outside_percent:.1f}%"
    )

    print(
        f"Left knee : "
        f"RMSE={left_rmse:.2f}°, "
        f"MAD={left_mad:.2f}°, "
        f"outside={left_outside_percent:.1f}%"
    )

    # -----------------------------------------------------
    # Store phase-by-phase results
    # -----------------------------------------------------

    for i in range(101):

        comparison_rows.append({
            "speed": speed,
            "gait_cycle_percent": i,

            "user_right_knee_flexion":
                user_right_mean[i],

            "rbds_right_knee_flexion":
                right_mean[i],

            "rbds_right_std":
                right_std[i],

            "rbds_right_p05":
                right_p05[i],

            "rbds_right_p95":
                right_p95[i],

            "right_deviation":
                right_difference[i],

            "right_outside_reference":
                bool(right_outside[i]),

            "user_left_knee_flexion":
                user_left_mean[i],

            "rbds_left_knee_flexion":
                left_mean[i],

            "rbds_left_std":
                left_std[i],

            "rbds_left_p05":
                left_p05[i],

            "rbds_left_p95":
                left_p95[i],

            "left_deviation":
                left_difference[i],

            "left_outside_reference":
                bool(left_outside[i])
        })

# ---------------------------------------------------------
# Save comparison
# ---------------------------------------------------------

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_path = (
    OUTPUT_DIR +
    "/user_vs_RBDS_knee_flexion.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("\n===================================")
print("COMPARISON COMPLETE")
print("===================================")

print(
    "Comparison shape:",
    comparison_df.shape
)

print(
    "Saved:",
    comparison_path
)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10),
    sharex=True
)

for col, speed in enumerate(speeds):

    data = comparison_df[
        comparison_df["speed"] == speed
    ]

    # -----------------------------
    # RIGHT
    # -----------------------------

    ax = axes[0, col]

    x = data["gait_cycle_percent"].values

    user = data[
        "user_right_knee_flexion"
    ].values

    ref_mean = data[
        "rbds_right_knee_flexion"
    ].values

    p05 = data[
        "rbds_right_p05"
    ].values

    p95 = data[
        "rbds_right_p95"
    ].values

    ax.fill_between(
        x,
        p05,
        p95,
        alpha=0.20,
        label="RBDS 5–95%"
    )

    ax.plot(
        x,
        ref_mean,
        linewidth=2,
        label="RBDS mean"
    )

    ax.plot(
        x,
        user,
        linewidth=2,
        label="User mean"
    )

    ax.set_title(
        f"Right Knee — {speed}"
    )

    ax.set_ylabel(
        "Knee flexion (degrees)"
    )

    ax.grid(True)

    # -----------------------------
    # LEFT
    # -----------------------------

    ax = axes[1, col]

    user = data[
        "user_left_knee_flexion"
    ].values

    ref_mean = data[
        "rbds_left_knee_flexion"
    ].values

    p05 = data[
        "rbds_left_p05"
    ].values

    p95 = data[
        "rbds_left_p95"
    ].values

    ax.fill_between(
        x,
        p05,
        p95,
        alpha=0.20,
        label="RBDS 5–95%"
    )

    ax.plot(
        x,
        ref_mean,
        linewidth=2,
        label="RBDS mean"
    )

    ax.plot(
        x,
        user,
        linewidth=2,
        label="User mean"
    )

    ax.set_title(
        f"Left Knee — {speed}"
    )

    ax.set_xlabel(
        "Gait cycle (%)"
    )

    ax.set_ylabel(
        "Knee flexion (degrees)"
    )

    ax.grid(True)

axes[0, 0].legend()
axes[1, 0].legend()

plt.tight_layout()

plot_path = (
    OUTPUT_DIR +
    "/user_vs_RBDS_knee_flexion.png"
)

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nSaved plot:")
print(plot_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

COCO_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

POSE3D_PATH = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

OUT_DIR = (
    PROJECT +
    "/data/user_video/results/reference_comparison"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load existing data
# ---------------------------------------------------------

coco = np.load(COCO_PATH)
pose3d = np.load(POSE3D_PATH)

print("===================================")
print("2D vs 3D KNEE MOTION DIAGNOSTIC")
print("===================================")

print("COCO 2D :", coco.shape)
print("3D pose :", pose3d.shape)

# ---------------------------------------------------------
# COCO indices
#
# 13 = left knee
# 14 = right knee
# 15 = left ankle
# 16 = right ankle
# 11 = left hip
# 12 = right hip
# ---------------------------------------------------------

# ---------------------------------------------------------
# Function: angle at knee
# ---------------------------------------------------------

def calculate_angle(a, b, c):
    """
    Angle ABC.
    a = hip
    b = knee
    c = ankle
    """

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba, axis=1) *
        np.linalg.norm(bc, axis=1)
    )

    cosine = np.sum(
        ba * bc,
        axis=1
    ) / np.maximum(
        denominator,
        1e-8
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine)
    )

# ---------------------------------------------------------
# 2D knee angles
# ---------------------------------------------------------

right_knee_2d = calculate_angle(
    coco[:, 12, :],
    coco[:, 14, :],
    coco[:, 16, :]
)

left_knee_2d = calculate_angle(
    coco[:, 11, :],
    coco[:, 13, :],
    coco[:, 15, :]
)

# ---------------------------------------------------------
# 3D knee angles
# ---------------------------------------------------------

right_knee_3d = calculate_angle(
    pose3d[:, 1, :],
    pose3d[:, 2, :],
    pose3d[:, 3, :]
)

left_knee_3d = calculate_angle(
    pose3d[:, 4, :],
    pose3d[:, 5, :],
    pose3d[:, 6, :]
)

# ---------------------------------------------------------
# Convert to approximate flexion
# ---------------------------------------------------------

right_flexion_2d = 180 - right_knee_2d
left_flexion_2d = 180 - left_knee_2d

right_flexion_3d = 180 - right_knee_3d
left_flexion_3d = 180 - left_knee_3d

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n===================================")
print("2D KNEE ANGLES")
print("===================================")

print(
    f"Right 2D angle: "
    f"min={np.nanmin(right_knee_2d):.2f} "
    f"max={np.nanmax(right_knee_2d):.2f} "
    f"mean={np.nanmean(right_knee_2d):.2f}"
)

print(
    f"Left 2D angle : "
    f"min={np.nanmin(left_knee_2d):.2f} "
    f"max={np.nanmax(left_knee_2d):.2f} "
    f"mean={np.nanmean(left_knee_2d):.2f}"
)

print("\n===================================")
print("3D KNEE ANGLES")
print("===================================")

print(
    f"Right 3D angle: "
    f"min={np.nanmin(right_knee_3d):.2f} "
    f"max={np.nanmax(right_knee_3d):.2f} "
    f"mean={np.nanmean(right_knee_3d):.2f}"
)

print(
    f"Left 3D angle : "
    f"min={np.nanmin(left_knee_3d):.2f} "
    f"max={np.nanmax(left_knee_3d):.2f} "
    f"mean={np.nanmean(left_knee_3d):.2f}"
)

# ---------------------------------------------------------
# Motion ranges
# ---------------------------------------------------------

print("\n===================================")
print("KNEE MOTION RANGE")
print("===================================")

print(
    "Right 2D flexion range:",
    round(
        np.nanmax(right_flexion_2d) -
        np.nanmin(right_flexion_2d),
        2
    ),
    "degrees"
)

print(
    "Left 2D flexion range :",
    round(
        np.nanmax(left_flexion_2d) -
        np.nanmin(left_flexion_2d),
        2
    ),
    "degrees"
)

print(
    "Right 3D flexion range:",
    round(
        np.nanmax(right_flexion_3d) -
        np.nanmin(right_flexion_3d),
        2
    ),
    "degrees"
)

print(
    "Left 3D flexion range :",
    round(
        np.nanmax(left_flexion_3d) -
        np.nanmin(left_flexion_3d),
        2
    ),
    "degrees"
)

# ---------------------------------------------------------
# Save diagnostic data
# ---------------------------------------------------------

diagnostic_df = pd.DataFrame({
    "frame": np.arange(len(coco)),

    "right_knee_2d":
        right_knee_2d,

    "left_knee_2d":
        left_knee_2d,

    "right_knee_3d":
        right_knee_3d,

    "left_knee_3d":
        left_knee_3d,

    "right_flexion_2d":
        right_flexion_2d,

    "left_flexion_2d":
        left_flexion_2d,

    "right_flexion_3d":
        right_flexion_3d,

    "left_flexion_3d":
        left_flexion_3d
})

diagnostic_path = (
    OUT_DIR +
    "/knee_2d_vs_3d_diagnostic.csv"
)

diagnostic_df.to_csv(
    diagnostic_path,
    index=False
)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 9),
    sharex=True
)

# Right
axes[0].plot(
    right_flexion_2d,
    label="Right knee 2D flexion"
)

axes[0].plot(
    right_flexion_3d,
    label="Right knee 3D flexion"
)

axes[0].set_title(
    "Right Knee — 2D vs 3D Flexion"
)

axes[0].set_ylabel(
    "Flexion (degrees)"
)

axes[0].grid(True)
axes[0].legend()

# Left
axes[1].plot(
    left_flexion_2d,
    label="Left knee 2D flexion"
)

axes[1].plot(
    left_flexion_3d,
    label="Left knee 3D flexion"
)

axes[1].set_title(
    "Left Knee — 2D vs 3D Flexion"
)

axes[1].set_xlabel("Frame")
axes[1].set_ylabel("Flexion (degrees)")

axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

plot_path = (
    OUT_DIR +
    "/knee_2d_vs_3d_diagnostic.png"
)

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(diagnostic_path)
print(plot_path)

import os
import json
import numpy as np
import pandas as pd

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

COCO_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

OUT_DIR = (
    PROJECT +
    "/data/user_video/results/video_quality"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load existing 2D pose
# ---------------------------------------------------------

pose = np.load(COCO_PATH)

print("===================================")
print("RUNNING VIDEO SUITABILITY CHECK")
print("===================================")

print("Pose shape:", pose.shape)

# COCO joints
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# ---------------------------------------------------------
# 1. Pose validity
# ---------------------------------------------------------

valid_frame = np.all(
    np.isfinite(pose),
    axis=(1, 2)
)

valid_ratio = np.mean(valid_frame)

# ---------------------------------------------------------
# 2. Compute joint trajectories
# ---------------------------------------------------------

left_ankle = pose[:, LEFT_ANKLE]
right_ankle = pose[:, RIGHT_ANKLE]

left_knee = pose[:, LEFT_KNEE]
right_knee = pose[:, RIGHT_KNEE]

left_hip = pose[:, LEFT_HIP]
right_hip = pose[:, RIGHT_HIP]

hip_center = (
    left_hip + right_hip
) / 2.0

# ---------------------------------------------------------
# 3. Movement ranges
# ---------------------------------------------------------

def motion_range(points):
    x_range = np.nanmax(points[:, 0]) - np.nanmin(points[:, 0])
    y_range = np.nanmax(points[:, 1]) - np.nanmin(points[:, 1])

    return float(
        np.sqrt(
            x_range**2 +
            y_range**2
        )
    )

right_ankle_motion = motion_range(right_ankle)
left_ankle_motion = motion_range(left_ankle)

right_knee_motion = motion_range(right_knee)
left_knee_motion = motion_range(left_knee)

hip_motion = motion_range(hip_center)

# ---------------------------------------------------------
# 4. Knee angle variation
# ---------------------------------------------------------

def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba, axis=1) *
        np.linalg.norm(bc, axis=1)
    )

    cosine = np.sum(
        ba * bc,
        axis=1
    ) / np.maximum(
        denominator,
        1e-8
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine)
    )


right_knee_angle = calculate_angle(
    right_hip,
    right_knee,
    right_ankle
)

left_knee_angle = calculate_angle(
    left_hip,
    left_knee,
    left_ankle
)

right_knee_angle_range = (
    np.nanmax(right_knee_angle)
    -
    np.nanmin(right_knee_angle)
)

left_knee_angle_range = (
    np.nanmax(left_knee_angle)
    -
    np.nanmin(left_knee_angle)
)

# ---------------------------------------------------------
# 5. Alternating leg movement
# ---------------------------------------------------------

right_ankle_y_range = (
    np.nanmax(right_ankle[:, 1])
    -
    np.nanmin(right_ankle[:, 1])
)

left_ankle_y_range = (
    np.nanmax(left_ankle[:, 1])
    -
    np.nanmin(left_ankle[:, 1])
)

# ---------------------------------------------------------
# 6. Suitability rules
#
# These are NOT biomechanical correctness thresholds.
# They are only input-quality thresholds.
# ---------------------------------------------------------

checks = {}

checks["pose_validity"] = (
    valid_ratio >= 0.95
)

checks["ankle_motion"] = (
    max(
        right_ankle_motion,
        left_ankle_motion
    ) >= 0.05
)

checks["knee_motion"] = (
    max(
        right_knee_angle_range,
        left_knee_angle_range
    ) >= 10.0
)

checks["vertical_leg_motion"] = (
    max(
        right_ankle_y_range,
        left_ankle_y_range
    ) >= 0.05
)

# ---------------------------------------------------------
# Overall decision
# ---------------------------------------------------------

video_accepted = all(
    checks.values()
)

# ---------------------------------------------------------
# Print report
# ---------------------------------------------------------

print("\n-----------------------------------")
print("QUALITY MEASUREMENTS")
print("-----------------------------------")

print(
    f"Valid frame ratio      : "
    f"{valid_ratio:.3f}"
)

print(
    f"Right ankle motion     : "
    f"{right_ankle_motion:.4f}"
)

print(
    f"Left ankle motion      : "
    f"{left_ankle_motion:.4f}"
)

print(
    f"Right knee angle range : "
    f"{right_knee_angle_range:.2f}°"
)

print(
    f"Left knee angle range  : "
    f"{left_knee_angle_range:.2f}°"
)

print(
    f"Right ankle Y range    : "
    f"{right_ankle_y_range:.4f}"
)

print(
    f"Left ankle Y range     : "
    f"{left_ankle_y_range:.4f}"
)

print("\n-----------------------------------")
print("QUALITY CHECKS")
print("-----------------------------------")

for name, result in checks.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{name:25s}: {status}"
    )

print("\n===================================")

if video_accepted:

    print("VIDEO ACCEPTED")
    print(
        "The pose contains sufficient "
        "movement for running analysis."
    )

else:

    print("VIDEO REJECTED")

    print(
        "\nPlease upload a clear running video "
        "with the following requirements:"
    )

    print(
        "1. Person must be running throughout "
        "most of the clip."
    )

    print(
        "2. Prefer a side-view camera."
    )

    print(
        "3. Full body and both legs should "
        "remain visible."
    )

    print(
        "4. Avoid sitting, standing, walking, "
        "or large occlusions."
    )

    print(
        "5. Prefer a 5–10 second continuous clip."
    )

print("===================================")

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

report = {
    "video_accepted": bool(video_accepted),

    "valid_frame_ratio": float(valid_ratio),

    "right_ankle_motion": float(right_ankle_motion),
    "left_ankle_motion": float(left_ankle_motion),

    "right_knee_angle_range": float(
        right_knee_angle_range
    ),

    "left_knee_angle_range": float(
        left_knee_angle_range
    ),

    "right_ankle_y_range": float(
        right_ankle_y_range
    ),

    "left_ankle_y_range": float(
        left_ankle_y_range
    ),

    "checks": {
        k: bool(v)
        for k, v in checks.items()
    }
}

report_path = (
    OUT_DIR +
    "/running_video_suitability.json"
)

with open(
    report_path,
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

print("\nSaved:")
print(report_path)

from google.colab import files
import os
import shutil

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

VIDEO_DIR = PROJECT + "/data/user_video"
os.makedirs(VIDEO_DIR, exist_ok=True)

print("Upload your clear running video...")
uploaded = files.upload()

if not uploaded:
    raise RuntimeError("No video uploaded.")

# Take the uploaded file
uploaded_name = list(uploaded.keys())[0]
uploaded_path = "/content/" + uploaded_name

# Save as the new active input video
new_video_path = VIDEO_DIR + "/running_input.mp4"

shutil.copy2(
    uploaded_path,
    new_video_path
)

print("\n===================================")
print("NEW VIDEO SAVED")
print("===================================")
print("Original filename :", uploaded_name)
print("Saved as          :", new_video_path)
print("Size (MB)         :", round(os.path.getsize(new_video_path) / (1024**2), 2))
print("===================================")

import cv2
import os
import json

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

VIDEO_PATH = (
    PROJECT +
    "/data/user_video/running_input.mp4"
)

FRAME_DIR = (
    PROJECT +
    "/data/user_video/running_input_frames"
)

os.makedirs(FRAME_DIR, exist_ok=True)

# ---------------------------------------------------------
# Check whether frames already exist
# ---------------------------------------------------------

existing_frames = [
    f for f in os.listdir(FRAME_DIR)
    if f.endswith(".jpg")
]

if len(existing_frames) > 0:

    print("Frames already exist.")
    print("Frame count:", len(existing_frames))
    print("Skipping extraction.")

else:

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print("===================================")
    print("VIDEO INFORMATION")
    print("===================================")
    print("FPS          :", fps)
    print("Total frames :", total_frames)
    print("Resolution   :", width, "x", height)

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(
            FRAME_DIR,
            f"frame_{frame_id:05d}.jpg"
        )

        cv2.imwrite(
            frame_path,
            frame
        )

        frame_id += 1

    cap.release()

    print("\n===================================")
    print("FRAME EXTRACTION COMPLETE")
    print("===================================")
    print("Frames extracted:", frame_id)
    print("Saved to:", FRAME_DIR)

    # Save metadata
    metadata = {
        "video": VIDEO_PATH,
        "fps": float(fps),
        "total_frames": int(total_frames),
        "width": int(width),
        "height": int(height),
        "extracted_frames": int(frame_id)
    }

    metadata_path = os.path.join(
        PROJECT,
        "data/user_video/",
        "running_input_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("Metadata:", metadata_path)

import os
import numpy as np
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

FRAME_DIR = (
    PROJECT +
    "/data/user_video/running_input_frames"
)

OUT_DIR = (
    PROJECT +
    "/data/user_video/poses_2d"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = (
    OUT_DIR +
    "/running_input_coco.npy"
)

MODEL_PATH = (
    "/content/mediapipe_models/"
    "pose_landmarker_full.task"
)

# ---------------------------------------------------------
# Skip if already processed
# ---------------------------------------------------------

if os.path.exists(OUTPUT_PATH):

    coco_pose = np.load(
        OUTPUT_PATH
    )

    print("2D pose already exists.")
    print("Shape:", coco_pose.shape)
    print("Skipping MediaPipe extraction.")

else:

    # -----------------------------------------------------
    # MediaPipe configuration
    # -----------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    # MediaPipe → COCO-17
    mp_to_coco = [
        0,   # nose
        2,   # left_eye
        5,   # right_eye
        7,   # left_ear
        8,   # right_ear
        11,  # left_shoulder
        12,  # right_shoulder
        13,  # left_elbow
        14,  # right_elbow
        15,  # left_wrist
        16,  # right_wrist
        23,  # left_hip
        24,  # right_hip
        25,  # left_knee
        26,  # right_knee
        27,  # left_ankle
        28   # right_ankle
    ]

    frame_files = sorted([
        f for f in os.listdir(FRAME_DIR)
        if f.endswith(".jpg")
    ])

    coco_pose = np.full(
        (len(frame_files), 17, 2),
        np.nan,
        dtype=np.float32
    )

    detected = 0

    # -----------------------------------------------------
    # Run MediaPipe
    # -----------------------------------------------------

    with vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:

        for i, filename in enumerate(frame_files):

            frame_path = os.path.join(
                FRAME_DIR,
                filename
            )

            image = mp.Image.create_from_file(
                frame_path
            )

            result = landmarker.detect(
                image
            )

            if (
                result.pose_landmarks
                and len(result.pose_landmarks) > 0
            ):

                landmarks = (
                    result.pose_landmarks[0]
                )

                for j, mp_idx in enumerate(
                    mp_to_coco
                ):

                    coco_pose[i, j, 0] = (
                        landmarks[mp_idx].x
                    )

                    coco_pose[i, j, 1] = (
                        landmarks[mp_idx].y
                    )

                detected += 1

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    np.save(
        OUTPUT_PATH,
        coco_pose
    )

    print("===================================")
    print("MEDIAPIPE 2D EXTRACTION COMPLETE")
    print("===================================")
    print("Frames          :", len(frame_files))
    print("Pose detections :", detected)
    print("Detection ratio :", round(
        detected / len(frame_files),
        3
    ))
    print("Shape           :", coco_pose.shape)
    print("NaN values      :", np.isnan(coco_pose).sum())
    print("Saved           :", OUTPUT_PATH)

import os
import json
import numpy as np

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

COCO_PATH = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "running_input_coco.npy"
)

OUT_DIR = (
    PROJECT +
    "/data/user_video/results/video_quality"
)

os.makedirs(OUT_DIR, exist_ok=True)

pose = np.load(COCO_PATH)

print("===================================")
print("RUNNING VIDEO SUITABILITY CHECK")
print("===================================")
print("Pose shape:", pose.shape)

# COCO-17 indices
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# ---------------------------------------------------------
# Joint trajectories
# ---------------------------------------------------------

left_ankle = pose[:, LEFT_ANKLE]
right_ankle = pose[:, RIGHT_ANKLE]

left_knee = pose[:, LEFT_KNEE]
right_knee = pose[:, RIGHT_KNEE]

left_hip = pose[:, LEFT_HIP]
right_hip = pose[:, RIGHT_HIP]

hip_center = (
    left_hip + right_hip
) / 2.0

# ---------------------------------------------------------
# Motion range
# ---------------------------------------------------------

def motion_range(points):

    x_range = (
        np.nanmax(points[:, 0])
        -
        np.nanmin(points[:, 0])
    )

    y_range = (
        np.nanmax(points[:, 1])
        -
        np.nanmin(points[:, 1])
    )

    return float(
        np.sqrt(
            x_range**2 +
            y_range**2
        )
    )

right_ankle_motion = motion_range(right_ankle)
left_ankle_motion = motion_range(left_ankle)

right_knee_motion = motion_range(right_knee)
left_knee_motion = motion_range(left_knee)

hip_motion = motion_range(hip_center)

# ---------------------------------------------------------
# Knee angle
# ---------------------------------------------------------

def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba, axis=1)
        *
        np.linalg.norm(bc, axis=1)
    )

    cosine = np.sum(
        ba * bc,
        axis=1
    ) / np.maximum(
        denominator,
        1e-8
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine)
    )

right_knee_angle = calculate_angle(
    right_hip,
    right_knee,
    right_ankle
)

left_knee_angle = calculate_angle(
    left_hip,
    left_knee,
    left_ankle
)

right_knee_angle_range = (
    np.nanmax(right_knee_angle)
    -
    np.nanmin(right_knee_angle)
)

left_knee_angle_range = (
    np.nanmax(left_knee_angle)
    -
    np.nanmin(left_knee_angle)
)

# ---------------------------------------------------------
# Vertical ankle movement
# ---------------------------------------------------------

right_ankle_y_range = (
    np.nanmax(right_ankle[:, 1])
    -
    np.nanmin(right_ankle[:, 1])
)

left_ankle_y_range = (
    np.nanmax(left_ankle[:, 1])
    -
    np.nanmin(left_ankle[:, 1])
)

# ---------------------------------------------------------
# Checks
# ---------------------------------------------------------

valid_ratio = np.mean(
    np.all(
        np.isfinite(pose),
        axis=(1, 2)
    )
)

checks = {

    "pose_validity":
        valid_ratio >= 0.95,

    "ankle_motion":
        max(
            right_ankle_motion,
            left_ankle_motion
        ) >= 0.05,

    "knee_motion":
        max(
            right_knee_angle_range,
            left_knee_angle_range
        ) >= 10.0,

    "vertical_leg_motion":
        max(
            right_ankle_y_range,
            left_ankle_y_range
        ) >= 0.05
}

video_accepted = all(
    checks.values()
)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("\n-----------------------------------")
print("QUALITY MEASUREMENTS")
print("-----------------------------------")

print(
    f"Valid frame ratio      : {valid_ratio:.3f}"
)

print(
    f"Right ankle motion     : {right_ankle_motion:.4f}"
)

print(
    f"Left ankle motion      : {left_ankle_motion:.4f}"
)

print(
    f"Right knee angle range : "
    f"{right_knee_angle_range:.2f}°"
)

print(
    f"Left knee angle range  : "
    f"{left_knee_angle_range:.2f}°"
)

print(
    f"Right ankle Y range    : "
    f"{right_ankle_y_range:.4f}"
)

print(
    f"Left ankle Y range     : "
    f"{left_ankle_y_range:.4f}"
)

print("\n-----------------------------------")
print("QUALITY CHECKS")
print("-----------------------------------")

for name, result in checks.items():

    print(
        f"{name:25s}: "
        f"{'PASS' if result else 'FAIL'}"
    )

print("\n===================================")

if video_accepted:

    print("VIDEO ACCEPTED")

else:

    print("VIDEO REJECTED")

    print("\nRequired:")
    print("1. Clear running motion")
    print("2. Prefer side-view camera")
    print("3. Full body visible")
    print("4. Both legs visible")
    print("5. Minimal occlusion")
    print("6. Continuous running")

print("===================================")

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

report = {
    "video_accepted": bool(video_accepted),
    "valid_frame_ratio": float(valid_ratio),
    "right_ankle_motion": float(right_ankle_motion),
    "left_ankle_motion": float(left_ankle_motion),
    "right_knee_angle_range": float(
        right_knee_angle_range
    ),
    "left_knee_angle_range": float(
        left_knee_angle_range
    ),
    "right_ankle_y_range": float(
        right_ankle_y_range
    ),
    "left_ankle_y_range": float(
        left_ankle_y_range
    ),
    "checks": {
        k: bool(v)
        for k, v in checks.items()
    }
}

report_path = (
    OUT_DIR +
    "/running_input_suitability.json"
)

with open(
    report_path,
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

print("\nSaved:")
print(report_path)

import os
import json
import hashlib
import numpy as np

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

print("==============================================")
print("ATHLETEPOSE3D PROJECT — FINAL SAVE CHECK")
print("==============================================")

# ---------------------------------------------------------
# Important project locations
# ---------------------------------------------------------

important_paths = {

    # Official code
    "athletepose3d_code":
        PROJECT + "/code/AthletePose3D",

    "tcpformer_code":
        PROJECT + "/code/TCPFormer",

    # Model
    "tcpformer_checkpoint":
        PROJECT +
        "/models/pretrained/"
        "TCPFormer_ap3d_81.pth.tr",

    # Reference 3D pose
    "reference_3d":
        PROJECT +
        "/data/poses_3d/"
        "Running_0_cam_1_TCPFormer_3D.npy",

    # Reference features
    "reference_tcpformer_features":
        PROJECT +
        "/data/features/"
        "Running_0_cam_1_TCPFormer_features.csv",

    "reference_groundtruth_features":
        PROJECT +
        "/data/features/"
        "Running_0_cam_1_GroundTruth_features.csv",

    "reference_tcpformer_normalized":
        PROJECT +
        "/data/features/"
        "Running_0_cam_1_TCPFormer_features_normalized.csv",

    "reference_groundtruth_normalized":
        PROJECT +
        "/data/features/"
        "Running_0_cam_1_GroundTruth_features_normalized.csv",

    # RBDS
    "rbds_reference":
        PROJECT +
        "/data/references/RBDS/"
        "reference_curves/"
        "RBDS_running_reference_curves.csv",

    # Current user video
    "user_video":
        PROJECT +
        "/data/user_video/"
        "running_input.mp4",

    # Current user 2D pose
    "user_2d_pose":
        PROJECT +
        "/data/user_video/poses_2d/"
        "running_input_coco.npy",

    # Current original user video pose
    "old_user_2d_pose":
        PROJECT +
        "/data/user_video/poses_2d/"
        "human_running_video_coco.npy",

    # Current full 3D pose from old video
    "old_user_3d_pose":
        PROJECT +
        "/data/user_video/poses_3d/"
        "human_running_video_TCPFormer_3D_full.npy",

    # Current user features
    "old_user_features":
        PROJECT +
        "/data/user_video/features/"
        "human_running_video_features_full.csv",

    # Gait cycles
    "stable_gait_cycles":
        PROJECT +
        "/data/user_video/results/"
        "gait_cycles/stable_gait_cycles.csv",

    "normalized_gait_pose":
        PROJECT +
        "/data/user_video/results/"
        "gait_cycles/normalized_101/"
        "user_gait_cycles_101_pose.npy",

    "normalized_gait_features":
        PROJECT +
        "/data/user_video/results/"
        "gait_cycles/normalized_101/"
        "user_gait_cycles_101_features.npy",

    "normalized_gait_metadata":
        PROJECT +
        "/data/user_video/results/"
        "gait_cycles/normalized_101/"
        "user_gait_cycles_101_metadata.csv",

    # Comparison
    "knee_comparison":
        PROJECT +
        "/data/user_video/results/"
        "reference_comparison/"
        "user_vs_RBDS_knee_flexion.csv",

    "knee_comparison_plot":
        PROJECT +
        "/data/user_video/results/"
        "reference_comparison/"
        "user_vs_RBDS_knee_flexion.png",

    # Quality
    "quality_report":
        PROJECT +
        "/data/user_video/results/"
        "video_quality/"
        "running_input_suitability.json"
}

# ---------------------------------------------------------
# Check files
# ---------------------------------------------------------

existing = {}
missing = {}

for name, path in important_paths.items():

    if os.path.exists(path):

        if os.path.isfile(path):

            size_mb = (
                os.path.getsize(path)
                / (1024 * 1024)
            )

            existing[name] = {
                "path": path,
                "type": "file",
                "size_MB": round(size_mb, 3)
            }

        else:

            existing[name] = {
                "path": path,
                "type": "directory"
            }

    else:

        missing[name] = path

print("\n==============================================")
print("EXISTING IMPORTANT FILES")
print("==============================================")

for name, info in existing.items():

    print(
        f"[OK] {name}"
    )

print("\n==============================================")
print("MISSING FILES")
print("==============================================")

if missing:

    for name, path in missing.items():

        print(
            f"[MISSING] {name}"
        )
        print(
            "          ",
            path
        )

else:

    print("None")

# ---------------------------------------------------------
# Dataset / model shapes
# ---------------------------------------------------------

shapes = {}

shape_files = {

    "reference_3d":
        important_paths["reference_3d"],

    "user_2d_pose":
        important_paths["user_2d_pose"],

    "old_user_3d_pose":
        important_paths["old_user_3d_pose"],

    "normalized_gait_pose":
        important_paths["normalized_gait_pose"],

    "normalized_gait_features":
        important_paths["normalized_gait_features"]
}

for name, path in shape_files.items():

    if os.path.exists(path):

        try:

            arr = np.load(
                path,
                mmap_mode="r"
            )

            shapes[name] = list(
                arr.shape
            )

        except Exception as e:

            shapes[name] = (
                "Could not read shape"
            )

# ---------------------------------------------------------
# Save project state
# ---------------------------------------------------------

state = {

    "project": "AthletePose3D_Project",

    "project_root": PROJECT,

    "status": {
        "dataset_setup": True,
        "official_mapping_verified": True,
        "tcpformer_setup": True,
        "tcpformer_checkpoint_saved": True,
        "reference_processing_completed": True,
        "rbds_reference_created": True,
        "user_video_pipeline_created": True,
        "gait_cycle_pipeline_created": True,
        "reference_comparison_created": True,
        "video_quality_gate_created": True,
        "final_coaching_layer_completed": False,
        "final_web_app_completed": False
    },

    "important_paths": important_paths,

    "existing_files": existing,

    "missing_files": missing,

    "array_shapes": shapes
}

STATE_PATH = (
    PROJECT +
    "/PROJECT_STATE.json"
)

with open(
    STATE_PATH,
    "w"
) as f:

    json.dump(
        state,
        f,
        indent=4
    )

# ---------------------------------------------------------
# Create a simple README
# ---------------------------------------------------------

README_PATH = (
    PROJECT +
    "/PROJECT_README.txt"
)

readme = """
ATHLETEPOSE3D PROJECT
=====================

This folder contains the persistent project state.

MAIN PIPELINE
-------------
Video
 -> MediaPipe 2D pose
 -> COCO-17
 -> COCO to H36M
 -> TCPFormer
 -> 3D pose
 -> gait-cycle detection
 -> 101-point normalization
 -> biomechanics/features
 -> RBDS reference comparison
 -> technique analysis
 -> LLM coaching

IMPORTANT
---------
Do NOT rerun expensive model inference if the corresponding
output already exists in Google Drive.

CURRENT STATUS
--------------
Dataset and reference setup: COMPLETE
TCPFormer setup: COMPLETE
RBDS reference: COMPLETE
User-video processing pipeline: COMPLETE
Gait-cycle pipeline: COMPLETE
Video-quality gate: COMPLETE

Final technique-deviation report:
NOT YET COMPLETE

Final LLM coaching:
NOT YET COMPLETE

Final web application:
NOT YET COMPLETE

CURRENT USER VIDEO
------------------
running_input.mp4

The current sample video was rejected by the running-video
suitability gate. A clearer running video can be substituted
later without rebuilding the reference/model pipeline.

PERSISTENT ROOT
---------------
/content/drive/MyDrive/AthletePose3D_Project
"""

with open(
    README_PATH,
    "w"
) as f:

    f.write(
        readme.strip()
    )

print("\n==============================================")
print("PROJECT STATE SAVED")
print("==============================================")
print("State file :")
print(STATE_PATH)

print("\nREADME:")
print(README_PATH)

print("\nIMPORTANT:")
print(
    "All important outputs are stored in Google Drive."
)

print(
    "After a Colab runtime reset, mount Drive and "
    "check PROJECT_STATE.json before rerunning anything."
)

print("\n==============================================")
print("SAVE COMPLETE")
print("==============================================")

import os
import json
import numpy as np

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

print("=" * 60)
print("ATHLETEPOSE3D PROJECT HEALTH CHECK")
print("=" * 60)

# ---------------------------------------------------------
# Important saved outputs
# ---------------------------------------------------------

files_to_check = {

    "TCPFormer checkpoint":
        PROJECT +
        "/models/pretrained/"
        "TCPFormer_ap3d_81.pth.tr",

    "AthletePose3D code":
        PROJECT +
        "/code/AthletePose3D",

    "TCPFormer code":
        PROJECT +
        "/code/TCPFormer",

    "RBDS reference":
        PROJECT +
        "/data/references/RBDS/reference_curves/"
        "RBDS_running_reference_curves.csv",

    "User video":
        PROJECT +
        "/data/user_video/running_input.mp4",

    "User 2D pose":
        PROJECT +
        "/data/user_video/poses_2d/"
        "running_input_coco.npy",

    "Project state":
        PROJECT +
        "/PROJECT_STATE.json"
}

# ---------------------------------------------------------
# Check existence
# ---------------------------------------------------------

print("\nFILE STATUS")
print("-" * 60)

for name, path in files_to_check.items():

    exists = os.path.exists(path)

    if exists:

        if os.path.isfile(path):

            size_mb = (
                os.path.getsize(path)
                / (1024 * 1024)
            )

            print(
                f"[OK]      {name:<25} "
                f"{size_mb:.2f} MB"
            )

        else:

            print(
                f"[OK]      {name:<25} DIRECTORY"
            )

    else:

        print(
            f"[MISSING] {name}"
        )

# ---------------------------------------------------------
# Check saved arrays
# ---------------------------------------------------------

print("\nARRAY SHAPES")
print("-" * 60)

arrays = {

    "User 2D":
        PROJECT +
        "/data/user_video/poses_2d/"
        "running_input_coco.npy",

    "Old user 3D":
        PROJECT +
        "/data/user_video/poses_3d/"
        "human_running_video_TCPFormer_3D_full.npy",

    "Normalized gait pose":
        PROJECT +
        "/data/user_video/results/"
        "gait_cycles/normalized_101/"
        "user_gait_cycles_101_pose.npy",

    "Normalized gait features":
        PROJECT +
        "/data/user_video/results/"
        "gait_cycles/normalized_101/"
        "user_gait_cycles_101_features.npy",

    "Reference 3D":
        PROJECT +
        "/data/poses_3d/"
        "Running_0_cam_1_TCPFormer_3D.npy"
}

for name, path in arrays.items():

    if os.path.exists(path):

        try:

            arr = np.load(
                path,
                mmap_mode="r"
            )

            print(
                f"[OK] {name:<28} "
                f"{arr.shape}"
            )

        except Exception as e:

            print(
                f"[ERROR] {name}: {e}"
            )

    else:

        print(
            f"[MISSING] {name}"
        )

# ---------------------------------------------------------
# Project state
# ---------------------------------------------------------

state_path = PROJECT + "/PROJECT_STATE.json"

print("\nPROJECT STATE")
print("-" * 60)

if os.path.exists(state_path):

    with open(state_path, "r") as f:
        state = json.load(f)

    for key, value in state["status"].items():

        print(
            f"{key:<35}: "
            f"{'DONE' if value else 'NOT DONE'}"
        )

else:

    print("PROJECT_STATE.json not found.")

print("\n" + "=" * 60)
print("HEALTH CHECK COMPLETE")
print("=" * 60)

import numpy as np

pose2d = np.load(
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_2d/running_input_coco.npy"
)

print("2D Pose Shape:", pose2d.shape)
print("NaN values:", np.isnan(pose2d).sum())

import numpy as np

pose3d = np.load(
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

print("3D Pose Shape:", pose3d.shape)
print("NaN:", np.isnan(pose3d).sum())
print("Inf:", np.isinf(pose3d).sum())

import pandas as pd

cycles = pd.read_csv(
    "/content/drive/MyDrive/AthletePose3D_Project/"
    "data/user_video/results/gait_cycles/"
    "stable_gait_cycles.csv"
)

print(cycles)
print("\nNumber of stable cycles:", len(cycles))

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

print("=" * 60)
print("AI SPORTS TECHNIQUE COACH — LIVE DEMONSTRATION")
print("=" * 60)

# =========================================================
# 1. INPUT VIDEO
# =========================================================

video_path = (
    PROJECT +
    "/data/user_video/human_running_video.mp4"
)

print("\n[1] INPUT VIDEO")
print("-" * 60)

if os.path.exists(video_path):
    size_mb = os.path.getsize(video_path) / (1024**2)
    print("Video:", os.path.basename(video_path))
    print("Size :", round(size_mb, 2), "MB")
else:
    print("Video not found.")

# =========================================================
# 2. 2D POSE
# =========================================================

pose2d_path = (
    PROJECT +
    "/data/user_video/poses_2d/"
    "human_running_video_coco.npy"
)

pose2d = np.load(pose2d_path)

print("\n[2] 2D POSE EXTRACTION")
print("-" * 60)
print("Model      : MediaPipe Pose")
print("Format     : COCO-17")
print("Shape      :", pose2d.shape)
print("NaN values :", np.isnan(pose2d).sum())

# =========================================================
# 3. 3D POSE
# =========================================================

pose3d_path = (
    PROJECT +
    "/data/user_video/poses_3d/"
    "human_running_video_TCPFormer_3D_full.npy"
)

pose3d = np.load(pose3d_path)

print("\n[3] 3D POSE RECONSTRUCTION")
print("-" * 60)
print("Model      : TCPFormer")
print("Shape      :", pose3d.shape)
print("NaN values :", np.isnan(pose3d).sum())
print("Inf values :", np.isinf(pose3d).sum())

# =========================================================
# 4. GAIT CYCLES
# =========================================================

cycles_path = (
    PROJECT +
    "/data/user_video/results/gait_cycles/"
    "stable_gait_cycles.csv"
)

cycles = pd.read_csv(cycles_path)

print("\n[4] GAIT CYCLE DETECTION")
print("-" * 60)
print("Stable gait cycles :", len(cycles))
print("Right cycles       :", sum(cycles["foot"] == "right"))
print("Left cycles        :", sum(cycles["foot"] == "left"))

# =========================================================
# 5. NORMALIZATION
# =========================================================

normalized_pose_path = (
    PROJECT +
    "/data/user_video/results/gait_cycles/"
    "normalized_101/"
    "user_gait_cycles_101_pose.npy"
)

normalized_pose = np.load(
    normalized_pose_path
)

print("\n[5] GAIT CYCLE NORMALIZATION")
print("-" * 60)
print("Normalized pose shape :", normalized_pose.shape)
print("Points per cycle      :", normalized_pose.shape[1])

# =========================================================
# 6. RBDS REFERENCE
# =========================================================

rbds_path = (
    PROJECT +
    "/data/references/RBDS/reference_curves/"
    "RBDS_running_reference_curves.csv"
)

rbds = pd.read_csv(rbds_path)

print("\n[6] RBDS RUNNING REFERENCE")
print("-" * 60)
print("Reference shape :", rbds.shape)
print("Speeds          :", sorted(rbds["speed"].unique()))
print("Joints          :", sorted(rbds["joint"].unique()))
print("Axes            :", sorted(rbds["axis"].unique()))

# =========================================================
# 7. FEATURE EXTRACTION
# =========================================================

feature_path = (
    PROJECT +
    "/data/user_video/features/"
    "human_running_video_features_full.csv"
)

features = pd.read_csv(feature_path)

print("\n[7] BIOMECHANICAL FEATURES")
print("-" * 60)
print("Feature matrix :", features.shape)
print("Number of features :", len(features.columns) - 1)

print("\nFeatures:")
print(
    ", ".join(
        features.columns[1:10]
    )
    + " ..."
)

# =========================================================
# 8. REFERENCE COMPARISON
# =========================================================

comparison_path = (
    PROJECT +
    "/data/user_video/results/reference_comparison/"
    "user_vs_RBDS_knee_flexion.csv"
)

comparison = pd.read_csv(
    comparison_path
)

print("\n[8] REFERENCE COMPARISON")
print("-" * 60)
print("Comparison shape :", comparison.shape)

print(
    "\nThis stage compares the user's normalized "
    "movement against the RBDS reference."
)

print(
    "Current comparison is experimental and "
    "not treated as a final technique score."
)

# =========================================================
# 9. SHOW 3D POSE
# =========================================================

print("\n[9] 3D POSE VISUALIZATION")
print("-" * 60)

connections = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16)
]

frame_idx = 120

p = pose3d[frame_idx]

fig = plt.figure(
    figsize=(7,7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

for a,b in connections:

    ax.plot(
        [p[a,0], p[b,0]],
        [p[a,1], p[b,1]],
        [p[a,2], p[b,2]]
    )

ax.scatter(
    p[:,0],
    p[:,1],
    p[:,2],
    s=30
)

ax.set_title(
    f"TCPFormer 3D Pose — Frame {frame_idx}"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

# =========================================================
# 10. SHOW GAIT NORMALIZATION
# =========================================================

print("\n[10] NORMALIZED KNEE MOVEMENT")
print("-" * 60)

# Feature indices from our normalized feature matrix
# right_knee_angle = column 0
# left_knee_angle  = column 1

right_knee = normalized_pose[:, :, 2, :]  # placeholder

import os
import json

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

LLM_DIR = PROJECT + "/data/user_video/results/llm"
os.makedirs(LLM_DIR, exist_ok=True)

example_findings = {
    "analysis_type": "running_technique",
    "reference": {
        "dataset": "RBDS",
        "description": "Healthy running biomechanics reference distribution",
        "speeds_mps": [2.5, 3.5, 4.5]
    },

    "video_quality": {
        "status": "PASS",
        "confidence": "high"
    },

    "findings": [

        {
            "feature": "knee_flexion",
            "side": "right",

            "gait_phase": {
                "start_percent": 20,
                "end_percent": 40
            },

            "user_value_deg": 82.0,

            "reference": {
                "mean_deg": 96.0,
                "p05_deg": 86.0,
                "p95_deg": 106.0
            },

            "deviation_deg": -14.0,

            "outside_reference": True,

            "confidence": 0.82
        },

        {
            "feature": "knee_flexion",
            "side": "left",

            "gait_phase": {
                "start_percent": 20,
                "end_percent": 40
            },

            "user_value_deg": 91.0,

            "reference": {
                "mean_deg": 96.0,
                "p05_deg": 86.0,
                "p95_deg": 106.0
            },

            "deviation_deg": -5.0,

            "outside_reference": False,

            "confidence": 0.78
        }
    ]
}

output_path = (
    LLM_DIR +
    "/example_structured_findings.json"
)

with open(output_path, "w") as f:
    json.dump(
        example_findings,
        f,
        indent=4
    )

print("===================================")
print("LLM INPUT CREATED")
print("===================================")
print("Saved:")
print(output_path)

!pip -q install -U transformers accelerate bitsandbytes

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("Loading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("===================================")
print("QWEN LOADED")
print("===================================")
print("Model :", MODEL_NAME)
print("Device:", next(model.parameters()).device)

import json
import torch

# ---------------------------------------------------------
# Load the example structured findings
# ---------------------------------------------------------

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

findings_path = (
    PROJECT +
    "/data/user_video/results/llm/"
    "example_structured_findings.json"
)

with open(findings_path, "r") as f:
    findings = json.load(f)

# ---------------------------------------------------------
# System instruction
# ---------------------------------------------------------

system_prompt = """
You are an AI running biomechanics coaching assistant.

You receive structured measurements produced by a
computer-vision biomechanics pipeline.

Your job is ONLY to explain the supplied measurements.

Rules:
- Do not invent measurements.
- Do not invent gait phases.
- Do not diagnose injuries or medical conditions.
- Do not claim that a movement is universally correct or incorrect.
- Do not say the athlete is a bad or good runner.
- Do not make claims unsupported by the supplied data.
- Clearly distinguish observation from interpretation.
- Use phrases such as "possible deviation",
  "differs from the reference", and "may indicate".
- Give practical, simple coaching suggestions.
- Keep the report concise.

For every meaningful finding, provide:

1. Observation
2. Reference comparison
3. Possible interpretation
4. Coaching focus

End with a short overall summary.
"""

# ---------------------------------------------------------
# User input
# ---------------------------------------------------------

user_prompt = f"""
Analyze the following running biomechanics findings.

REFERENCE:
{json.dumps(findings["reference"], indent=2)}

VIDEO QUALITY:
{json.dumps(findings["video_quality"], indent=2)}

FINDINGS:
{json.dumps(findings["findings"], indent=2)}

Generate a concise coaching report.
"""

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]

# ---------------------------------------------------------
# Convert to Qwen chat format
# ---------------------------------------------------------

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}

# ---------------------------------------------------------
# Generate
# ---------------------------------------------------------

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False
    )

# Remove prompt tokens
generated_tokens = outputs[
    0,
    inputs["input_ids"].shape[1]:
]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("=" * 60)
print("QWEN RUNNING COACH")
print("=" * 60)
print()
print(response)

import os
import json
import torch

PROJECT = "/content/drive/MyDrive/AthletePose3D_Project"

LLM_DIR = (
    PROJECT +
    "/data/user_video/results/llm"
)

os.makedirs(LLM_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load structured findings
# ---------------------------------------------------------

findings_path = (
    LLM_DIR +
    "/example_structured_findings.json"
)

with open(findings_path, "r") as f:
    findings = json.load(f)

# ---------------------------------------------------------
# Strict coaching prompt
# ---------------------------------------------------------

system_prompt = """
You are an AI running biomechanics feedback assistant.

You receive structured measurements generated by a
computer-vision biomechanics pipeline.

Your job is to explain ONLY the information supplied.

STRICT RULES:

1. Do not invent measurements.
2. Do not invent gait phases.
3. Do not diagnose injuries or medical conditions.
4. Do not claim the athlete is a good or bad runner.
5. Do not say a movement is universally correct or incorrect.
6. Do not recommend specific exercises unless the input
   explicitly provides evidence supporting that exercise.
7. Do not claim that strength, stability, alignment, or
   injury risk has been measured unless those measurements
   are explicitly provided.
8. Clearly separate observation from interpretation.
9. Use cautious language:
   "possible deviation",
   "differs from the reference",
   "may indicate",
   "consider focusing on".
10. Keep the feedback concise and understandable.

For each finding provide:

- Observation
- Reference comparison
- Possible interpretation
- Coaching focus
- Confidence

The coaching focus must be directly related to the
measured movement and must not introduce unsupported
medical or physiological claims.

Finish with a short overall summary.
"""

user_prompt = f"""
Analyze these structured running biomechanics findings.

REFERENCE:
{json.dumps(findings["reference"], indent=2)}

VIDEO QUALITY:
{json.dumps(findings["video_quality"], indent=2)}

FINDINGS:
{json.dumps(findings["findings"], indent=2)}

Generate the final coaching report.
"""

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]

# ---------------------------------------------------------
# Qwen chat template
# ---------------------------------------------------------

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}

# ---------------------------------------------------------
# Generate
# ---------------------------------------------------------

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False
    )

generated_tokens = outputs[
    0,
    inputs["input_ids"].shape[1]:
]

report = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

txt_path = (
    LLM_DIR +
    "/running_coaching_report.txt"
)

json_path = (
    LLM_DIR +
    "/running_coaching_report.json"
)

with open(txt_path, "w") as f:
    f.write(report)

report_data = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "input_file": findings_path,
    "report": report
}

with open(json_path, "w") as f:
    json.dump(
        report_data,
        f,
        indent=4
    )

# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("=" * 60)
print("FINAL QWEN COACHING REPORT")
print("=" * 60)
print()
print(report)

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)
print(txt_path)
print(json_path)