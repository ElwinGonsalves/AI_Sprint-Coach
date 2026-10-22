import os
import gdown
import urllib.request
import subprocess
import zipfile

def setup_environment():
    # 1. Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

    # 2. Download MediaPipe Pose Landmarker
    mp_task_path = "models/pose_landmarker_full.task"
    if not os.path.exists(mp_task_path):
        print("Downloading MediaPipe Pose Landmarker...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, mp_task_path)
        print("MediaPipe model downloaded.")
    else:
        print("MediaPipe model already exists.")

    # 3. Download TCPFormer Checkpoint
    tcp_ckpt_path = "models/TCPFormer_ap3d_81.pth.tr"
    if not os.path.exists(tcp_ckpt_path):
        print("Downloading TCPFormer Checkpoint...")
        gdown.download(id="1H4tq_oHQkGbYqZxS_5SgOqNicRYA1zsI", output=tcp_ckpt_path, quiet=False)
        print("TCPFormer checkpoint downloaded.")
    else:
        print("TCPFormer checkpoint already exists.")

    # 4. Clone TCPFormer repository
    if not os.path.exists("TCPFormer"):
        print("Cloning TCPFormer repository...")
        subprocess.run(["git", "clone", "https://github.com/AsukaCamellia/TCPFormer.git"], check=True)
        print("TCPFormer cloned.")
    else:
        print("TCPFormer repository already exists.")

    print("\nSetup complete. Next steps:")
    print("1. Ensure you have the reference CSVs inside the 'data/' folder.")
    print("2. Run 'pip install -r requirements.txt'")

if __name__ == "__main__":
    setup_environment()
