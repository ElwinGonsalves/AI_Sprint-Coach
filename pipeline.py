"""
Sprint Coach — Production ML Pipeline
======================================
Extracted and refactored from Sprint_model.ipynb (Colab).
All logic is taken directly from the verified notebook.

Steps:
  1. Video quality validation
  2. MediaPipe 2D pose extraction (COCO-17)
  3. COCO-17 → H36M mapping
  4. TCPFormer 3D pose estimation (81-frame windows)
  5. Biomechanical feature extraction
  6. Gait-cycle detection & 101-point normalization
  7. Reference comparison vs TCPFormer CSV
  8. Structured technique-deviation findings
  9. Groq LLM coaching feedback
"""

import os
import sys
import json
import re
import numpy as np
import cv2
import torch
import pandas as pd
from scipy.interpolate import interp1d
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TCPFORMER_DIR = os.path.join(BASE_DIR, "TCPFormer")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "TCPFormer_ap3d_81.pth.tr")
MP_MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")
REFERENCE_CSV = os.path.join(BASE_DIR, "data", "Running_0_cam_1_TCPFormer_features_normalized.csv")

# Add TCPFormer to path so its modules can be imported
if TCPFORMER_DIR not in sys.path:
    sys.path.insert(0, TCPFORMER_DIR)

# ─────────────────────────────────────────────
# Joint indices — AthletePose3D H36M order
# ─────────────────────────────────────────────
PELVIS        = 0
RIGHT_HIP     = 1
RIGHT_KNEE    = 2
RIGHT_ANKLE   = 3
LEFT_HIP      = 4
LEFT_KNEE     = 5
LEFT_ANKLE    = 6
SPINE         = 7
CLAVICLE      = 8
NECK          = 9
HEAD          = 10
LEFT_SHOULDER = 11
LEFT_ELBOW    = 12
LEFT_WRIST    = 13
RIGHT_SHOULDER= 14
RIGHT_ELBOW   = 15
RIGHT_WRIST   = 16

# MediaPipe → COCO-17 landmark indices
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# COCO-17 → H36M mapping (from notebook)
COCO_TO_H36M = {
    0: [12, 11],     # pelvis = avg(right_hip, left_hip)
    1: 12,           # right_hip
    2: 14,           # right_knee
    3: 16,           # right_ankle
    4: 11,           # left_hip
    5: 13,           # left_knee
    6: 15,           # left_ankle
    7: [6, 5, 12, 11],  # spine = avg(shoulders, hips)
    8: [6, 5],       # clavicle = avg(shoulders)
    9: [6, 5, 0, 0], # neck
    10: 0,           # head = nose
    11: 5,           # left_shoulder
    12: 7,           # left_elbow
    13: 9,           # left_wrist
    14: 6,           # right_shoulder
    15: 8,           # right_elbow
    16: 10           # right_wrist
}

# ─────────────────────────────────────────────
# 1. Video Quality Check
# ─────────────────────────────────────────────
def check_video_quality(video_path: str) -> dict:
    """
    Validates that the video has enough frames and detectable
    human poses in at least 60% of frames using MediaPipe.
    """
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < 81:
        cap.release()
        return {
            "suitable": False,
            "message": "Video is too short. Please upload a running video with at least 81 frames (~3 seconds).",
            "total_frames": total_frames,
            "fps": fps,
            "detection_ratio": 0.0
        }

    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    detected = 0
    sample_every = max(1, total_frames // 30)  # sample ~30 frames for speed
    sampled = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                sampled += 1
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    detected += 1
            frame_idx += 1

    cap.release()

    detection_ratio = detected / sampled if sampled > 0 else 0.0

    if detection_ratio < 0.6:
        return {
            "suitable": False,
            "message": (
                f"Video is not suitable for analysis (body detected in only {detection_ratio:.0%} of frames). "
                "Please upload a clear running video where the full body and leg movement are visible."
            ),
            "total_frames": total_frames,
            "fps": fps,
            "detection_ratio": round(detection_ratio, 3)
        }

    return {
        "suitable": True,
        "message": f"Suitable for analysis (body detected in {detection_ratio:.0%} of frames)",
        "total_frames": total_frames,
        "fps": fps,
        "detection_ratio": round(detection_ratio, 3)
    }


# ─────────────────────────────────────────────
# 2. MediaPipe 2D Extraction → COCO-17
# ─────────────────────────────────────────────
def extract_2d_pose(video_path: str) -> np.ndarray:
    """
    Reads every frame of the video and runs MediaPipe Pose.
    Returns (N, 17, 2) array of normalized COCO-17 keypoints.
    """
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    coco_pose = np.full((total_frames, 17, 2), np.nan, dtype=np.float32)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                for coco_idx, mp_idx in enumerate(MP_TO_COCO):
                    lm = landmarks[mp_idx]
                    coco_pose[frame_idx, coco_idx, 0] = lm.x
                    coco_pose[frame_idx, coco_idx, 1] = lm.y
            frame_idx += 1

    cap.release()
    return coco_pose[:frame_idx]


# ─────────────────────────────────────────────
# 3. COCO-17 → H36M + TCPFormer Input Prep
# ─────────────────────────────────────────────
def prepare_tcpformer_input(coco_2d: np.ndarray) -> np.ndarray:
    """
    Converts COCO-17 2D pose to H36M format, normalises to [-1,1],
    adds confidence channel, and creates 81-frame sliding windows.
    Returns (W, 81, 17, 3) tensor-ready array.
    """
    num_frames = coco_2d.shape[0]

    # Map COCO → H36M
    h36m_2d = np.zeros((num_frames, 17, 2), dtype=np.float32)
    for h36m_idx, coco_idx in COCO_TO_H36M.items():
        if isinstance(coco_idx, list):
            # Average of multiple joints, ignoring NaN
            vals = coco_2d[:, coco_idx]
            h36m_2d[:, h36m_idx] = np.nanmean(vals, axis=1)
        else:
            h36m_2d[:, h36m_idx] = coco_2d[:, coco_idx]

    # Replace remaining NaNs with 0
    h36m_2d = np.nan_to_num(h36m_2d, nan=0.0)

    # Normalise MediaPipe [0,1] → [-1,1] (TCPFormer convention)
    h36m_norm = h36m_2d.copy()
    h36m_norm[:, :, 0] = h36m_norm[:, :, 0] * 2.0 - 1.0
    h36m_norm[:, :, 1] = h36m_norm[:, :, 1] * 2.0 - 1.0

    # Add confidence = 1.0 for all joints
    confidence = np.ones((num_frames, 17, 1), dtype=np.float32)
    tcp_input_full = np.concatenate([h36m_norm, confidence], axis=-1)

    # Create 81-frame windows (non-overlapping)
    WINDOW = 81
    windows = []
    for start in range(0, num_frames - WINDOW + 1, WINDOW):
        windows.append(tcp_input_full[start:start + WINDOW])

    if len(windows) == 0:
        raise ValueError(f"Video too short: need at least 81 frames, got {num_frames}.")

    return np.stack(windows)  # (W, 81, 17, 3)


# ─────────────────────────────────────────────
# 4. TCPFormer 3D Pose Generation
# ─────────────────────────────────────────────
_tcpformer_model = None  # module-level cache

def get_tcpformer_model():
    global _tcpformer_model
    if _tcpformer_model is not None:
        return _tcpformer_model

    from model.Model import MemoryInducedTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"[TCPFormer] Loaded on {device}")
    _tcpformer_model = (model, device)
    return _tcpformer_model


def run_tcpformer(tcp_input: np.ndarray) -> np.ndarray:
    """
    Runs TCPFormer inference on (W, 81, 17, 3) windows.
    Returns (N_frames, 17, 3) combined 3D pose.
    """
    model, device = get_tcpformer_model()

    input_tensor = torch.from_numpy(tcp_input).float().to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    pose_3d_windows = prediction.detach().cpu().numpy()  # (W, 81, 17, 3)
    pose_3d = pose_3d_windows.reshape(-1, 17, 3)         # (W*81, 17, 3)
    return pose_3d


# ─────────────────────────────────────────────
# 5. Biomechanical Feature Extraction
# ─────────────────────────────────────────────
def angle_3points(pose, a_idx, b_idx, c_idx):
    a = pose[:, a_idx]
    b = pose[:, b_idx]
    c = pose[:, c_idx]
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cosine = np.sum(ba * bc, axis=1) / np.maximum(denom, 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def frame_speed(points):
    speed = np.zeros(len(points), dtype=np.float32)
    speed[1:] = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return speed


def extract_features(pose_3d: np.ndarray) -> pd.DataFrame:
    """
    Extracts all biomechanical features from 3D pose sequence.
    Exactly mirrors the notebook feature extraction cell.
    """
    right_knee_angle  = angle_3points(pose_3d, RIGHT_HIP,  RIGHT_KNEE,  RIGHT_ANKLE)
    left_knee_angle   = angle_3points(pose_3d, LEFT_HIP,   LEFT_KNEE,   LEFT_ANKLE)
    right_hip_angle   = angle_3points(pose_3d, PELVIS,     RIGHT_HIP,   RIGHT_KNEE)
    left_hip_angle    = angle_3points(pose_3d, PELVIS,     LEFT_HIP,    LEFT_KNEE)

    knee_angle_difference = np.abs(right_knee_angle - left_knee_angle)
    hip_angle_difference  = np.abs(right_hip_angle  - left_hip_angle)

    dist = lambda a, b: np.linalg.norm(pose_3d[:, a] - pose_3d[:, b], axis=1)

    hip_width       = dist(RIGHT_HIP, LEFT_HIP)
    shoulder_width  = dist(RIGHT_SHOULDER, LEFT_SHOULDER)
    right_leg_len   = dist(RIGHT_HIP, RIGHT_KNEE) + dist(RIGHT_KNEE, RIGHT_ANKLE)
    left_leg_len    = dist(LEFT_HIP, LEFT_KNEE)  + dist(LEFT_KNEE,  LEFT_ANKLE)
    knee_separation = dist(RIGHT_KNEE, LEFT_KNEE)
    ankle_separation= dist(RIGHT_ANKLE, LEFT_ANKLE)

    right_ankle_speed = frame_speed(pose_3d[:, RIGHT_ANKLE])
    left_ankle_speed  = frame_speed(pose_3d[:, LEFT_ANKLE])
    right_knee_speed  = frame_speed(pose_3d[:, RIGHT_KNEE])
    left_knee_speed   = frame_speed(pose_3d[:, LEFT_KNEE])
    ankle_speed_diff  = np.abs(right_ankle_speed - left_ankle_speed)

    features = pd.DataFrame({
        "frame":                   np.arange(len(pose_3d)),
        "right_knee_angle":        right_knee_angle,
        "left_knee_angle":         left_knee_angle,
        "right_hip_angle":         right_hip_angle,
        "left_hip_angle":          left_hip_angle,
        "knee_angle_difference":   knee_angle_difference,
        "hip_angle_difference":    hip_angle_difference,
        "hip_width":               hip_width,
        "shoulder_width":          shoulder_width,
        "right_leg_length":        right_leg_len,
        "left_leg_length":         left_leg_len,
        "knee_separation":         knee_separation,
        "ankle_separation":        ankle_separation,
        "right_ankle_speed":       right_ankle_speed,
        "left_ankle_speed":        left_ankle_speed,
        "right_knee_speed":        right_knee_speed,
        "left_knee_speed":         left_knee_speed,
        "ankle_speed_difference":  ankle_speed_diff,
    })

    # Normalize distance/speed by average leg length
    body_scale = ((features["right_leg_length"] + features["left_leg_length"]) / 2.0).replace(0, np.nan)
    for col in ["hip_width", "shoulder_width", "knee_separation", "ankle_separation",
                "right_leg_length", "left_leg_length", "right_ankle_speed",
                "left_ankle_speed", "right_knee_speed", "left_knee_speed", "ankle_speed_difference"]:
        features[f"{col}_norm"] = features[col] / body_scale

    return features


# ─────────────────────────────────────────────
# 6. Gait-Cycle Detection & 101-point Normalization
# ─────────────────────────────────────────────
def detect_gait_cycles(pose_3d: np.ndarray) -> list:
    """
    Detects gait cycle boundaries using the vertical (Y) position
    of the right ankle — local minima = heel strikes.
    Returns a list of (start_frame, end_frame) tuples.
    """
    from scipy.signal import find_peaks

    right_ankle_y = pose_3d[:, RIGHT_ANKLE, 1]

    # Heel strikes = local minima in ankle Y (downward peaks)
    peaks, _ = find_peaks(-right_ankle_y, distance=15, prominence=0.005)

    cycles = []
    for i in range(len(peaks) - 1):
        cycles.append((peaks[i], peaks[i + 1]))

    return cycles


def normalize_gait_cycles(features: pd.DataFrame, cycles: list) -> pd.DataFrame:
    """
    Resamples each gait cycle to 101 points (0-100% gait cycle),
    then averages across all cycles.
    """
    angle_cols = ["right_knee_angle", "left_knee_angle",
                  "right_hip_angle", "left_hip_angle",
                  "knee_angle_difference", "hip_angle_difference"]

    all_cycles = []
    for start, end in cycles:
        cycle_data = features.iloc[start:end][angle_cols].values
        if len(cycle_data) < 5:
            continue
        x_old = np.linspace(0, 100, len(cycle_data))
        x_new = np.linspace(0, 100, 101)
        resampled = {}
        for j, col in enumerate(angle_cols):
            f = interp1d(x_old, cycle_data[:, j], kind="linear")
            resampled[col] = f(x_new)
        all_cycles.append(pd.DataFrame(resampled))

    if not all_cycles:
        return pd.DataFrame()

    avg = pd.concat(all_cycles).groupby(level=0).mean()
    avg["gait_cycle_percent"] = np.linspace(0, 100, 101)
    return avg


# ─────────────────────────────────────────────
# 7. Reference Comparison
# ─────────────────────────────────────────────
def compare_to_reference(features: pd.DataFrame) -> dict:
    """
    Compares extracted features against the TCPFormer reference CSV
    (Running_0_cam_1_TCPFormer_features_normalized.csv).
    Returns per-joint mean observed vs reference range.
    """
    try:
        ref = pd.read_csv(REFERENCE_CSV)
    except FileNotFoundError:
        return {"status": "reference_not_found"}

    comparison = {}
    feature_cols = {
        "right_knee_angle": ("right_knee_angle", "Right Knee Flexion"),
        "left_knee_angle":  ("left_knee_angle",  "Left Knee Flexion"),
        "right_hip_angle":  ("right_hip_angle",  "Right Hip Angle"),
        "left_hip_angle":   ("left_hip_angle",   "Left Hip Angle"),
    }

    for feat_key, (ref_col, label) in feature_cols.items():
        if feat_key not in features.columns or ref_col not in ref.columns:
            continue
        observed_mean = float(features[feat_key].mean()) if not np.isnan(features[feat_key].mean()) else 0.0
        ref_mean  = float(ref[ref_col].mean())
        ref_std   = float(ref[ref_col].std())
        ref_min   = float(ref[ref_col].min())
        ref_max   = float(ref[ref_col].max())
        deviation = observed_mean - ref_mean

        comparison[label] = {
            "observed_value": f"{observed_mean:.1f}°",
            "reference_mean": f"{ref_mean:.1f}°",
            "reference_range": f"{ref_min:.1f}–{ref_max:.1f}°",
            "deviation": f"{deviation:+.1f}°",
            "within_1std": bool(abs(deviation) <= ref_std)
        }

    return comparison


# ─────────────────────────────────────────────
# 8. Groq LLM Coaching Feedback
# ─────────────────────────────────────────────
def generate_coaching_feedback(technique_findings: dict, video_quality: dict) -> dict:
    """
    Sends structured biomechanical findings to Groq (Qwen/LLaMA)
    and returns parsed coaching feedback sections.
    Uses the same strict system prompt as the verified Colab notebook.
    """
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {
            "what_we_observed": "Groq API key not configured.",
            "possible_interpretation": "Please set the GROQ_API_KEY environment variable in your .env file.",
            "coaching_focus": "Add GROQ_API_KEY=gsk_... to .env file to enable AI coaching.",
            "overall_summary": "API key required for automated LLM coaching feedback.",
            "raw_report": ""
        }

    client = Groq(api_key=api_key)

    system_prompt = """
You are an elite track & field sprint coach and biomechanics specialist.
Your goal is to give feedback that any everyday sprinter, school/club athlete, or track coach can immediately understand and put into practice on the track.

Translate technical motion data into intuitive sprinting language:
- Knee Flexion = Heel Recovery & Knee Fold (how tightly the heel snaps up under the glute during recovery, and how cleanly the knee drives up in front).
- Hip Angle = Hip Extension & Punch (driving force behind you on push-off, and punching the thigh forward in front).
- Left vs. Right Differences = Stride Balance & Symmetry (ensuring both legs generate equal power and cycle at the same tempo).

COACHING RULES:
1. Tone: Energetic, constructive, and direct—like an experienced track coach at practice.
2. NO Medical Jargon: Never use clinical or medical terms (do NOT say 'patient', 'pathology', 'etiology', 'deficit', or diagnose medical conditions). This is purely sports performance technique analysis.
3. Plain-English Angles: Always explain what the numbers actually mean in real sprinting (e.g. 'Your right knee averaged 145° compared to the 155° benchmark, meaning your right leg swings slightly open rather than folding up compact').
4. Actionable Cues & Drills: In COACHING FOCUS, provide 1-2 memorable mental cues (e.g. 'Step over the opposite knee', 'Punch the knee forward', 'Snap the heel up') and 2-3 common track drills (e.g. A-Skips, Wall Drives, High Knee Bleed-ins, B-Skips).
5. Formatting: Use clean bullet points and short, punchy paragraphs so it is effortless to read on a trackside phone or tablet.

Structure your response with EXACTLY these four sections:

WHAT WE OBSERVED:
<Plain-English breakdown of video quality, key angles, and what the runner's legs are doing>

POSSIBLE INTERPRETATION:
<How this affects the athlete's speed, ground contact time, stride turnover, and energy>

COACHING FOCUS:
<Actionable mental cues and 2-3 specific sprint drills to practice on the track>

OVERALL SUMMARY:
<A 2-3 sentence punchy bottom line the sprinter can keep in mind right at the starting line>
"""

    user_prompt = f"""
Analyze these running biomechanics measurements for a sprinter:

VIDEO QUALITY:
{json.dumps(video_quality, indent=2)}

TECHNIQUE FINDINGS (Observed vs Elite Sprint Baseline):
{json.dumps(technique_findings, indent=2)}

Generate the sprinter coaching feedback using the four sections exactly as instructed.
"""

    # List of candidate models to try in order of preference
    candidate_models = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
    raw = None
    last_err = None

    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                raw = content.strip()
                break
        except Exception as e:
            last_err = e
            continue

    if not raw:
        return {
            "what_we_observed": f"Coaching feedback unavailable: {str(last_err)}",
            "possible_interpretation": "",
            "coaching_focus": "",
            "overall_summary": "",
            "raw_report": ""
        }

    # Robust regex section extraction
    def extract_section(text, sec_name):
        pattern = rf"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?{sec_name}(?:\*\*)?:?\s*\n?(.*?)(?=(?:\n\s*(?:#+\s*)?(?:\*\*)?(?:WHAT WE OBSERVED|POSSIBLE INTERPRETATION|COACHING FOCUS|OVERALL SUMMARY)(?:\*\*)?:?)|$)"
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip()
            return re.sub(r"^\*+\s*", "", val).strip()
        return ""

    obs = extract_section(raw, "WHAT WE OBSERVED")
    interp = extract_section(raw, "POSSIBLE INTERPRETATION")
    focus = extract_section(raw, "COACHING FOCUS")
    summary = extract_section(raw, "OVERALL SUMMARY")

    if not obs and not interp and not focus and not summary:
        obs = raw

    return {
        "what_we_observed": obs,
        "possible_interpretation": interp,
        "coaching_focus": focus,
        "overall_summary": summary,
        "raw_report": raw
    }


# ─────────────────────────────────────────────
# 9. Master Pipeline Entry Point
# ─────────────────────────────────────────────
def analyze_video_pipeline(video_path: str) -> dict:
    """
    Full end-to-end pipeline. Called by FastAPI /analyze endpoint.
    """
    print(f"\n{'='*50}")
    print(f"Starting analysis: {video_path}")
    print(f"{'='*50}\n")

    # ── Step 1: Video Quality ──────────────────
    print("[1/8] Checking video quality...")
    video_quality = check_video_quality(video_path)
    if not video_quality["suitable"]:
        return {
            "status": "unsuitable",
            "video_quality": video_quality,
            "pose_2d": {}, "pose_3d": {}, "gait_analysis": {},
            "features": {}, "reference_comparison": {},
            "technique_findings": {}, "coaching_feedback": {}
        }

    # ── Step 2: MediaPipe 2D Extraction ───────
    print("[2/8] Running MediaPipe 2D pose extraction...")
    coco_2d = extract_2d_pose(video_path)
    nan_ratio = np.isnan(coco_2d).mean()
    pose_2d_result = {
        "status": "extracted",
        "total_frames": int(coco_2d.shape[0]),
        "nan_ratio": round(float(nan_ratio), 3),
        # Pass the 2D keypoints to the frontend for the skeleton overlay (nan replaced with 0.0)
        "keypoints_sample": np.nan_to_num(coco_2d[::5], nan=0.0).tolist()
    }

    # ── Step 3 & 4: TCPFormer 3D Pose ─────────
    print("[3/8] Preparing TCPFormer input (COCO -> H36M)...")
    tcp_input = prepare_tcpformer_input(coco_2d)

    print("[4/8] Running TCPFormer 3D pose estimation...")
    pose_3d = run_tcpformer(tcp_input)
    pose_3d_result = {
        "status": "generated",
        "total_frames": int(pose_3d.shape[0]),
        "joints": 17,
        # Pass every 5th frame for 3D visualization (nan replaced with 0.0)
        "frames_sample": np.nan_to_num(pose_3d[::5], nan=0.0).tolist()
    }

    # ── Step 5: Feature Extraction ────────────
    print("[5/8] Extracting biomechanical features...")
    features_df = extract_features(pose_3d)

    def safe_mean(col):
        val = features_df[col].mean()
        return 0.0 if np.isnan(val) else round(float(val), 2)

    features_summary = {
        "status": "extracted",
        "frames": int(len(features_df)),
        "mean_right_knee_angle": safe_mean("right_knee_angle"),
        "mean_left_knee_angle":  safe_mean("left_knee_angle"),
        "mean_right_hip_angle":  safe_mean("right_hip_angle"),
        "mean_left_hip_angle":   safe_mean("left_hip_angle"),
    }

    # ── Step 6: Gait Cycle Detection ──────────
    print("[6/8] Detecting gait cycles...")
    cycles = detect_gait_cycles(pose_3d)
    normalized_gait = normalize_gait_cycles(features_df, cycles)
    gait_analysis = {
        "detected_cycles": len(cycles),
        "normalized_points": 101 if not normalized_gait.empty else 0,
        "normalized_data": normalized_gait.fillna(0.0).to_dict(orient="list") if not normalized_gait.empty else {}
    }

    # ── Step 7: Reference Comparison ──────────
    print("[7/8] Comparing against reference data...")
    reference_comparison = compare_to_reference(features_df)
    technique_findings = reference_comparison  # same object used in frontend

    # ── Step 8: Groq LLM Coaching ─────────────
    print("[8/8] Generating AI coaching feedback via Groq...")
    coaching_feedback = generate_coaching_feedback(technique_findings, video_quality)

    print("\n[OK] Analysis complete.\n")

    return {
        "status": "success",
        "video_quality": video_quality,
        "pose_2d": pose_2d_result,
        "pose_3d": pose_3d_result,
        "gait_analysis": gait_analysis,
        "features": features_summary,
        "reference_comparison": reference_comparison,
        "technique_findings": technique_findings,
        "coaching_feedback": coaching_feedback
    }
