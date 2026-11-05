---
title: AI Sprint Coach
emoji: 🏃
colorFrom: cyan
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🏃 AI Sprint Coach (AthletePose3D)

> **End-to-End Monocular 3D Biomechanics & AI Technique Coaching Web Application**  
> Powered by **FastAPI**, **Google MediaPipe**, **TCPFormer (Transformer 3D Pose)**, and **Groq (Qwen/LLaMA)**.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Groq](https://img.shields.io/badge/Groq-API%20Llama%2FQwen-F55036?style=flat-square)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

---

## 📖 Overview

**AI Sprint Coach** is an AI-driven sports biomechanics platform that transforms standard mobile phone running video into clinical-grade biomechanical insights, interactive 3D kinematic playback, and actionable athletic coaching cues without requiring expensive motion-capture marker systems or force plates.

### Core Pipeline
```
Runner Video (.mp4/.mov)
       │
       ▼
[1] Video Quality & Suitability Gate (>=81 frames, >=60% body detection)
       │
       ▼
[2] MediaPipe 2D Pose Estimation (17 COCO keypoints)
       │
       ▼
[3] COCO-17 to H36M Joint Remapping & 81-frame Windowing
       │
       ▼
[4] TCPFormer 3D Pose Estimation (16-layer MemoryInduced Transformer)
       │
       ▼
[5] 3D Biomechanical Feature Extraction (Joint angles, speeds, leg scaling)
       │
       ▼
[6] Heel-Strike Detection & 101-Point Gait Normalization
       │
       ▼
[7] Reference Comparison vs. RBDS Baseline (Deviations, Ranges, ±1 Std)
       │
       ▼
[8] Guardrailed AI Coaching Feedback via Groq (Qwen/LLaMA)
       │
       ▼
Interactive Web Dashboard & 3D Skeleton Playback
```

---

## ✨ Features

- **Monocular 3D Kinematics:** Reconstructs full 3D spatial joint coordinates `(x, y, z)` from a single camera angle using TCPFormer.
- **Automated Gait Normalization:** Detects heel-strike events via vertical ankle trajectory minima and resamples strides to a standardized 0%–100% (101-point) gait cycle.
- **Reference Comparison Engine:** Automatically benchmarks observed knee flexion, hip angles, and asymmetries against the RBDS running dataset.
- **Live 3D Skeleton Player:** Interactive HTML5 Canvas player rendering the 17-joint skeleton with bone lines, play/pause controls, and frame scrubber.
- **Guardrailed AI Coach:** Powered by Groq LLMs under 10 strict ethical coaching rules—ensures observations, interpretations, and coaching focuses are factually grounded without hallucinations or injury diagnoses.
- **Safe JSON & CPU/CUDA Support:** Includes automated JSON sanitization (`NaN`/`Infinity` safe) and CPU/CUDA device compatibility.

---

## 🛠️ Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/ElwinGonsalves/AI_Sprint-Coach.git
cd AI_Sprint-Coach
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Checkpoints
Run the automated downloader to fetch the MediaPipe Pose Landmarker and the pre-trained TCPFormer 3D checkpoint:
```bash
python setup_ml.py
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and add your [Groq API Key](https://console.groq.com):
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 6. Run the Application
```bash
uvicorn app:app --reload --port 8000
```
Open your browser at **`http://localhost:8000`**.

---

## 📊 Dashboard Telemetry

| Card | Description |
|---|---|
| **Video Quality** | Frame count, FPS, pose detection ratio, and camera framing advisories |
| **Pose Visualization** | Interactive canvas player rendering real 3D/2D skeleton joint trajectory |
| **Gait Analysis** | Stride cycle count, 101-point normalization status, and bilateral flexion means |
| **Technique Analysis** | Per-joint comparison (observed vs. reference baseline mean, ranges, deviation badges) |
| **AI Coaching Feedback** | 4-part structured report (*What We Observed*, *Interpretation*, *Coaching Focus*, *Summary*) |

---

## 📚 Technical Documentation

Detailed technical documents are available in the [`documentation/`](./documentation/) directory:

- **[01_PROJECT_OVERVIEW.md](./documentation/01_PROJECT_OVERVIEW.md)**: Executive summary, architecture, directory tree, and quickstart.
- **[02_ARCHITECTURE_AND_PIPELINE.md](./documentation/02_ARCHITECTURE_AND_PIPELINE.md)**: Math and implementation of all 8 pipeline stages.
- **[03_DASHBOARD_AND_METRICS.md](./documentation/03_DASHBOARD_AND_METRICS.md)**: Full UI specification, Canvas engine, and JSON response schema.
- **[04_LLM_COACHING_AND_PROMPTS.md](./documentation/04_LLM_COACHING_AND_PROMPTS.md)**: System prompt, 10 coaching guardrails, and regex parser.
- **[05_TROUBLESHOOTING_AND_DEPLOYMENT.md](./documentation/05_TROUBLESHOOTING_AND_DEPLOYMENT.md)**: Resolved bug post-mortems, capture guidelines, and concurrency.
- **[06_HOW_IT_WAS_BUILT_AND_INTERVIEW_PREP.md](./documentation/06_HOW_IT_WAS_BUILT_AND_INTERVIEW_PREP.md)**: Master interview & viva guide, 5-minute technical story, math formulas, and Top 15 Q&A with model answers.

---

## 📄 License

This project is licensed under the MIT License.

