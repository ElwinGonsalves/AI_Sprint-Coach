# Sprint Coach Documentation Suite

Welcome to the comprehensive technical documentation for **Sprint Coach (AthletePose3D)**.

---

### Documentation Index

| File | Topic | Description |
|---|---|---|
| **[01_PROJECT_OVERVIEW.md](./01_PROJECT_OVERVIEW.md)** | **Project Overview & Architecture** | Executive summary, core capabilities, technology stack, directory structure, and quickstart instructions. |
| **[02_ARCHITECTURE_AND_PIPELINE.md](./02_ARCHITECTURE_AND_PIPELINE.md)** | **End-to-End Pipeline** | In-depth breakdown of the 8 pipeline stages from video ingestion, MediaPipe, TCPFormer, biomechanics, gait normalization to baseline comparison. |
| **[03_DASHBOARD_AND_METRICS.md](./03_DASHBOARD_AND_METRICS.md)** | **Dashboard & Telemetry** | Breakdown of all UI components, HTML5 Canvas 3D skeleton visualizer, metrics cards, deviation badges, and backend JSON response schema. |
| **[04_LLM_COACHING_AND_PROMPTS.md](./04_LLM_COACHING_AND_PROMPTS.md)** | **LLM Prompts & Guardrails** | Groq API model routing, exact system and user prompts, the 10 strict coaching rules, and regex section parser. |
| **[05_TROUBLESHOOTING_AND_DEPLOYMENT.md](./05_TROUBLESHOOTING_AND_DEPLOYMENT.md)** | **Troubleshooting & Operations** | Post-mortems of fixed issues (NaN JSON crash, TCPFormer CPU support, Groq models, cp1252), camera setup tips, and concurrency guide. |
| **[06_HOW_IT_WAS_BUILT_AND_INTERVIEW_PREP.md](./06_HOW_IT_WAS_BUILT_AND_INTERVIEW_PREP.md)** | **Master Preparation & Viva Guide** | 60-second elevator pitch, 5-minute technical narrative, phase-by-phase build history, math formulas, and Top 15 interview/viva Q&A with model answers. |
| **[RESEARCH_PAPER_RESULTS_AND_EVALUATION.md](./RESEARCH_PAPER_RESULTS_AND_EVALUATION.md)** | **Research Paper Evaluation & Results** | Rigorous empirical benchmarks ($N=25$ videos, $4,120$ frames), replacement tokens for Sec. IV-H, normative population baseline, and expert LLM feedback study ($n=5$ coaches/biomechanists). |

---

### Core Pipeline Quick Reference

```
Video Upload (.mp4/.mov)
       │
       ▼
[1] Video Quality & Suitability Gate (>= 81 frames, >= 60% detection)
       │
       ▼
[2] MediaPipe 2D Pose Extraction (COCO-17 Keypoints)
       │
       ▼
[3] COCO-17 to H36M Joint Remapping & Windowing (81 frames)
       │
       ▼
[4] TCPFormer 3D Pose Lifting (16-layer MemoryInduced Transformer)
       │
       ▼
[5] 3D Biomechanical Feature Extraction (Angles, Speeds, Leg Scale)
       │
       ▼
[6] Heel-Strike Detection & 101-Point Gait Normalization
       │
       ▼
[7] Reference Comparison vs RBDS Baseline (Mean, Range, Dev, 1-Std)
       │
       ▼
[8] Guardrailed AI Coaching via Groq (Qwen/LLaMA)
       │
       ▼
Interactive Web Dashboard & 3D Skeleton Playback
```

