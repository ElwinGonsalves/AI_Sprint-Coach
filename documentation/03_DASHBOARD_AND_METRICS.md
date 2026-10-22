# Dashboard Interface & Metrics Guide

This document describes all metrics, visualizations, and user interface components rendered on the Sprint Coach dashboard.

---

### Dashboard Layout Overview

The user interface is designed as a single-page sports telemetry dashboard organized into a two-column responsive grid with a full-width coaching terminal:

```
+-----------------------------------------------------------------------------------+
|                        AI Sports Technique Coach (Header)                        |
+-----------------------------------------------------------------------------------+
|  [ Upload Video / Drag-and-Drop Dropzone / Video Preview / Analyze Button ]       |
+-----------------------------------------------------------------------------------+
|  ANALYSIS RESULTS                                         [Analyze New Video Btn] |
+-----------------------------------------+-----------------------------------------+
| [Card 1: Video Quality & Gate Status]   | [Card 2: 3D Pose Visualization]         |
|  - Status Icon (Check/Cross)            |  - Interactive HTML5 Canvas             |
|  - Frame count, FPS, Detection Ratio    |  - 17-Joint Skeleton with Bones         |
|  - Quality Advisory Diagnostics         |  - Play / Pause / Scrubber / Frame Ctr  |
+-----------------------------------------+-----------------------------------------+
| [Card 3: Gait Analysis Metrics]         | [Card 4: Technique Deviation Analysis]  |
|  - Detected Gait Cycles                 |  - Right Knee Flexion vs Baseline       |
|  - 101-Point Normalized Points          |  - Left Knee Flexion vs Baseline        |
|  - Mean Right Knee Flexion (deg)        |  - Right Hip Angle vs Baseline          |
|  - Mean Left Knee Flexion (deg)         |  - Left Hip Angle vs Baseline           |
|                                         |  - Status Badges (Within Ref / Deviate) |
+-----------------------------------------+-----------------------------------------+
| [Card 5: AI Coaching Feedback (Full Width)]                                       |
|  - Section 1: What We Observed                                                    |
|  - Section 2: Possible Interpretation                                             |
|  - Section 3: Coaching Focus                                                      |
|  - Section 4: Overall Summary                                                     |
+-----------------------------------------------------------------------------------+
```

---

### Detailed Card Specifications

#### 1. Video Quality Card (`quality-card`)
- **Status Indicator:**
  - Green Checkmark (`✓`): Video passed all quality gates.
  - Red Cross (`✗`): Video was rejected by suitability gates.
- **Metrics Rendered:**
  - **Analyzed Frames:** Total frames processed (e.g. `90 frames`).
  - **Frame Rate:** Video acquisition rate in FPS (e.g. `30 FPS`).
  - **Detection Confidence:** Percentage of frames where full human landmarks were identified (e.g. `95%`).
- **Unsuitable Handling:** If detection ratio is $< 60\%$ or total frames $< 81$, the card displays guidance on camera angles, lighting, and framing.

#### 2. Pose Visualization Card (`viz-card`)
- **Rendering Engine:** Dedicated HTML5 2D Canvas (`#skeleton-canvas`) running on `requestAnimationFrame`.
- **Skeleton Topology:** 17 joints connected according to the AthletePose3D / H36M kinematic tree:
  - Pelvis $\to$ Right Hip $\to$ Knee $\to$ Ankle (cyan line)
  - Pelvis $\to$ Left Hip $\to$ Knee $\to$ Ankle (cyan line)
  - Pelvis $\to$ Spine $\to$ Clavicle $\to$ Neck $\to$ Head (magenta head marker)
  - Clavicle $\to$ Shoulders $\to$ Elbows $\to$ Wrists (cyan lines)
- **Interactive Controls:**
  - **Play / Pause Toggle:** Controls temporal playback loop.
  - **Interactive Scrubber:** Scrub to examine any individual frame in the stride cycle.
  - **Frame Counter:** Displays `Current Frame / Total Sampled Frames`.
  - **Auto-Fit Projection:** Scales coordinates dynamically to maintain center framing regardless of camera distance.

#### 3. Gait Analysis Card (`gait-card`)
- **Detected Cycles:** Number of complete heel-strike-to-heel-strike gait strides identified via vertical ankle minima.
- **Normalized Points:** Displays `101`, verifying that strides were successfully resampled onto a $0\% \text{ to } 100\%$ standardized gait cycle continuum.
- **Mean Right Knee Angle:** Average 3D knee angle across all analyzed frames (in degrees).
- **Mean Left Knee Angle:** Average 3D knee angle for the opposite limb, enabling rapid asymmetry checks.

#### 4. Technique Analysis Card (`technique-card`)
Compares the runner's joint kinematics against the RBDS reference baseline:
- **Joint Titles:**
  - `Right Knee Flexion`
  - `Left Knee Flexion`
  - `Right Hip Angle`
  - `Left Hip Angle`
- **Telemetry Displayed:**
  - **Observed Mean:** The runner's average measured angle (e.g. `145.2°`).
  - **Reference Mean:** Average angle from the reference runner (e.g. `155.0°`).
  - **Reference Range:** Acceptable athletic envelope (e.g. `140.0°–165.0°`).
  - **Deviation Pill:** Absolute signed deviation with unit (e.g. `-9.8°` or `+1.2°`).
- **Status Badges:**
  - `Within Reference` (Green badge): Deviation is within $1.0 \times \text{Std Dev}$ of the baseline.
  - `Deviation` (Amber badge): Metric deviates outside normal baseline variance, highlighted for coach attention.

#### 5. AI Coaching Feedback Card (`coaching-card`)
Displays structured natural language feedback generated by Groq (Qwen/LLaMA):
- **What We Observed:** Pure factual recapitulation of measurements and video quality (no speculation).
- **Possible Interpretation:** Biomechanical hypothesis explaining why the deviation may occur (e.g., knee extension deficit during late swing phase).
- **Coaching Focus:** Actionable cue for the runner or coach during their next training session.
- **Overall Summary:** High-level synthesis of technique quality.

---

### Backend JSON Response Payload Reference

Whenever a request is sent to `POST /analyze`, the server returns the following structured JSON:

```json
{
  "status": "success",
  "video_quality": {
    "suitable": true,
    "message": "Suitable for analysis (body detected in 95% of frames)",
    "total_frames": 90,
    "fps": 30.0,
    "detection_ratio": 0.95
  },
  "pose_2d": {
    "status": "extracted",
    "total_frames": 90,
    "nan_ratio": 0.02,
    "keypoints_sample": [...]
  },
  "pose_3d": {
    "status": "generated",
    "total_frames": 81,
    "joints": 17,
    "frames_sample": [
      [ [x0, y0, z0], ..., [x16, y16, z16] ],
      ...
    ]
  },
  "features": {
    "status": "extracted",
    "frames": 81,
    "mean_right_knee_angle": 145.2,
    "mean_left_knee_angle": 154.1,
    "mean_right_hip_angle": 162.3,
    "mean_left_hip_angle": 163.0
  },
  "gait_analysis": {
    "detected_cycles": 2,
    "normalized_points": 101,
    "normalized_data": {
      "right_knee_angle": [101 floats],
      "left_knee_angle": [101 floats],
      "gait_cycle_percent": [0.0, 1.0, ..., 100.0]
    }
  },
  "technique_findings": {
    "Right Knee Flexion": {
      "observed_value": "145.2°",
      "reference_mean": "155.0°",
      "reference_range": "140.0–165.0°",
      "deviation": "-9.8°",
      "within_1std": true
    }
  },
  "coaching_feedback": {
    "what_we_observed": "The runner achieved 145.2° right knee flexion...",
    "possible_interpretation": "This indicates a possible deficit in terminal knee flexion...",
    "coaching_focus": "Focus on high knee drive and complete heel recovery drills...",
    "overall_summary": "Slight asymmetric flexion on the right leg, stable hip mechanics.",
    "raw_report": "..."
  }
}
```

