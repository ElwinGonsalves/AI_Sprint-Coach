# Generative AI Coaching Layer: Prompts & Guardrails

This document describes the large language model (LLM) coaching layer that interprets biomechanical telemetry into friendly, actionable athletic guidance for everyday sprinters and coaches.

---

### 1. Design Philosophy: Athlete & Coach Understandability

In sports biomechanics, automated feedback systems often sound overly clinical, orthopedic, or academic—talking about "patients" and "flexion angle deficits" rather than how a sprinter actually moves down the track.

The AI Sprint Coach prompt translates complex 3D kinematic joint angles into intuitive sprinting concepts:
- **Knee Flexion:** Translated to **Heel Recovery & Knee Fold** (how tightly the heel snaps up under the hip during swing, and how freely the knee folds).
- **Hip Angle:** Translated to **Hip Extension & Punch** (pushing behind on push-off and punching the thigh forward in front).
- **Left vs. Right Differences:** Translated to **Stride Symmetry & Balance**.

#### Strict Coaching Guardrails:
1. **Coach-to-Athlete Tone:** Energetic, encouraging, constructive, and direct—like an experienced track coach at practice.
2. **NO Medical Jargon:** Never uses clinical words like "patient", "pathology", "etiology", or diagnoses injuries.
3. **Plain-English Angles:** Always explains what the degrees physically mean for the runner's legs.
4. **Concrete Track Cues & Drills:** Provides memorable mental cues (e.g., *"Step over the opposite knee"*, *"Punch the knee forward"*, *"Snap the heel up"*) and standard track drills (A-Skips, Wall Drives, High Knee Bleed-ins).
5. **Skimmable Formatting:** Clear bullet points and short paragraphs.

---

### 2. Model Routing & Fallback Architecture

The system communicates with the **Groq Cloud API** using low-latency inference with sequential model fallback:

```python
candidate_models = [
    "qwen/qwen3.8-27b",          # Primary: Outstanding reasoning and athletic phrasing
    "openai/gpt-oss-120b",       # Fallback 1: Ultra-large parameter open-weights model
    "openai/gpt-oss-20b",        # Fallback 2: Fast, efficient inference
    "llama-3.3-70b-versatile"    # Fallback 3: Meta LLaMA 3.3
]
```

---

### 3. Exact System Prompt

Below is the verbatim system prompt configured in `pipeline.py`:

```text
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
```

---

### 4. User Prompt Construction

The user prompt dynamically injects the telemetry dictionaries computed in Stages 1 and 7:

```python
user_prompt = f"""
Analyze these running biomechanics measurements for a sprinter:

VIDEO QUALITY:
{json.dumps(video_quality, indent=2)}

TECHNIQUE FINDINGS (Observed vs Elite Sprint Baseline):
{json.dumps(technique_findings, indent=2)}

Generate the sprinter coaching feedback using the four sections exactly as instructed.
"""
```

#### Hyperparameters
- **Temperature:** `0.3` (Low randomness to preserve factual consistency and prompt adherence)
- **Max Tokens:** `800`
- **Response Format:** Plain text parsed into structured sections

---

### 5. Robust Section Parsing & UI Markdown Formatter

The backend regex parser extracts the four sections regardless of header styling:

```python
import re

def extract_section(text, sec_name):
    pattern = (
        rf"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?{sec_name}(?:\*\*)?:?\s*\n?"
        rf"(.*?)(?=(?:\n\s*(?:#+\s*)?(?:\*\*)?"
        rf"(?:WHAT WE OBSERVED|POSSIBLE INTERPRETATION|COACHING FOCUS|OVERALL SUMMARY)"
        rf"(?:\*\*)?:?)|$)"
    )
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        val = m.group(1).strip()
        return re.sub(r"^\*+\s*", "", val).strip()
    return ""
```

In the frontend (`static/script.js`), a dedicated `formatCoachingContent` function translates bullet points (`* `, `- `) and bold tags (`**text**`) into styled HTML with custom accent borders and a highlighted **Coach's Quick Takeaway** banner.

---

### 6. Example In-Out Execution

#### Input Payload
```json
{
  "VIDEO QUALITY": {
    "suitable": true,
    "fps": 30.0,
    "total_frames": 90,
    "detection_ratio": 0.95
  },
  "TECHNIQUE FINDINGS": {
    "Right Knee Flexion": {
      "observed_value": "145.2°",
      "reference_mean": "155.0°",
      "reference_range": "140.0–165.0°",
      "deviation": "-9.8°",
      "within_1std": true
    },
    "Left Knee Flexion": {
      "observed_value": "154.5°",
      "reference_mean": "155.0°",
      "reference_range": "140.0–165.0°",
      "deviation": "-0.5°",
      "within_1std": true
    }
  }
}
```

#### Generated Coaching Output
- **WHAT WE OBSERVED:**
  The video footage provided 90 frames at 30 FPS with high detection quality (95%). The athlete demonstrated a mean right knee flexion of 145.2° and left knee flexion of 154.5°. While both limbs fall within the broader reference range (140.0°–165.0°), the right knee deviates by -9.8° relative to the reference mean, exhibiting an asymmetry of approximately 9.3° compared to the left limb.

- **POSSIBLE INTERPRETATION:**
  The lower flexion angle on the right knee may indicate reduced knee flexion during the swing phase or early stance contact. This bilateral discrepancy differs from the reference symmetry and may reflect a tendency to maintain a stiffer right leg through mid-stride.

- **COACHING FOCUS:**
  Consider focusing on bilateral balance and active right-leg heel recovery. Drills emphasizing dynamic knee drive and rhythmic stride symmetry (such as high-knee skips and step-over drills) may help encourage symmetrical lower-limb mechanics.

- **OVERALL SUMMARY:**
  The runner displays solid foundational movement with overall metrics within acceptable envelopes, with a possible asymmetry in right knee flexion worth monitoring during training.

