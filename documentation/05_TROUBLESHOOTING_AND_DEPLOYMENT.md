# Troubleshooting, Bug Fixes & Deployment Guide

This document captures historical issues, root cause analyses, debugging strategies, and operational guidelines for running Sprint Coach in local development and production.

---

### 1. Root Cause Analyses of Solved Bugs

#### Bug 1: `ValueError: Out of range float values are not JSON compliant: nan` (500 Error during `POST /analyze`)
- **Symptoms:** The entire 8-step ML pipeline finished successfully and printed `[OK] Analysis complete.`, but Uvicorn immediately responded with `500 Internal Server Error`.
- **Root Cause:** When human landmarks are partially out of frame or occluded, MediaPipe initializes keypoints as `np.nan`. Downstream operations like `np.nanmean` on empty slices retain `NaN`. When FastAPI's `JSONResponse(content=results)` invoked Starlette's `json.dumps()`, Python's JSON encoder strictly enforced RFC 8259 compliance, which forbids `NaN` and `Infinity`.
- **Resolution:**
  1. In `pipeline.py`, applied `np.nan_to_num(coco_2d[::5], nan=0.0)` and `np.nan_to_num(pose_3d[::5], nan=0.0)`.
  2. Applied `.fillna(0.0)` across normalized gait cycle tables.
  3. Implemented a recursive `sanitize_for_json(obj)` function in `app.py` that intercepts all `math.isnan()` and `math.isinf()` values and replaces them with `0.0`, while casting NumPy scalar types (`np.float32`, `np.bool_`) to native Python primitives.

---

#### Bug 2: `AssertionError: Torch not compiled with CUDA enabled`
- **Symptoms:** Loading TCPFormer crashed during `model = MemoryInducedTransformer(...)`.
- **Root Cause:** In `TCPFormer/model/Model.py` (line 246), the model initialized its learnable layer scale parameters with hardcoded CUDA placement:
  ```python
  for i in range(len(self.layer_scale)):
      self.layer_scale[i] = self.layer_scale[i].to('cuda')  # Fails on CPU
  ```
- **Resolution:**
  Replaced hardcoded `.to('cuda')` with dynamic device checking:
  ```python
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  for i in range(len(self.layer_scale)):
      self.layer_scale[i] = self.layer_scale[i].to(device)
  ```
  And in the `forward` pass, ensured tensor matching via:
  ```python
  self.layer_scale[i].to(x[i].device)
  ```

---

#### Bug 3: `groq.NotFoundError: model_not_found` (404)
- **Symptoms:** `generate_coaching_feedback` returned `Coaching feedback unavailable: Error code: 404 - The model llama-3.3-70b-versatile does not exist or you do not have access to it.`
- **Root Cause:** The Groq account in use had access to Qwen and GPT-OSS models (`qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`), but the endpoint requested a non-existent or inaccessible model name.
- **Resolution:**
  Targeted `qwen/qwen3.8-27b` as the primary model with graceful fallback through candidate models and added module-level `load_dotenv()` in `pipeline.py`.

---

#### Bug 4: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2717'`
- **Symptoms:** Running terminal scripts or logging on Windows crashed with `UnicodeEncodeError`.
- **Root Cause:** Windows PowerShell and Command Prompt default to `cp1252` encoding, which cannot map Unicode symbols like `✗` (`\u2717`) or `✓` (`\u2713`).
- **Resolution:** Sanitized all backend log strings to ASCII-safe tags (`[OK]`, `[X]`).

---

### 2. Best Practices for Athlete Video Capture

For reliable pose detection and accurate gait cycle extraction:
1. **Camera Position:**
   - Place the camera directly perpendicular (lateral/sagittal view) to the runner's line of motion.
   - Maintain a height roughly level with the athlete's hips (approx. 1.0 to 1.2 meters).
2. **Framing & Framing Distance:**
   - Keep the runner's entire body visible (from top of head to soles of the running shoes) throughout the stride cycle.
   - Avoid extreme wide-angle or fish-eye lenses.
3. **Lighting & Contrast:**
   - Avoid severe backlighting (e.g. running directly toward the sun).
   - High-contrast clothing relative to the background improves MediaPipe confidence.
4. **Duration & Frame Rate:**
   - Minimum 81 frames (approx. 3.0 seconds at standard 30 FPS).
   - Higher frame rates (60 FPS or 120 FPS slow-motion) provide clearer heel-strike identification.

---

### 3. Server Concurrency & Performance

- **Thread Pool Offloading:**
  Because TCPFormer inference and MediaPipe video decoding are CPU/GPU-bound tasks, `app.py` offloads them via `ThreadPoolExecutor(max_workers=1)`. This ensures that the FastAPI asynchronous event loop remains responsive to incoming HTTP requests, health checks, and static asset serving.
- **Temporary Upload Management:**
  Every upload is saved to `uploads/<uuid>_<filename>` and cleared inside a `finally:` block after pipeline completion to prevent disk exhaustion.

---

### 4. Health Check Endpoint

A dedicated health probe is available at `GET /api/health`:
```bash
curl http://127.0.0.1:8000/api/health
```
Response:
```json
{
  "status": "healthy",
  "groq_configured": true,
  "models": {
    "pose_landmarker": true,
    "tcpformer": true,
    "reference_data": true
  }
}
```

