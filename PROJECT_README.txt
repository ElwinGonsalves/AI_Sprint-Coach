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