document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('video-input');
    const uploadArea = document.getElementById('upload-area');
    const previewSection = document.getElementById('preview-section');
    const videoPreview = document.getElementById('video-preview');
    const analyzeBtn = document.getElementById('analyze-btn');

    const landingSection = document.getElementById('landing-section');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');

    // Skeleton visualizer elements
    const canvas = document.getElementById('skeleton-canvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    const playBtn = document.getElementById('viz-play-btn');
    const scrubber = document.getElementById('viz-scrubber');
    const frameCounter = document.getElementById('viz-frame-counter');

    let selectedFile = null;
    let animId = null;
    let isPlaying = true;
    let currentFrame = 0;
    let skeletonFrames = [];

    // H36M kinematic tree bones
    const BONES = [
        [0, 1], [1, 2], [2, 3],        // Pelvis -> Right Hip -> Right Knee -> Right Ankle
        [0, 4], [4, 5], [5, 6],        // Pelvis -> Left Hip -> Left Knee -> Left Ankle
        [0, 7], [7, 8], [8, 9], [9, 10],// Pelvis -> Spine -> Clavicle -> Neck -> Head
        [8, 11], [11, 12], [12, 13],   // Clavicle -> Left Shoulder -> Left Elbow -> Left Wrist
        [8, 14], [14, 15], [15, 16]    // Clavicle -> Right Shoulder -> Right Elbow -> Right Wrist
    ];

    // Handle File Selection
    videoInput.addEventListener('change', handleFileSelect);

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--accent-color)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--card-border)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--card-border)';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            videoInput.files = e.dataTransfer.files;
            handleFileSelect({ target: videoInput });
        }
    });

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            const videoURL = URL.createObjectURL(selectedFile);
            videoPreview.src = videoURL;

            uploadArea.classList.add('hidden');
            previewSection.classList.remove('hidden');
        }
    }

    // Handle Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading state
        landingSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        simulateLoadingSteps();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.message || 'Analysis request failed (Status ' + response.status + ')');
            }

            displayResults(result);

        } catch (error) {
            console.error('Analysis error:', error);
            alert('Error during analysis: ' + error.message);
            loadingSection.classList.add('hidden');
            landingSection.classList.remove('hidden');
        }
    });

    function simulateLoadingSteps() {
        const steps = document.querySelectorAll('.loading-steps li');
        let currentStep = 0;

        const interval = setInterval(() => {
            if (currentStep < steps.length - 1) {
                steps[currentStep].classList.remove('active');
                currentStep++;
                steps[currentStep].classList.add('active');
            } else {
                clearInterval(interval);
            }
        }, 1200);
    }

    function displayResults(data) {
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        const quality = data.video_quality || {};
        const isSuitable = quality.suitable === true;

        // 1. Video Quality Result
        const qualityDiv = document.getElementById('quality-result');
        if (isSuitable) {
            qualityDiv.innerHTML = `
                <div class="status-icon success">✓</div>
                <div>
                    <div>${quality.message || 'Suitable for biomechanics analysis'}</div>
                    <small style="color: var(--text-muted);">
                        Analyzed ${quality.total_frames || 0} frames @ ${Math.round(quality.fps || 30)} FPS
                        (Detection confidence: ${Math.round((quality.detection_ratio || 1) * 100)}%)
                    </small>
                </div>
            `;
        } else {
            qualityDiv.innerHTML = `
                <div class="status-icon error">✗</div>
                <div>
                    <div>${quality.message || 'Video not suitable for analysis'}</div>
                    <small style="color: var(--text-muted); display: block; margin-top: 0.5rem;">
                        <strong>Tip:</strong> Ensure full-body visibility (head to toe), side/lateral camera angle, and at least 3 seconds of continuous movement.
                    </small>
                </div>
            `;
        }

        // 2. Gait Metrics
        const gait = data.gait_analysis || {};
        const features = data.features || {};
        const gaitMetrics = document.getElementById('gait-metrics');
        gaitMetrics.innerHTML = `
            <div class="metric-item">
                <span>Detected Cycles</span>
                <strong>${gait.detected_cycles !== undefined ? gait.detected_cycles : (isSuitable ? 0 : 'N/A')}</strong>
            </div>
            <div class="metric-item">
                <span>Normalized Pts</span>
                <strong>${gait.normalized_points !== undefined ? gait.normalized_points : (isSuitable ? 101 : 'N/A')}</strong>
            </div>
            <div class="metric-item">
                <span>Mean Right Knee</span>
                <strong>${features.mean_right_knee_angle !== undefined ? features.mean_right_knee_angle + '°' : '--'}</strong>
            </div>
            <div class="metric-item">
                <span>Mean Left Knee</span>
                <strong>${features.mean_left_knee_angle !== undefined ? features.mean_left_knee_angle + '°' : '--'}</strong>
            </div>
        `;

        // 3. Technique Findings
        const techniqueList = document.getElementById('technique-findings');
        techniqueList.innerHTML = '';
        const findings = data.technique_findings || data.reference_comparison || {};
        const findingKeys = Object.keys(findings);

        if (findingKeys.length === 0) {
            techniqueList.innerHTML = `<p style="color: var(--text-muted); font-size: 0.9rem;">No technique comparison available for this footage.</p>`;
        } else {
            for (const [joint, metrics] of Object.entries(findings)) {
                if (!metrics || typeof metrics !== 'object') continue;
                const isWithin = metrics.within_1std !== false;
                const badgeClass = isWithin ? 'success' : 'warn';
                const badgeText = isWithin ? 'Within Reference' : 'Deviation';

                techniqueList.innerHTML += `
                    <div class="technique-item">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4>${joint}</h4>
                            <span class="badge ${badgeClass}">${badgeText} (${metrics.deviation || '0°'})</span>
                        </div>
                        <div><span>Observed Mean:</span> <strong>${metrics.observed_value || '--'}</strong></div>
                        <div><span>Reference Mean:</span> <span>${metrics.reference_mean || '--'}</span></div>
                        <div><span>Reference Range:</span> <span>${metrics.reference_range || '--'}</span></div>
                    </div>
                `;
            }
        }

        // 4. Coaching Feedback (LLM - Sprint Coach)
        const coaching = data.coaching_feedback || {};
        const coachingDiv = document.getElementById('coaching-feedback');

        function formatCoachingContent(text) {
            if (!text) return '<p style="color: var(--text-muted);">No feedback recorded.</p>';

            // Clean any replacement characters
            let safe = text.replace(/\uFFFD/g, '°');

            // Escape basic HTML entities
            safe = safe
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Convert markdown bold **text** to <strong>text</strong>
            safe = safe.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

            // Convert bullet lines to <ul><li>
            const lines = safe.split('\n');
            let html = '';
            let inList = false;

            for (let rawLine of lines) {
                const line = rawLine.trim();
                if (!line) {
                    if (inList) {
                        html += '</ul>';
                        inList = false;
                    }
                    continue;
                }

                // Check bullet syntax: * item or - item or *   item
                const bulletMatch = line.match(/^(\*|-)\s+(.*)/);
                if (bulletMatch) {
                    if (!inList) {
                        html += '<ul class="coaching-bullets">';
                        inList = true;
                    }
                    html += `<li>${bulletMatch[2]}</li>`;
                } else {
                    if (inList) {
                        html += '</ul>';
                        inList = false;
                    }
                    html += `<p>${line}</p>`;
                }
            }

            if (inList) html += '</ul>';
            return html;
        }

        coachingDiv.innerHTML = `
            <div class="coaching-section">
                <h4>🔍 What We Observed (The Numbers in Plain English)</h4>
                ${formatCoachingContent(coaching.what_we_observed)}
            </div>
            <div class="coaching-section">
                <h4>⚡ What This Means For Your Sprint (Track Performance Impact)</h4>
                ${formatCoachingContent(coaching.possible_interpretation)}
            </div>
            <div class="coaching-section">
                <h4>🎯 Coaching Focus (Mental Cues & Track Drills)</h4>
                ${formatCoachingContent(coaching.coaching_focus)}
            </div>
            ${coaching.overall_summary ? `
            <div class="coaching-summary-box">
                <h4>🏁 Coach's Quick Takeaway</h4>
                ${formatCoachingContent(coaching.overall_summary)}
            </div>
            ` : ''}
        `;

        // 5. Pose Visualization
        initSkeletonVisualizer(data);
    }

    // ── Skeleton Visualizer Player ──────────────────────────────
    function initSkeletonVisualizer(data) {
        if (!canvas || !ctx) return;

        // Stop any existing animation
        if (animId) {
            cancelAnimationFrame(animId);
            animId = null;
        }

        // Extract frames (either 3D or 2D)
        if (data.pose_3d && Array.isArray(data.pose_3d.frames_sample) && data.pose_3d.frames_sample.length > 0) {
            skeletonFrames = data.pose_3d.frames_sample;
        } else if (data.pose_2d && Array.isArray(data.pose_2d.keypoints_sample) && data.pose_2d.keypoints_sample.length > 0) {
            skeletonFrames = data.pose_2d.keypoints_sample;
        } else {
            skeletonFrames = [];
        }

        if (skeletonFrames.length === 0) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#a0a5b1';
            ctx.font = '14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No skeleton pose data available', canvas.width / 2, canvas.height / 2);
            return;
        }

        currentFrame = 0;
        isPlaying = true;
        if (scrubber) {
            scrubber.max = skeletonFrames.length - 1;
            scrubber.value = 0;
        }
        if (playBtn) playBtn.textContent = 'Pause';

        let lastTime = 0;
        const fpsInterval = 1000 / 15; // 15 fps playback for smooth sampled viewing

        function animate(time) {
            animId = requestAnimationFrame(animate);

            if (!isPlaying) return;

            const elapsed = time - lastTime;
            if (elapsed > fpsInterval) {
                lastTime = time - (elapsed % fpsInterval);
                drawFrame(currentFrame);
                currentFrame = (currentFrame + 1) % skeletonFrames.length;
                if (scrubber) scrubber.value = currentFrame;
            }
        }

        animId = requestAnimationFrame(animate);
    }

    function drawFrame(frameIdx) {
        if (!canvas || !ctx || !skeletonFrames[frameIdx]) return;

        const joints = skeletonFrames[frameIdx]; // (17, 3) or (17, 2)
        const W = canvas.width;
        const H = canvas.height;

        ctx.clearRect(0, 0, W, H);

        // Normalize / fit points to canvas
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let j = 0; j < joints.length; j++) {
            const x = joints[j][0];
            const y = joints[j][1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        const rangeX = (maxX - minX) || 1;
        const rangeY = (maxY - minY) || 1;
        const scale = Math.min((W - 80) / rangeX, (H - 60) / rangeY);
        const offsetX = (W - rangeX * scale) / 2 - minX * scale;
        const offsetY = (H - rangeY * scale) / 2 - minY * scale;

        const coords = joints.map(pt => [
            pt[0] * scale + offsetX,
            pt[1] * scale + offsetY
        ]);

        // Draw bones
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.7)';

        for (const [a, b] of BONES) {
            if (a < coords.length && b < coords.length) {
                ctx.beginPath();
                ctx.moveTo(coords[a][0], coords[a][1]);
                ctx.lineTo(coords[b][0], coords[b][1]);
                ctx.stroke();
            }
        }

        // Draw joints
        for (let j = 0; j < coords.length; j++) {
            const [x, y] = coords[j];
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = j === 10 ? '#ff007f' : '#00f0ff'; // highlight head
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        if (frameCounter) {
            frameCounter.textContent = `${frameIdx + 1} / ${skeletonFrames.length}`;
        }
    }

    if (playBtn) {
        playBtn.addEventListener('click', () => {
            isPlaying = !isPlaying;
            playBtn.textContent = isPlaying ? 'Pause' : 'Play';
        });
    }

    if (scrubber) {
        scrubber.addEventListener('input', (e) => {
            isPlaying = false;
            if (playBtn) playBtn.textContent = 'Play';
            currentFrame = parseInt(e.target.value, 10);
            drawFrame(currentFrame);
        });
    }

    // Reset / New Analysis
    newAnalysisBtn.addEventListener('click', () => {
        if (animId) {
            cancelAnimationFrame(animId);
            animId = null;
        }
        selectedFile = null;
        videoInput.value = '';
        videoPreview.src = '';

        resultsSection.classList.add('hidden');
        previewSection.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        landingSection.classList.remove('hidden');

        // Reset loading steps
        const steps = document.querySelectorAll('.loading-steps li');
        steps.forEach(step => step.classList.remove('active'));
        steps[0].classList.add('active');
    });
});
