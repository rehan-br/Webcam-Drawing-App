import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision';
import { Gesture, GestureStabilizer, mapMediapipeGesture, geometricGesture } from './gestures.js';
import { DrawingEngine } from './drawing.js';
import { LandmarkSmoother } from './filters.js';

// Requested capture size. Most webcams will deliver this; if not, the
// canvas re-syncs to whatever the camera actually returned.
const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;
const GRACE_FRAMES_DEFAULT = 8;
const GRACE_FRAMES_DRAW = 24;
const VELOCITY_SMOOTHING = 0.3;
const BRUSH_STEP = 0.2;
// Maps the central 1 - 2*ACTIVE_MARGIN of the camera frame onto the full
// canvas. Cursor reaches canvas edges before the hand exits the camera,
// so users can draw at the boundaries without losing tracking.
const ACTIVE_MARGIN = 0.12;

let recognizer = null;
let drawingEngine = null;
let stabilizer = null;
let smoother = null;
let lastVideoTime = -1;
let videoStream = null;

// FPS tracking — counts inference frames only, not rAF ticks.
let fpsFrameCount = 0;
let lastFpsTime = performance.now();

// Edge-triggered one-shot gestures: fire once per gesture entry.
let prevConfirmedGesture = Gesture.UNKNOWN;

// Grace period state
let handLostFrames = 0;
let lastSmoothedLandmarks = null;
let lastGesture = Gesture.UNKNOWN;
let lastRawGesture = Gesture.UNKNOWN;

// Fingertip velocity tracking for extrapolation during DRAW
let tipVelocity = { x: 0, y: 0 };
let lastTipPos = null;

async function initRecognizer() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  );

  recognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function initWebcam() {
  const video = document.getElementById('webcam');
  videoStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: VIDEO_WIDTH },
      height: { ideal: VIDEO_HEIGHT },
      facingMode: 'user',
    },
    audio: false,
  });
  video.srcObject = videoStream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve(video);
    };
  });
}

function updateFps() {
  fpsFrameCount++;
  const now = performance.now();
  const elapsed = now - lastFpsTime;
  if (elapsed >= 1000) {
    const fps = Math.round((fpsFrameCount * 1000) / elapsed);
    document.getElementById('fps-label').textContent = `${fps} FPS`;
    fpsFrameCount = 0;
    lastFpsTime = now;
  }
}

function buildPalette() {
  const container = document.getElementById('color-palette');
  if (!container) return;
  container.innerHTML = '';
  drawingEngine.palette.forEach((color, i) => {
    const swatch = document.createElement('button');
    swatch.className = 'swatch';
    swatch.style.background = color;
    swatch.title = color;
    swatch.dataset.index = String(i);
    swatch.addEventListener('click', () => {
      drawingEngine.setColorIndex(i);
      updatePaletteSelection();
    });
    container.appendChild(swatch);
  });
  updatePaletteSelection();
}

function updatePaletteSelection() {
  const swatches = document.querySelectorAll('#color-palette .swatch');
  swatches.forEach((el, i) => {
    el.classList.toggle('active', i === drawingEngine.activeColorIndex);
  });
}

function updateBrushIndicator() {
  const el = document.getElementById('brush-size');
  if (!el) return;
  const pct = Math.round(drawingEngine.brushScale * 100);
  el.textContent = `${pct}%`;
}

function updateRawReadout(category, score) {
  const el = document.getElementById('raw-label');
  if (!el) return;
  if (!category || category === 'None') {
    el.textContent = '';
    return;
  }
  el.textContent = `${category} ${Math.round(score * 100)}%`;
}

function updateTipVelocity(smoothedLandmarks) {
  const tip = smoothedLandmarks[8];
  if (lastTipPos) {
    const rawVx = tip.x - lastTipPos.x;
    const rawVy = tip.y - lastTipPos.y;
    tipVelocity.x += VELOCITY_SMOOTHING * (rawVx - tipVelocity.x);
    tipVelocity.y += VELOCITY_SMOOTHING * (rawVy - tipVelocity.y);
  }
  lastTipPos = { x: tip.x, y: tip.y };
}

function getExtrapolatedTip(framesAhead) {
  if (!lastTipPos) return null;
  const decay = Math.pow(0.85, framesAhead);
  return {
    x: Math.max(0, Math.min(1, lastTipPos.x + tipVelocity.x * framesAhead * decay)),
    y: Math.max(0, Math.min(1, lastTipPos.y + tipVelocity.y * framesAhead * decay)),
  };
}

const gestureLabels = {
  [Gesture.IDLE]: 'IDLE',
  [Gesture.DRAW]: 'DRAW',
  [Gesture.ERASE]: 'ERASE',
  [Gesture.COLOR]: 'COLOR',
  [Gesture.UNDO]: 'UNDO',
  [Gesture.UNKNOWN]: '...',
};

/**
 * Act on the gesture using SMOOTHED landmarks for position (smooth drawing).
 * COLOR / UNDO are edge-triggered: they fire once on entry to the gesture and
 * not again until the user leaves and re-enters it.
 *
 * For DRAW we paint only when BOTH the stabilized AND the raw signals agree.
 * The stabilizer's stickiness keeps us in the DRAW state across brief
 * flickers (so the cursor stays put, prevPoint isn't reset), but pausing
 * paint the moment the raw signal diverges stops the end-of-stroke streak
 * caused by the fingertip traveling toward the palm during a curl.
 */
function handleGesture(gesture, rawGesture, smoothedLandmarks) {
  const gestureLabel = document.getElementById('gesture-label');
  gestureLabel.textContent = gestureLabels[gesture] || '...';

  const justEntered = gesture !== prevConfirmedGesture;

  switch (gesture) {
    case Gesture.DRAW:
      if (rawGesture === Gesture.DRAW) {
        drawingEngine.draw(smoothedLandmarks[8].x, smoothedLandmarks[8].y);
      }
      // else: hold drawing state without painting — resumes if raw returns
      // to DRAW; finalises via stopDrawing once stabilizer commits to exit.
      break;

    case Gesture.ERASE:
      drawingEngine.erase(smoothedLandmarks[9].x, smoothedLandmarks[9].y);
      break;

    case Gesture.COLOR:
      if (justEntered) {
        drawingEngine.cycleColor();
        updatePaletteSelection();
      }
      drawingEngine.stopDrawing();
      break;

    case Gesture.UNDO:
      if (justEntered) {
        drawingEngine.undo();
      }
      drawingEngine.stopDrawing();
      break;

    default:
      drawingEngine.stopDrawing();
      break;
  }

  prevConfirmedGesture = gesture;
}

function renderLoop() {
  const video = document.getElementById('webcam');
  const gestureLabel = document.getElementById('gesture-label');

  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    updateFps();

    const result = recognizer.recognizeForVideo(video, now);

    if (result.landmarks.length > 0) {
      const rawLandmarks = result.landmarks[0];
      const timestamp = now / 1000;

      // Top recognized gesture for this hand (sorted by confidence).
      const topCategory = result.gestures[0]?.[0];
      const categoryName = topCategory?.categoryName ?? '';
      const score = topCategory?.score ?? 0;
      updateRawReadout(categoryName, score);

      // SPLIT PIPELINE:
      // 1. Mediapipe-recognized gesture → app enum. If the model is unsure,
      //    fall back to a rotation-invariant geometric pointing test so
      //    sideways/upside-down pointing still works.
      let rawGesture = mapMediapipeGesture(categoryName, score);
      if (rawGesture === Gesture.UNKNOWN) {
        rawGesture = geometricGesture(rawLandmarks);
      }
      const stabilized = stabilizer.update(rawGesture);
      const gesture = stabilized.gesture;

      // 2. Smooth landmarks for drawing position only (no jitter).
      const smoothedLandmarks = smoother.smooth(rawLandmarks, timestamp);
      updateTipVelocity(smoothedLandmarks);

      handLostFrames = 0;
      lastSmoothedLandmarks = smoothedLandmarks;
      lastGesture = gesture;
      lastRawGesture = rawGesture;

      handleGesture(gesture, rawGesture, smoothedLandmarks);
      drawingEngine.drawOverlay(smoothedLandmarks, gesture, stabilized);
    } else {
      handLostFrames++;

      const wasDrawing = lastGesture === Gesture.DRAW;
      const graceLimit = wasDrawing ? GRACE_FRAMES_DRAW : GRACE_FRAMES_DEFAULT;

      if (handLostFrames <= graceLimit && lastSmoothedLandmarks) {
        if (wasDrawing) {
          const extTip = getExtrapolatedTip(handLostFrames);
          if (extTip) {
            drawingEngine.draw(extTip.x, extTip.y);
          }
        }
        drawingEngine.drawOverlay(lastSmoothedLandmarks, lastGesture, null);
      } else if (lastSmoothedLandmarks !== null) {
        gestureLabel.textContent = 'No hand';
        updateRawReadout('', 0);
        drawingEngine.stopDrawing();
        drawingEngine.drawOverlay(null, null, null);
        lastSmoothedLandmarks = null;
        lastGesture = Gesture.UNKNOWN;
        lastRawGesture = Gesture.UNKNOWN;
        prevConfirmedGesture = Gesture.UNKNOWN;
        lastTipPos = null;
        tipVelocity = { x: 0, y: 0 };
        smoother.reset();
        stabilizer.reset();
      }
    }
  }

  requestAnimationFrame(renderLoop);
}

function bindThemeToggle() {
  const btn = document.getElementById('themeToggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const root = document.documentElement;
    const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  });
}

function applyBgMode(mode) {
  const container = document.getElementById('video-container');
  const label = document.getElementById('bg-toggle-label');
  if (!container) return;
  if (mode === 'webcam') {
    container.classList.add('bg-webcam');
    if (label) label.textContent = 'Webcam';
  } else {
    container.classList.remove('bg-webcam');
    if (label) label.textContent = 'Blank';
  }
}

function bindBgToggle() {
  const btn = document.getElementById('bg-toggle');
  if (!btn) return;
  applyBgMode(localStorage.getItem('bgMode') || 'blank');
  btn.addEventListener('click', () => {
    const container = document.getElementById('video-container');
    const next = container.classList.contains('bg-webcam') ? 'blank' : 'webcam';
    applyBgMode(next);
    localStorage.setItem('bgMode', next);
  });
}

function bindKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    // Ignore if user is typing in some input (none currently, but future-proof).
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

    const ctrlOrMeta = e.ctrlKey || e.metaKey;

    if (ctrlOrMeta && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      drawingEngine.undo();
      return;
    }
    if (ctrlOrMeta && e.key.toLowerCase() === 's') {
      e.preventDefault();
      drawingEngine.saveAsImage();
      return;
    }

    switch (e.key) {
      case 'c':
      case 'C':
        drawingEngine.clearAll();
        break;
      case ',':
      case '<':
        drawingEngine.cycleColor(-1);
        updatePaletteSelection();
        break;
      case '.':
      case '>':
        drawingEngine.cycleColor(1);
        updatePaletteSelection();
        break;
      case '[':
        drawingEngine.bumpBrushScale(-BRUSH_STEP);
        updateBrushIndicator();
        break;
      case ']':
        drawingEngine.bumpBrushScale(BRUSH_STEP);
        updateBrushIndicator();
        break;
      default:
        return;
    }
  });
}

function bindUnloadCleanup() {
  window.addEventListener('beforeunload', () => {
    if (videoStream) {
      videoStream.getTracks().forEach((t) => t.stop());
    }
    if (recognizer && typeof recognizer.close === 'function') {
      try { recognizer.close(); } catch (_) { /* noop */ }
    }
  });
}

async function main() {
  const gestureLabel = document.getElementById('gesture-label');
  gestureLabel.textContent = 'Initializing...';

  try {
    const [video] = await Promise.all([initWebcam(), initRecognizer()]);

    // Use the camera's actual resolution — webcams sometimes ignore the
    // ideal size and deliver something else.
    const captureW = video.videoWidth || VIDEO_WIDTH;
    const captureH = video.videoHeight || VIDEO_HEIGHT;
    document.getElementById('video-container').style.aspectRatio = `${captureW} / ${captureH}`;

    const drawCanvas = document.getElementById('draw-canvas');
    const overlayCanvas = document.getElementById('overlay-canvas');
    drawingEngine = new DrawingEngine(drawCanvas, overlayCanvas, captureW, captureH);
    drawingEngine.setActiveMargin(ACTIVE_MARGIN);
    stabilizer = new GestureStabilizer();

    // One Euro Filter: minCutoff=0.8 (light smoothing), beta=0.5 (responsive)
    smoother = new LandmarkSmoother(21, { freq: 30, minCutoff: 0.8, beta: 0.5 });

    buildPalette();
    updateBrushIndicator();

    document.getElementById('clear-btn').addEventListener('click', () => {
      drawingEngine.clearAll();
    });

    document.getElementById('undo-btn').addEventListener('click', () => {
      drawingEngine.undo();
    });

    document.getElementById('save-btn').addEventListener('click', () => {
      drawingEngine.saveAsImage();
    });

    document.getElementById('brush-down').addEventListener('click', () => {
      drawingEngine.bumpBrushScale(-BRUSH_STEP);
      updateBrushIndicator();
    });

    document.getElementById('brush-up').addEventListener('click', () => {
      drawingEngine.bumpBrushScale(BRUSH_STEP);
      updateBrushIndicator();
    });

    bindKeyboardShortcuts();
    bindUnloadCleanup();
    bindThemeToggle();
    bindBgToggle();

    gestureLabel.textContent = 'Ready';
    renderLoop();
  } catch (err) {
    gestureLabel.textContent = `Error: ${err.message}`;
    console.error('Init failed:', err);
  }
}

main();
