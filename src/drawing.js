// Pixel-based constants are tuned at this reference width and scale
// linearly from there, so a 1280-wide canvas gets twice-thicker strokes
// than a 640-wide one.
const REFERENCE_WIDTH = 640;
const DRAW_LINE_WIDTH_MIN = 2;
const DRAW_LINE_WIDTH_MAX = 10;
const ERASE_RADIUS = 50;
const GLOW_BLUR = 4;
const SPEED_DIVISOR = 15;
const MAX_UNDO_STEPS = 12;
const SPEED_SMOOTHING = 0.3;
const BRUSH_SCALE_MIN = 0.4;
const BRUSH_SCALE_MAX = 3.0;

// Curated palette — saturated but not eye-burning neon. Each colour is
// distinct under the dark canvas without resorting to the standard
// "AI demo" green/cyan/magenta triad.
const COLOR_PALETTE = [
  '#e8e8e8', // chalk white
  '#ef4444', // red
  '#f59e0b', // amber
  '#facc15', // yellow
  '#22c55e', // green
  '#06b6d4', // cyan
  '#6366f1', // indigo
  '#ec4899', // pink
];

// Hand landmark connections for overlay visualization
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export class DrawingEngine {
  constructor(drawCanvas, overlayCanvas, width, height) {
    this.drawCanvas = drawCanvas;
    this.width = width;
    this.height = height;

    drawCanvas.width = width;
    drawCanvas.height = height;
    overlayCanvas.width = width;
    overlayCanvas.height = height;

    this.drawCtx = drawCanvas.getContext('2d');
    this.overlayCtx = overlayCanvas.getContext('2d');

    // Drawing state — no more smoothing here since landmarks are pre-smoothed
    this.prevPoint = null;
    this.isDrawing = false;
    this.smoothedSpeed = 0;

    // Color state
    this.colorIndex = 0;
    this.currentColor = COLOR_PALETTE[0];

    // Brush scale — applies to both draw stroke width and erase radius
    this.brushScale = 1.0;

    // Resolution-relative scale for pixel constants. At reference width
    // (640) this is 1.0; at 1280 it's 2.0, so strokes on a bigger canvas
    // have proportional visual weight.
    this.pxScale = width / REFERENCE_WIDTH;

    // Active-zone margin: maps central (1 - 2m) of the camera frame onto
    // the full canvas. Skeleton stays at raw coords so it overlays the
    // actual hand; cursor + stroke positions use the mapped coords so the
    // user can reach canvas edges without their hand exiting the camera.
    this.activeMargin = 0;

    // Undo history
    this.undoStack = [];
    this._eraseSavedUndo = false;

    this._applyBrushStyle();
  }

  _applyBrushStyle() {
    const ctx = this.drawCtx;
    ctx.strokeStyle = this.currentColor;
    ctx.lineWidth = DRAW_LINE_WIDTH_MAX * this.pxScale;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }

  toPixel(nx, ny) {
    // Raw camera-frame → canvas-pixel. Used for the skeleton overlay so
    // bones land where the actual hand is in the video.
    return { x: nx * this.width, y: ny * this.height };
  }

  toCanvasPixel(nx, ny) {
    // Active-zone aware mapping. Used for stroke positions and cursors so
    // the central region of the camera frame covers the whole canvas.
    const m = this.activeMargin;
    if (m <= 0) return this.toPixel(nx, ny);
    const u = 1 - 2 * m;
    const cx = Math.max(0, Math.min(1, (nx - m) / u));
    const cy = Math.max(0, Math.min(1, (ny - m) / u));
    return { x: cx * this.width, y: cy * this.height };
  }

  setActiveMargin(m) {
    this.activeMargin = Math.max(0, Math.min(0.45, m));
  }

  _saveUndoSnapshot() {
    const imageData = this.drawCtx.getImageData(0, 0, this.width, this.height);
    this.undoStack.push(imageData);
    if (this.undoStack.length > MAX_UNDO_STEPS) {
      this.undoStack.shift();
    }
  }

  _getSpeedLineWidth(point) {
    const maxW = DRAW_LINE_WIDTH_MAX * this.brushScale * this.pxScale;
    const minW = DRAW_LINE_WIDTH_MIN * this.brushScale * this.pxScale;

    if (!this.prevPoint) return maxW;

    const dx = point.x - this.prevPoint.x;
    const dy = point.y - this.prevPoint.y;
    const speed = Math.sqrt(dx * dx + dy * dy);

    this.smoothedSpeed += SPEED_SMOOTHING * (speed - this.smoothedSpeed);

    const t = Math.min(this.smoothedSpeed / (SPEED_DIVISOR * this.pxScale), 1);
    return maxW - t * (maxW - minW);
  }

  get eraseRadius() {
    return ERASE_RADIUS * this.brushScale * this.pxScale;
  }

  draw(landmarkX, landmarkY) {
    // Landmarks are already smoothed by One Euro Filter — use directly
    const point = this.toCanvasPixel(landmarkX, landmarkY);

    if (this.isDrawing && this.prevPoint) {
      const lineWidth = this._getSpeedLineWidth(point);
      const ctx = this.drawCtx;
      ctx.strokeStyle = this.currentColor;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(this.prevPoint.x, this.prevPoint.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
    } else if (!this.isDrawing) {
      this._saveUndoSnapshot();
    }

    this.prevPoint = point;
    this.isDrawing = true;
  }

  erase(landmarkX, landmarkY) {
    if (!this._eraseSavedUndo) {
      this._saveUndoSnapshot();
      this._eraseSavedUndo = true;
    }

    const { x, y } = this.toCanvasPixel(landmarkX, landmarkY);

    this.drawCtx.save();
    this.drawCtx.shadowBlur = 0;
    this.drawCtx.globalCompositeOperation = 'destination-out';
    this.drawCtx.beginPath();
    this.drawCtx.arc(x, y, this.eraseRadius, 0, Math.PI * 2);
    this.drawCtx.fill();
    this.drawCtx.restore();

    this.stopDrawing();
  }

  stopDrawing() {
    this.isDrawing = false;
    this.prevPoint = null;
    this.smoothedSpeed = 0;
    this._eraseSavedUndo = false;
  }

  cycleColor(direction = 1) {
    const n = COLOR_PALETTE.length;
    this.colorIndex = (this.colorIndex + direction + n) % n;
    this.currentColor = COLOR_PALETTE[this.colorIndex];
    this._applyBrushStyle();
  }

  setColorIndex(i) {
    if (i < 0 || i >= COLOR_PALETTE.length) return;
    this.colorIndex = i;
    this.currentColor = COLOR_PALETTE[this.colorIndex];
    this._applyBrushStyle();
  }

  setBrushScale(scale) {
    this.brushScale = Math.max(BRUSH_SCALE_MIN, Math.min(BRUSH_SCALE_MAX, scale));
  }

  bumpBrushScale(delta) {
    this.setBrushScale(this.brushScale + delta);
  }

  undo() {
    if (this.undoStack.length === 0) return false;
    const imageData = this.undoStack.pop();
    this.drawCtx.putImageData(imageData, 0, 0);
    this.stopDrawing();
    return true;
  }

  clearAll() {
    this._saveUndoSnapshot();
    this.drawCtx.clearRect(0, 0, this.width, this.height);
    this.stopDrawing();
  }

  saveAsImage() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = this.width;
    tempCanvas.height = this.height;
    const tempCtx = tempCanvas.getContext('2d');

    tempCtx.fillStyle = '#141414';
    tempCtx.fillRect(0, 0, this.width, this.height);
    tempCtx.drawImage(this.drawCanvas, 0, 0);

    const link = document.createElement('a');
    link.download = `hand-draw-${Date.now()}.png`;
    link.href = tempCanvas.toDataURL('image/png');
    link.click();
  }

  _drawActiveZoneOutline(ctx) {
    if (this.activeMargin <= 0) return;
    const m = this.activeMargin;
    const x = m * this.width;
    const y = m * this.height;
    const w = (1 - 2 * m) * this.width;
    const h = (1 - 2 * m) * this.height;
    const tick = 14 * this.pxScale;

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.lineWidth = 1;

    ctx.beginPath();
    // top-left
    ctx.moveTo(x, y + tick); ctx.lineTo(x, y); ctx.lineTo(x + tick, y);
    // top-right
    ctx.moveTo(x + w - tick, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + tick);
    // bottom-right
    ctx.moveTo(x + w, y + h - tick); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - tick, y + h);
    // bottom-left
    ctx.moveTo(x + tick, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - tick);
    ctx.stroke();

    ctx.restore();
  }

  drawOverlay(landmarks, gesture, stabilization) {
    const ctx = this.overlayCtx;
    ctx.clearRect(0, 0, this.width, this.height);

    this._drawActiveZoneOutline(ctx);

    if (!landmarks) return;

    // Draw connections
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    for (const [a, b] of CONNECTIONS) {
      const pa = this.toPixel(landmarks[a].x, landmarks[a].y);
      const pb = this.toPixel(landmarks[b].x, landmarks[b].y);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    // Draw landmark dots
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    for (const lm of landmarks) {
      const { x, y } = this.toPixel(lm.x, lm.y);
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Gesture-specific cursors — these mark the active draw/erase point,
    // so use the active-zone mapping (matches where strokes will land).
    if (gesture === 'DRAW') {
      const tip = this.toCanvasPixel(landmarks[8].x, landmarks[8].y);
      const cursorR = Math.max(6, DRAW_LINE_WIDTH_MAX * this.brushScale * this.pxScale * 0.9);
      ctx.strokeStyle = this.currentColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, cursorR, 0, Math.PI * 2);
      ctx.stroke();
    } else if (gesture === 'ERASE') {
      const palm = this.toCanvasPixel(landmarks[9].x, landmarks[9].y);
      const r = this.eraseRadius;
      ctx.strokeStyle = '#e74c3c';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(palm.x, palm.y, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(231, 76, 60, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(palm.x - r * 0.5, palm.y);
      ctx.lineTo(palm.x + r * 0.5, palm.y);
      ctx.moveTo(palm.x, palm.y - r * 0.5);
      ctx.lineTo(palm.x, palm.y + r * 0.5);
      ctx.stroke();
    } else if (gesture === 'COLOR') {
      const idx = this.toCanvasPixel(landmarks[8].x, landmarks[8].y);
      const mid = this.toCanvasPixel(landmarks[12].x, landmarks[12].y);
      const cx = (idx.x + mid.x) / 2;
      const cy = (idx.y + mid.y) / 2;
      ctx.fillStyle = this.currentColor;
      ctx.beginPath();
      ctx.arc(cx, cy, 10 * this.pxScale, 0, Math.PI * 2);
      ctx.fill();
    } else if (gesture === 'UNDO') {
      const palm = this.toCanvasPixel(landmarks[9].x, landmarks[9].y);
      const r = 14 * this.pxScale;
      ctx.strokeStyle = '#d4a73a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(palm.x, palm.y, r, Math.PI * 0.3, Math.PI * 1.7);
      ctx.stroke();
      const ax = palm.x + r * Math.cos(Math.PI * 0.3);
      const ay = palm.y + r * Math.sin(Math.PI * 0.3);
      ctx.beginPath();
      ctx.moveTo(ax - 6, ay - 4);
      ctx.lineTo(ax, ay);
      ctx.lineTo(ax + 2, ay - 8);
      ctx.stroke();
    }

    // Stabilization progress ring
    if (stabilization && !stabilization.isStable && stabilization.rawGesture !== 'UNKNOWN') {
      const tip = this.toCanvasPixel(landmarks[8].x, landmarks[8].y);
      const progress = stabilization.progress;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 16, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * progress);
      ctx.stroke();
    }
  }

  get palette() {
    return COLOR_PALETTE;
  }

  get activeColorIndex() {
    return this.colorIndex;
  }
}
