/**
 * One Euro Filter — adaptive low-pass filter that smooths jitter
 * while preserving responsiveness during fast movement.
 *
 * When the signal moves slowly → heavy smoothing (kills jitter)
 * When the signal moves fast   → light smoothing (low latency)
 *
 * https://cristal.univ-lille.fr/~casiez/1euro/
 */

class LowPassFilter {
  constructor(alpha, initial) {
    this.y = initial;
    this.s = initial;
    this.a = alpha;
    this.initialized = false;
  }

  reset(value) {
    this.y = value;
    this.s = value;
    this.initialized = true;
  }

  filter(value, alpha) {
    if (!this.initialized) {
      this.reset(value);
      return value;
    }
    this.a = alpha;
    this.s = alpha * value + (1 - alpha) * this.s;
    return this.s;
  }

  last() {
    return this.s;
  }
}

class OneEuroFilter {
  /**
   * @param {number} freq     - Expected signal frequency (Hz). Use your FPS.
   * @param {number} minCutoff - Minimum cutoff frequency. Lower = smoother at rest.
   * @param {number} beta      - Speed coefficient. Higher = more responsive.
   * @param {number} dCutoff   - Cutoff for derivative filter.
   */
  constructor(freq = 30, minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = new LowPassFilter(this._alpha(minCutoff), 0);
    this.dx = new LowPassFilter(this._alpha(dCutoff), 0);
    this.lastTime = null;
  }

  _alpha(cutoff) {
    const te = 1.0 / this.freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }

  reset() {
    this.x = new LowPassFilter(this._alpha(this.minCutoff), 0);
    this.dx = new LowPassFilter(this._alpha(this.dCutoff), 0);
    this.lastTime = null;
  }

  filter(value, timestamp) {
    if (this.lastTime != null && timestamp != null) {
      const dt = timestamp - this.lastTime;
      if (dt > 0) this.freq = 1.0 / dt;
    }
    this.lastTime = timestamp;

    const prevX = this.x.last();
    const dx = this.x.initialized ? (value - prevX) * this.freq : 0;
    const edx = this.dx.filter(dx, this._alpha(this.dCutoff));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.x.filter(value, this._alpha(cutoff));
  }
}

/**
 * Smooths all 21 MediaPipe hand landmarks using per-coordinate One Euro Filters.
 * Dramatically reduces jitter while keeping fast movement responsive.
 */
export class LandmarkSmoother {
  /**
   * @param {number} numLandmarks - Number of landmarks (21 for hand)
   * @param {object} opts         - One Euro Filter params
   */
  constructor(numLandmarks = 21, opts = {}) {
    const freq = opts.freq || 30;
    const minCutoff = opts.minCutoff || 1.5;
    const beta = opts.beta || 0.01;
    const dCutoff = opts.dCutoff || 1.0;

    this.filters = [];
    for (let i = 0; i < numLandmarks; i++) {
      this.filters.push({
        x: new OneEuroFilter(freq, minCutoff, beta, dCutoff),
        y: new OneEuroFilter(freq, minCutoff, beta, dCutoff),
      });
    }
    this.numLandmarks = numLandmarks;
  }

  /**
   * Smooth a frame of landmarks. Returns a new array with smoothed values.
   * @param {Array} landmarks - Raw landmarks from MediaPipe [{x, y, z}, ...]
   * @param {number} timestamp - Current time in seconds
   * @returns {Array} Smoothed landmarks
   */
  smooth(landmarks, timestamp) {
    const result = [];
    for (let i = 0; i < this.numLandmarks; i++) {
      const raw = landmarks[i];
      const f = this.filters[i];
      result.push({
        x: f.x.filter(raw.x, timestamp),
        y: f.y.filter(raw.y, timestamp),
        z: raw.z, // z is less important, pass through
      });
    }
    return result;
  }

  reset() {
    for (const f of this.filters) {
      f.x.reset();
      f.y.reset();
    }
  }
}
