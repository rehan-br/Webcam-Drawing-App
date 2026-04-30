/**
 * Gesture vocabulary + frame-to-frame stabilizer.
 *
 * Detection itself is delegated to MediaPipe's GestureRecognizer task
 * (a CNN trained on hand poses). This file contains:
 *   - the app's gesture enum
 *   - the mapping from Mediapipe category names to that enum
 *   - the stabilizer that requires N consecutive frames before committing
 *
 * Mediapipe's default gesture model recognizes 7 categories:
 *   None, Closed_Fist, Open_Palm, Pointing_Up,
 *   Thumb_Up, Thumb_Down, Victory, ILoveYou
 *
 * We map five of them onto our app actions; the rest become UNKNOWN.
 */

export const Gesture = Object.freeze({
  IDLE: 'IDLE',
  DRAW: 'DRAW',
  ERASE: 'ERASE',
  COLOR: 'COLOR',
  UNDO: 'UNDO',
  UNKNOWN: 'UNKNOWN',
});

const MEDIAPIPE_TO_GESTURE = {
  Pointing_Up: Gesture.DRAW,
  Open_Palm: Gesture.ERASE,
  Closed_Fist: Gesture.IDLE,
  Victory: Gesture.COLOR,
  Thumb_Up: Gesture.UNDO,
};

const MIN_CONFIDENCE = 0.5;

/**
 * Map a Mediapipe gesture category to our enum, gated by confidence.
 * Returns UNKNOWN for categories we don't use, the "None" class, or
 * anything below the confidence threshold.
 */
export function mapMediapipeGesture(categoryName, score) {
  if (!categoryName || score < MIN_CONFIDENCE) return Gesture.UNKNOWN;
  return MEDIAPIPE_TO_GESTURE[categoryName] || Gesture.UNKNOWN;
}

// --- Rotation-invariant geometric fallback --------------------------------
//
// Mediapipe's pose classes are orientation-sensitive — sideways, tilted,
// or upside-down poses often fall below the confidence threshold. This
// fallback uses PIP-joint angles, which are rotation-invariant by
// construction, to recognise the same five poses at any hand orientation.
//
// Strict thresholds with no hysteresis: a finger has to be CLEARLY
// extended or curled to count. Frames in the dead zone fall through to
// UNKNOWN and the stabilizer holds the previous state. This avoids the
// flicker that doomed the original boolean classifier — and the fallback
// only ever runs when Mediapipe is UNKNOWN anyway, so confident model
// output always wins.

function angleDeg(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y, v1z = (a.z || 0) - (b.z || 0);
  const v2x = c.x - b.x, v2y = c.y - b.y, v2z = (c.z || 0) - (b.z || 0);
  const dot = v1x * v2x + v1y * v2y + v1z * v2z;
  const m1 = Math.hypot(v1x, v1y, v1z);
  const m2 = Math.hypot(v2x, v2y, v2z);
  if (m1 === 0 || m2 === 0) return 180;
  const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  return Math.acos(cos) * (180 / Math.PI);
}

const FINGER_PIP = {
  index: [5, 6, 7],
  middle: [9, 10, 11],
  ring: [13, 14, 15],
  pinky: [17, 18, 19],
};
const THUMB_IP = [2, 3, 4];

const FINGER_EXT_MIN = 155;
const FINGER_CURL_MAX = 130;
const THUMB_EXT_MIN = 150;

function isExt(L, name) {
  const [a, b, c] = FINGER_PIP[name];
  return angleDeg(L[a], L[b], L[c]) > FINGER_EXT_MIN;
}
function isCurl(L, name) {
  const [a, b, c] = FINGER_PIP[name];
  return angleDeg(L[a], L[b], L[c]) < FINGER_CURL_MAX;
}
function isThumbExt(L) {
  const [a, b, c] = THUMB_IP;
  return angleDeg(L[a], L[b], L[c]) > THUMB_EXT_MIN;
}

/**
 * Geometric pose classifier — rotation-invariant via PIP-joint angles.
 * Returns the matching app gesture or UNKNOWN if the pose is in the
 * dead zone for any finger.
 */
export function geometricGesture(landmarks) {
  if (!landmarks || landmarks.length < 21) return Gesture.UNKNOWN;

  const iE = isExt(landmarks, 'index');
  const mE = isExt(landmarks, 'middle');
  const rE = isExt(landmarks, 'ring');
  const pE = isExt(landmarks, 'pinky');

  const iC = isCurl(landmarks, 'index');
  const mC = isCurl(landmarks, 'middle');
  const rC = isCurl(landmarks, 'ring');
  const pC = isCurl(landmarks, 'pinky');

  const tE = isThumbExt(landmarks);

  if (iE && mC && rC && pC) return Gesture.DRAW;       // index only
  if (iE && mE && rE && pE) return Gesture.ERASE;      // open palm
  if (iE && mE && rC && pC) return Gesture.COLOR;      // peace
  if (tE && iC && mC && rC && pC) return Gesture.UNDO; // thumbs-up
  if (iC && mC && rC && pC) return Gesture.IDLE;       // fist

  return Gesture.UNKNOWN;
}

/**
 * Frame-to-frame gesture stabilizer with sticky DRAW.
 *
 * Even with a holistic CNN, raw classifications can flicker for a frame
 * or two during transitions. Requiring N consecutive matching frames
 * before committing absorbs that noise; the longer DRAW exit window
 * stops mid-stroke flickers from breaking the line.
 */
export class GestureStabilizer {
  constructor(enterFrames = 2, exitDrawFrames = 5) {
    this.enterFrames = enterFrames;
    this.exitDrawFrames = exitDrawFrames;
    this.currentRaw = Gesture.UNKNOWN;
    this.confirmedGesture = Gesture.UNKNOWN;
    this.frameCount = 0;
  }

  update(rawGesture) {
    if (rawGesture === this.currentRaw) {
      this.frameCount++;
    } else {
      this.currentRaw = rawGesture;
      this.frameCount = 1;
    }

    const required = (this.confirmedGesture === Gesture.DRAW)
      ? this.exitDrawFrames
      : this.enterFrames;

    if (this.frameCount >= required) {
      this.confirmedGesture = this.currentRaw;
    }

    const progress = Math.min(this.frameCount / required, 1);

    return {
      gesture: this.confirmedGesture,
      rawGesture: this.currentRaw,
      progress,
      isStable: this.frameCount >= required,
    };
  }

  reset() {
    this.currentRaw = Gesture.UNKNOWN;
    this.confirmedGesture = Gesture.UNKNOWN;
    this.frameCount = 0;
  }
}
