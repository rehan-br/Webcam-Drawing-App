# hand-draw

A web application that turns webcam-tracked hand gestures into a drawing canvas.
Point with your index finger to draw, hold up an open palm to erase, and use
peace, fist, or thumbs-up gestures to control colour and undo.

All inference runs in the browser. No video leaves the device.

## Requirements

- A modern Chromium-based browser (Chrome, Edge) or Firefox 100 or later.
- A working webcam.
- Node.js 18 or later and npm for development.

## Running

```
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. The app will request webcam
access on first load. The first run also fetches the MediaPipe gesture
recognition model from Google's CDN (approximately 8 MB); subsequent loads
are cached by the browser.

For a production build:

```
npm run build
npm run preview
```

The build output is written to `dist/` and is fully static — any web server
that serves files over HTTPS (or `localhost`) will host it. A secure context
is required because the app uses `getUserMedia`.

## Gestures

| Pose         | Action                              |
| ------------ | ----------------------------------- |
| Index finger | Draw                                |
| Open palm    | Erase                               |
| Closed fist  | Idle (lift the pen)                 |
| Peace sign   | Cycle to the next colour            |
| Thumbs up    | Undo the last stroke                |

Colour and undo gestures fire once per entry — drop the gesture and form it
again to fire a second time. The colour palette in the top-right of the
canvas is also clickable.

## Keyboard shortcuts

```
Ctrl+Z    Undo
Ctrl+S    Save canvas as PNG
C         Clear canvas
, / .     Cycle colour
[ / ]     Decrease / increase brush size
```

## How it works

Hand pose recognition uses MediaPipe Tasks Vision' `GestureRecognizer` model
— a convolutional network trained by Google to classify hand poses from
camera frames. Each frame's recognised category is mapped to an app action.

When the model returns no confident result, a rotation-invariant geometric
check runs as a fallback, using PIP joint angles to recognise the same five
poses at any hand orientation. A short frame stabilizer requires a few
consecutive matching frames before committing a new gesture, with a longer
exit window for the drawing pose so brief misclassifications do not break
strokes.

The drawing canvas is mapped to a central region of the camera frame
(active-zone margin), letting the cursor reach the canvas edges without the
user's wrist exiting the camera view. Per-frame landmarks pass through a
One Euro Filter for jitter-free stroke positioning, while gesture
classification works on the unsmoothed signal for instant response.

## Project layout

```
index.html          Page shell
src/main.js         Entry point, render loop, UI wiring
src/gestures.js     Gesture vocabulary, mapping, stabilizer, fallback
src/drawing.js      Canvas drawing engine, undo, palette, brush
src/filters.js      One Euro Filter for landmark smoothing
src/style.css       Styles
```

## Privacy

The webcam stream is consumed by `getUserMedia` and processed entirely in
the browser tab. No frames are uploaded. The only network traffic after
first load is the page assets and, on first run, the MediaPipe model file
fetched from Google's CDN.

## License

MIT.
