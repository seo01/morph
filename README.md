# morph

Builds a chronological face-morph video from a folder of family photos —
one face per day, aligned and warped smoothly from newborn to now.

⚠️ This project was vibe coded with Claude. It works on my photo library;
read it before trusting it with yours.

## How it works

1. Detects faces in every image (OpenCV YuNet) and keeps photos containing
   exactly one camera-facing face.
2. Picks at most one photo per calendar day (EXIF date, falling back to file
   creation time), preferring the most confident detection.
3. Filters out other people: each day's face is embedded with SFace and
   compared to its temporal neighbours — the subject's face drifts gradually
   with age, so a different person scores sharply lower and is rejected
   (falling back to that day's next-best photo).
4. Aligns every face with a similarity transform so the eyes land on fixed
   positions (upper center, consistent size).
5. Renders a morph video: 68 facial landmarks per crop (Facemark LBF),
   piecewise-affine warp to intermediate shapes, cross-dissolve.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Models (YuNet detector, SFace recogniser, LBF landmarks)
curl -sL -O https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sL -O https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
curl -sL -O https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml
```

## Usage

```bash
# Full pipeline: scan photos, select, crop, render alec_morph.mp4
.venv/bin/python morph.py AlecDynamic --duration 90

# Re-render from existing crops (fast — skips detection/selection),
# e.g. after deleting unwanted dates from crops/
.venv/bin/python morph.py AlecDynamic --video-only --duration 60

# Plain crossfade instead of landmark warping
.venv/bin/python morph.py AlecDynamic --morph fade
```

Useful knobs (see `--help` for all): `--width/--height` (frame size),
`--fps`, `--duration` (target length in seconds), `--frontal-threshold`
(how strictly faces must face the camera), `--identity-threshold`
(how strictly faces must match their neighbours in time).

Outputs: aligned crops in `crops/` (one `YYYY-MM-DD.jpg` per day, with a
`manifest.json` recording source file and detection confidence per day)
and the video at `--output` (default `alec_morph.mp4`).
