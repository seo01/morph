#!/usr/bin/env python3
"""Build a chronological face-morph video from a folder of photos.

Pipeline:
  1. Detect faces in every image with OpenCV's YuNet DNN detector.
  2. Keep only images containing exactly one camera-facing face (nose tip
     roughly centered between the eyes).
  3. Pick at most one image per calendar day (EXIF DateTimeOriginal, falling
     back to the file's creation timestamp), preferring the most confident
     face detection.
  4. Identity check: embed each selected face with SFace and compare it to
     its temporal neighbors. The subject's face drifts gradually with age,
     so a different person (or an unusable shot) scores low against nearby
     days — those fall back to the day's next-best photo, or drop the day.
  5. Align each selected face so the eyes land on fixed positions (similarity
     transform: rotation + scale + translation), placing the face in the
     upper center of the frame at a consistent size.
  6. Write the aligned crops to disk and render a crossfade morph video.

Usage:
  python morph.py AlecDynamic                # full pipeline
  python morph.py AlecDynamic --video-only   # rebuild video from existing crops
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306

# Where the face lands in the output frame ("upper center"):
# eyes on a horizontal line at EYE_LINE_Y of frame height, centered,
# interocular distance a fixed fraction of frame width.
EYE_LINE_Y = 0.40
INTEROCULAR_FRAC = 0.20
DETECT_MAX_SIDE = 1024


def find_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_date(path: Path) -> datetime:
    """EXIF capture date if present, else filesystem creation time."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get(EXIF_DATETIME_ORIGINAL) or exif.get_ifd(0x8769).get(
                EXIF_DATETIME_ORIGINAL
            ) or exif.get(EXIF_DATETIME)
            if raw:
                return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    stat = path.stat()
    created = getattr(stat, "st_birthtime", stat.st_mtime)
    return datetime.fromtimestamp(created)


def make_detector(model_path: Path, score_threshold: float) -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        str(model_path), "", (320, 320),
        score_threshold=score_threshold, nms_threshold=0.3, top_k=50,
    )


def detect_faces(detector: cv2.FaceDetectorYN, image: np.ndarray) -> np.ndarray:
    """Run YuNet on a possibly-downscaled copy; return faces in original coords.

    Each row: x, y, w, h, landmarks (right eye, left eye, nose,
    mouth corners as x,y pairs), confidence.
    """
    h, w = image.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    if scale < 1.0:
        small = cv2.resize(image, (round(w * scale), round(h * scale)))
    else:
        small = image
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    faces = faces.copy()
    faces[:, :14] /= scale
    return faces


def frontality(face: np.ndarray) -> float:
    """How far the nose tip sits off-center between the eyes, as a fraction
    of interocular distance. ~0 for a camera-facing face, grows with yaw."""
    right_eye = face[4:6]
    left_eye = face[6:8]
    nose = face[8:10]
    eye_axis = left_eye - right_eye
    eye_dist = float(np.linalg.norm(eye_axis))
    if eye_dist < 1e-6:
        return 1.0
    mid = (right_eye + left_eye) / 2.0
    offset = float(np.dot(nose - mid, eye_axis / eye_dist))
    return abs(offset) / eye_dist


def align_face(image: np.ndarray, face: np.ndarray,
               out_w: int, out_h: int) -> np.ndarray:
    """Similarity-transform the image so the eyes hit fixed target points."""
    right_eye = face[4:6]   # subject's right eye (image left)
    left_eye = face[6:8]
    eye_dx, eye_dy = left_eye - right_eye
    eye_dist = math.hypot(eye_dx, eye_dy)
    angle = math.degrees(math.atan2(eye_dy, eye_dx))
    scale = (INTEROCULAR_FRAC * out_w) / max(eye_dist, 1e-6)

    eyes_mid = (right_eye + left_eye) / 2.0
    m = cv2.getRotationMatrix2D(tuple(eyes_mid), angle, scale)
    m[0, 2] += out_w / 2.0 - eyes_mid[0]
    m[1, 2] += out_h * EYE_LINE_Y - eyes_mid[1]
    return cv2.warpAffine(
        image, m, (out_w, out_h),
        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE,
    )


def boundary_points(w: int, h: int) -> np.ndarray:
    """Frame corners and edge midpoints, appended to the face landmarks so
    the warp mesh covers the whole frame (background stays ~static)."""
    return np.array([
        (0, 0), (w // 2, 0), (w - 1, 0),
        (0, h // 2), (w - 1, h // 2),
        (0, h - 1), (w // 2, h - 1), (w - 1, h - 1),
    ], dtype=np.float32)


def crop_landmarks(frames: list[Path], args: argparse.Namespace) -> list:
    """68 facial landmarks + 8 frame-boundary points per crop (None if the
    landmark fit fails)."""
    detector = make_detector(Path(args.model), 0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(args.landmark_model)
    # Fallback face box derived from the fixed alignment geometry.
    eye_dist = INTEROCULAR_FRAC * args.width
    box = 2.4 * eye_dist
    canonical = (round(args.width / 2 - box / 2),
                 round(EYE_LINE_Y * args.height - 0.4 * box),
                 round(box), round(box))
    results = []
    for path in frames:
        image = cv2.imread(str(path))
        faces = detect_faces(detector, image)
        if len(faces) > 0:
            best = faces[np.argmax(faces[:, 14])]
            rect = best[:4].astype(int)
        else:
            rect = np.array(canonical)
        ok, landmarks = facemark.fit(image, np.array([rect]))
        if not ok:
            results.append(None)
            continue
        points = landmarks[0].reshape(-1, 2).astype(np.float32)
        points[:, 0] = points[:, 0].clip(0, args.width - 1)
        points[:, 1] = points[:, 1].clip(0, args.height - 1)
        results.append(np.vstack([points, boundary_points(args.width,
                                                          args.height)]))
    return results


def delaunay_triangles(points: np.ndarray, w: int, h: int) -> list:
    """Delaunay triangulation as index triples into `points`."""
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    index_of = {}
    for i, p in enumerate(points):
        key = (round(float(p[0]), 1), round(float(p[1]), 1))
        index_of[key] = i
        subdiv.insert((float(p[0]), float(p[1])))
    triangles = []
    for t in subdiv.getTriangleList():
        keys = [(round(float(t[k]), 1), round(float(t[k + 1]), 1))
                for k in (0, 2, 4)]
        if all(k in index_of for k in keys):
            triangles.append(tuple(index_of[k] for k in keys))
    return triangles


def warp_mesh(image: np.ndarray, src: np.ndarray, dst: np.ndarray,
              triangles: list) -> np.ndarray:
    """Piecewise-affine warp of `image` moving `src` points to `dst`."""
    out = np.zeros_like(image)
    h, w = image.shape[:2]
    for tri in triangles:
        t_src = src[list(tri)]
        t_dst = dst[list(tri)]
        r1 = cv2.boundingRect(t_src)
        r2 = cv2.boundingRect(t_dst)
        if min(r1[2], r1[3], r2[2], r2[3]) <= 0:
            continue
        m = cv2.getAffineTransform(t_src - np.float32(r1[:2]),
                                   t_dst - np.float32(r2[:2]))
        patch = image[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        warped = cv2.warpAffine(patch, m, (r2[2], r2[3]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(t_dst - np.float32(r2[:2]))
                           .astype(np.int32), 255)
        roi = out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
        roi[mask > 0] = warped[mask > 0]
    return out


def embed_face(recognizer: cv2.FaceRecognizerSF, image: np.ndarray,
               face: np.ndarray) -> np.ndarray:
    feature = recognizer.feature(recognizer.alignCrop(image, face)).flatten()
    return feature / np.linalg.norm(feature)


def identity_filter(candidates: dict[str, list[tuple[float, Path, np.ndarray]]],
                    args: argparse.Namespace) -> dict[str, int]:
    """Pick a candidate index per day such that every chosen face resembles
    its temporal neighbors; days with no acceptable candidate are dropped.

    Returns day -> index into candidates[day].
    """
    recognizer = cv2.FaceRecognizerSF.create(args.recognition_model, "")
    chosen = {day: 0 for day in candidates}
    cache: dict[tuple[str, int], np.ndarray] = {}

    def embedding(day: str, idx: int) -> np.ndarray:
        if (day, idx) not in cache:
            _, path, face = candidates[day][idx]
            cache[(day, idx)] = embed_face(
                recognizer, cv2.imread(str(path)), face)
        return cache[(day, idx)]

    # Backward elimination: rescore everything, reject only the worst
    # offender, repeat. Rejecting in bulk is order-sensitive — removing one
    # outlier from a window inflates the scores of the ones still in it.
    neighbors = 8
    while len(chosen) > 2 * neighbors:
        days = sorted(chosen)
        scores = {}
        for i, day in enumerate(days):
            window = [d for d in days[max(0, i - neighbors):i + neighbors + 1]
                      if d != day]
            emb = embedding(day, chosen[day])
            scores[day] = float(np.median(
                [emb @ embedding(d, chosen[d]) for d in window]))
        worst = min(scores, key=scores.get)
        if scores[worst] >= args.identity_threshold:
            break
        _, path, _ = candidates[worst][chosen[worst]]
        if chosen[worst] + 1 < len(candidates[worst]):
            chosen[worst] += 1
            print(f"  [identity] {worst}: {path.name} rejected "
                  f"(sim {scores[worst]:.2f}), trying next-best photo")
        else:
            del chosen[worst]
            print(f"  [identity] {worst}: {path.name} rejected "
                  f"(sim {scores[worst]:.2f}), day dropped")
    return chosen


def select_and_crop(args: argparse.Namespace, crops_dir: Path) -> None:
    input_dir = Path(args.input_dir)
    images = find_images(input_dir)
    if not images:
        sys.exit(f"No images found in {input_dir}")
    print(f"Scanning {len(images)} images in {input_dir}")

    detector = make_detector(Path(args.model), args.score_threshold)

    # per-day candidates, best-first: date -> [(confidence, path, face row)]
    candidates: dict[str, list[tuple[float, Path, np.ndarray]]] = {}
    n_single = 0
    n_frontal = 0
    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  [skip] unreadable: {path.name}")
            continue
        faces = detect_faces(detector, image)
        if len(faces) != 1:
            continue
        n_single += 1
        if frontality(faces[0]) > args.frontal_threshold:
            continue
        n_frontal += 1
        confidence = float(faces[0][14])
        day = image_date(path).strftime("%Y-%m-%d")
        candidates.setdefault(day, []).append((confidence, path, faces[0]))
        if i % 200 == 0:
            print(f"  {i}/{len(images)} scanned, {n_single} single-face, "
                  f"{n_frontal} frontal, {len(candidates)} days")

    print(f"Done scanning: {n_single} single-face images, "
          f"{n_frontal} facing the camera, across {len(candidates)} "
          f"distinct days")

    for day_candidates in candidates.values():
        day_candidates.sort(key=lambda c: c[0], reverse=True)
    print("Checking subject identity against temporal neighbors...")
    chosen = identity_filter(candidates, args)
    print(f"Identity filter kept {len(chosen)} of {len(candidates)} days")

    crops_dir.mkdir(parents=True, exist_ok=True)
    for old in crops_dir.glob("*.jpg"):
        old.unlink()
    manifest = {}
    for day in sorted(chosen):
        confidence, path, face = candidates[day][chosen[day]]
        image = cv2.imread(str(path))
        aligned = align_face(image, face, args.width, args.height)
        out_path = crops_dir / f"{day}.jpg"
        cv2.imwrite(str(out_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest[day] = {"source": path.name, "confidence": round(confidence, 4),
                         "frontality": round(frontality(face), 3)}
    (crops_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} aligned crops to {crops_dir}")


def build_video(args: argparse.Namespace, crops_dir: Path) -> None:
    frames = sorted(crops_dir.glob("*.jpg"))
    if len(frames) < 2:
        sys.exit(f"Need at least 2 crops in {crops_dir} to build a video")

    if args.duration:
        # Fit the video to a target length: one hold frame per image, the
        # rest of the budget goes to the transition.
        per_image = max(2, round(args.duration * args.fps / len(frames)))
        args.hold_frames = 1
        args.transition_frames = per_image - args.hold_frames

    landmarks = [None] * len(frames)
    triangles = []
    if args.morph == "warp":
        print(f"Fitting facial landmarks on {len(frames)} crops...")
        landmarks = crop_landmarks(frames, args)
        n_ok = sum(lm is not None for lm in landmarks)
        print(f"  landmarks fitted on {n_ok}/{len(frames)} crops "
              f"(pairs without landmarks fall back to crossfade)")
        mean_shape = np.mean([lm for lm in landmarks if lm is not None],
                             axis=0).astype(np.float32)
        triangles = delaunay_triangles(mean_shape, args.width, args.height)

    out_path = Path(args.output)
    writer = None
    for fourcc in ("avc1", "mp4v"):  # prefer H.264; mp4v as fallback
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*fourcc),
            args.fps, (args.width, args.height),
        )
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        sys.exit("Could not open video writer (no usable mp4 codec)")

    print(f"Rendering {len(frames)} frames -> {out_path} "
          f"({args.fps} fps, {args.transition_frames} transition "
          f"+ {args.hold_frames} hold frames per image)")
    previous = cv2.imread(str(frames[0])).astype(np.float32)
    for _ in range(args.hold_frames):
        writer.write(previous.astype(np.uint8))
    for i, frame_path in enumerate(frames[1:], start=1):
        current = cv2.imread(str(frame_path)).astype(np.float32)
        pts_a, pts_b = landmarks[i - 1], landmarks[i]
        for t in range(1, args.transition_frames + 1):
            alpha = t / args.transition_frames
            if pts_a is not None and pts_b is not None:
                # True morph: warp both faces to the intermediate shape,
                # then cross-dissolve.
                pts_mid = ((1.0 - alpha) * pts_a + alpha * pts_b)
                warp_a = warp_mesh(previous, pts_a, pts_mid, triangles)
                warp_b = warp_mesh(current, pts_b, pts_mid, triangles)
                blend = cv2.addWeighted(warp_a, 1.0 - alpha,
                                        warp_b, alpha, 0.0)
            else:
                blend = cv2.addWeighted(previous, 1.0 - alpha,
                                        current, alpha, 0.0)
            writer.write(blend.astype(np.uint8))
        for _ in range(args.hold_frames):
            writer.write(current.astype(np.uint8))
        previous = current
        if i % 50 == 0:
            print(f"  {i}/{len(frames) - 1} transitions rendered")
    writer.release()

    total = args.hold_frames + (len(frames) - 1) * (
        args.transition_frames + args.hold_frames)
    print(f"Wrote {out_path} ({total} frames, "
          f"{total / args.fps:.1f}s at {args.fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", help="Folder of source photos")
    parser.add_argument("--output", default="alec_morph.mp4",
                        help="Output video path (default: %(default)s)")
    parser.add_argument("--crops-dir", default="crops",
                        help="Folder for aligned face crops (default: %(default)s)")
    parser.add_argument("--model", default="face_detection_yunet_2023mar.onnx",
                        help="YuNet ONNX model path (default: %(default)s)")
    parser.add_argument("--width", type=int, default=810,
                        help="Output frame width (default: %(default)s)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Output frame height (default: %(default)s)")
    parser.add_argument("--score-threshold", type=float, default=0.8,
                        help="Min face confidence (default: %(default)s)")
    parser.add_argument("--morph", choices=["warp", "fade"], default="warp",
                        help="Transition style: landmark warp morph or plain "
                             "crossfade (default: %(default)s)")
    parser.add_argument("--landmark-model", default="lbfmodel.yaml",
                        help="Facemark LBF model path (default: %(default)s)")
    parser.add_argument("--recognition-model",
                        default="face_recognition_sface_2021dec.onnx",
                        help="SFace ONNX model path (default: %(default)s)")
    parser.add_argument("--identity-threshold", type=float, default=0.36,
                        help="Min median cosine similarity to temporal "
                             "neighbors (default: %(default)s)")
    parser.add_argument("--frontal-threshold", type=float, default=0.15,
                        help="Max nose offset between the eyes as a fraction of "
                             "interocular distance; lower = stricter camera-facing "
                             "filter (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Target video length in seconds; overrides "
                             "--transition-frames/--hold-frames")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frame rate (default: %(default)s)")
    parser.add_argument("--transition-frames", type=int, default=8,
                        help="Crossfade frames between images (default: %(default)s)")
    parser.add_argument("--hold-frames", type=int, default=3,
                        help="Still frames per image (default: %(default)s)")
    parser.add_argument("--video-only", action="store_true",
                        help="Skip detection/cropping; build video from --crops-dir")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    if not args.video_only:
        select_and_crop(args, crops_dir)
    build_video(args, crops_dir)


if __name__ == "__main__":
    main()
