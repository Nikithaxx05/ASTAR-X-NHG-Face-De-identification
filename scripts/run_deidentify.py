#!/usr/bin/env python3
"""
Stage 2 — Geometric Face De-Identification.

Pipeline: MediaPipe FaceLandmarker (468 landmarks)
  → deterministic geometric perturbation
  → Delaunay piecewise-affine warp
  → convex-hull mask blending with feathered edges
  → write de-identified video + save before/after comparison images.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# ── Config ──────────────────────────────────────────────────────────────
SUBJECTS = ["subject1", "subject3", "subject4"]
BASE_DIR = "data/DATASET_2"
OUT_DIR = "outputs/deid_samples"
MODEL_DIR = "ME-rPPG"
FACE_MODEL = os.path.join(MODEL_DIR, "face_landmarker.task")

PERTURB_MAGNITUDE = 0.10

# MediaPipe face oval landmark indices (for mask)
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]


def create_face_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_face_landmarks(frame_bgr, landmarker):
    """Return (N, 2) pixel-coords array or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    h, w = frame_bgr.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
    return pts


def generate_perturbation(landmarks, seed, magnitude_frac=0.10):
    """Deterministic per-landmark perturbation. Radial + tangential."""
    rng = np.random.RandomState(seed)
    centroid = landmarks.mean(axis=0)
    directions = landmarks - centroid
    norms = np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-5, None)
    unit_dirs = directions / norms

    face_w = landmarks[:, 0].max() - landmarks[:, 0].min()
    max_shift = magnitude_frac * face_w

    magnitudes = rng.uniform(-max_shift, max_shift, size=(len(landmarks), 1))
    tangent_dirs = np.column_stack([-unit_dirs[:, 1], unit_dirs[:, 0]])
    tang_mag = rng.uniform(-max_shift * 0.5, max_shift * 0.5, size=(len(landmarks), 1))

    perturbation = unit_dirs * magnitudes + tangent_dirs * tang_mag

    contour_set = set(FACE_OVAL_IDX)
    for i in range(len(landmarks)):
        if i in contour_set:
            perturbation[i] *= 0.2

    return perturbation.astype(np.float32)


def get_delaunay_triangles(points, shape):
    """Compute Delaunay triangulation, return list of (i, j, k) index triples."""
    rect = (0, 0, shape[1], shape[0])
    subdiv = cv2.Subdiv2D(rect)

    # Insert points, building index map
    clamped = points.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, shape[1] - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, shape[0] - 1)

    for pt in clamped:
        subdiv.insert((float(pt[0]), float(pt[1])))

    triangles = []
    for t in subdiv.getTriangleList():
        pts_t = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]])
        # Check all points are inside rect
        if (pts_t[:, 0].min() < 0 or pts_t[:, 1].min() < 0 or
            pts_t[:, 0].max() >= shape[1] or pts_t[:, 1].max() >= shape[0]):
            continue
        # Find closest landmark index for each triangle vertex
        idx = []
        for p in pts_t:
            dists = np.sum((clamped - p) ** 2, axis=1)
            idx.append(int(np.argmin(dists)))
        if len(set(idx)) == 3:
            triangles.append(tuple(idx))

    return triangles


def warp_triangle(src_img, dst_img, src_tri, dst_tri):
    """Warp a single triangle from src to dst using affine transform."""
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))

    if r1[2] == 0 or r1[3] == 0 or r2[2] == 0 or r2[3] == 0:
        return

    src_tri_off = [(p[0] - r1[0], p[1] - r1[1]) for p in src_tri]
    dst_tri_off = [(p[0] - r2[0], p[1] - r2[1]) for p in dst_tri]

    src_crop = src_img[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    if src_crop.size == 0:
        return

    M = cv2.getAffineTransform(np.float32(src_tri_off), np.float32(dst_tri_off))
    warped = cv2.warpAffine(
        src_crop, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_off), (1, 1, 1), cv2.LINE_AA)

    x, y, w, h = r2
    if y + h > dst_img.shape[0] or x + w > dst_img.shape[1]:
        return
    region = dst_img[y : y + h, x : x + w].astype(np.float32)
    region = region * (1 - mask) + warped.astype(np.float32) * mask
    dst_img[y : y + h, x : x + w] = np.clip(region, 0, 255).astype(np.uint8)


def create_face_mask(shape, landmarks):
    """Soft face mask from face oval landmarks."""
    n_lms = len(landmarks)
    valid_idx = [i for i in FACE_OVAL_IDX if i < n_lms]
    contour_pts = landmarks[valid_idx].astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour_pts, 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    return mask.astype(np.float32) / 255.0


def deidentify_frame(frame, landmarks, perturbation, triangles):
    """Apply geometric de-identification."""
    dst_landmarks = landmarks + perturbation
    h, w = frame.shape[:2]
    dst_landmarks[:, 0] = np.clip(dst_landmarks[:, 0], 0, w - 1)
    dst_landmarks[:, 1] = np.clip(dst_landmarks[:, 1], 0, h - 1)

    warped = frame.copy()

    for i, j, k in triangles:
        src_tri = [(float(landmarks[ii][0]), float(landmarks[ii][1])) for ii in (i, j, k)]
        dst_tri = [(float(dst_landmarks[ii][0]), float(dst_landmarks[ii][1])) for ii in (i, j, k)]
        warp_triangle(frame, warped, src_tri, dst_tri)

    mask = create_face_mask(frame.shape, dst_landmarks)
    mask_3c = mask[:, :, np.newaxis]
    result = frame.astype(np.float32) * (1 - mask_3c) + warped.astype(np.float32) * mask_3c
    return np.clip(result, 0, 255).astype(np.uint8)


def _save_comparison(original, deidentified, subj):
    """Save side-by-side before/after."""
    h, w = original.shape[:2]
    canvas = np.full((h + 40, w * 2 + 20, 3), 30, dtype=np.uint8)
    canvas[30 : 30 + h, 0:w] = original
    canvas[30 : 30 + h, w + 20 : 2 * w + 20] = deidentified

    cv2.putText(canvas, "ORIGINAL", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, "DE-IDENTIFIED", (w + 30, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imwrite(os.path.join(OUT_DIR, f"{subj}_comparison.png"), canvas)

    # Face close-up crop
    cy, cx = h // 2, w // 2
    s = min(h, w) // 3
    fo = original[max(0, cy - s) : cy + s, max(0, cx - s) : cx + s]
    fd = deidentified[max(0, cy - s) : cy + s, max(0, cx - s) : cx + s]
    if fo.shape == fd.shape and fo.size > 0:
        face_canvas = np.zeros((fo.shape[0], fo.shape[1] * 2 + 10, 3), dtype=np.uint8)
        face_canvas[:, : fo.shape[1]] = fo
        face_canvas[:, fo.shape[1] + 10 :] = fd
        cv2.imwrite(os.path.join(OUT_DIR, f"{subj}_face_closeup.png"), face_canvas)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    landmarker = create_face_landmarker()

    for subj in SUBJECTS:
        print(f"\n{'='*50}")
        print(f"De-identifying {subj}...")
        print(f"{'='*50}")

        vid_path = os.path.join(BASE_DIR, subj, "vid.avi")
        out_vid_path = os.path.join(BASE_DIR, subj, "vid_deid.avi")

        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_vid_path, fourcc, fps, (w, h))

        perturbation = None
        triangles = None
        frame_idx = 0
        sample_saved = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = get_face_landmarks(frame, landmarker)

            if landmarks is not None:
                if perturbation is None:
                    seed = int(np.sum(landmarks[:10].astype(np.int32))) % (2**31)
                    perturbation = generate_perturbation(landmarks, seed, PERTURB_MAGNITUDE)
                    triangles = get_delaunay_triangles(landmarks, frame.shape)

                result = deidentify_frame(frame, landmarks, perturbation, triangles)

                if not sample_saved:
                    _save_comparison(frame, result, subj)
                    sample_saved = True

                writer.write(result)
            else:
                writer.write(frame)

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"  Frame {frame_idx}/{total}...")

        cap.release()
        writer.release()
        print(f"  Saved: {out_vid_path} ({frame_idx} frames)")

    print(f"\nComparison images saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
