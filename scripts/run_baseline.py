#!/usr/bin/env python3
"""
Stage 1 — Baseline rPPG using ME-rPPG ONNX model.

Uses MediaPipe FaceLandmarker (Tasks API) for face crop → ME-rPPG ONNX
for per-frame BVP → Welch PSD for HR estimation → compare vs UBFC-rPPG GT.
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# ── Config ──────────────────────────────────────────────────────────────
SUBJECTS = ["subject1", "subject3", "subject4"]
BASE_DIR = "data/DATASET_2"
OUT_DIR = "outputs/baseline"
MODEL_DIR = "ME-rPPG"
FACE_MODEL = os.path.join(MODEL_DIR, "face_landmarker.task")

HR_MIN = 40
HR_MAX = 180


# ── ME-rPPG helpers ────────────────────────────────────────────────────
def load_state(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model(path):
    import onnxruntime as ort

    sess = ort.InferenceSession(path)

    def run(img, state, dt=1 / 30):
        result = sess.run(
            None,
            {"arg_0.1": img[None, None], "onnx::Mul_37": [dt], **state},
        )
        bvp_val = result[0][0, 0]
        new_state = result[1:]
        return bvp_val, dict(zip(state, new_state))

    return run


def create_face_landmarker():
    """Create a FaceLandmarker using the Tasks API."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def crop_face(frame_bgr, landmarker):
    """Crop face to 36×36 RGB for ME-rPPG. Returns None if no face."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    h, w = frame_bgr.shape[:2]
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]

    xmin, xmax = int(min(xs)), int(max(xs))
    ymin, ymax = int(min(ys)), int(max(ys))

    pad_x = int(0.05 * (xmax - xmin))
    pad_y = int(0.05 * (ymax - ymin))
    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(w, xmax + pad_x)
    ymax = min(h, ymax + pad_y)

    roi = frame_bgr[ymin:ymax, xmin:xmax, ::-1]  # BGR → RGB
    roi = roi.astype("float32") / 255.0
    return cv2.resize(roi, (36, 36), interpolation=cv2.INTER_AREA)


def get_hr_welch(bvp_signal, sr=30, hr_min=40, hr_max=180):
    """Estimate HR (BPM) from BVP signal using Welch PSD."""
    if len(bvp_signal) < 64:
        return np.nan
    freqs, psd = welch(
        bvp_signal, fs=sr, nfft=int(1e5 / sr), nperseg=min(len(bvp_signal) - 1, 256)
    )
    mask = (freqs > hr_min / 60.0) & (freqs < hr_max / 60.0)
    if not np.any(mask):
        return np.nan
    peak_freq = freqs[mask][np.argmax(psd[mask])]
    return float(peak_freq * 60.0)


def load_gt_hr(gt_path):
    """Load UBFC-rPPG ground truth. Row 1 = per-frame HR."""
    data = np.loadtxt(gt_path)
    if data.ndim == 2 and data.shape[0] >= 2:
        return float(np.nanmean(data[1]))
    data = data.flatten()
    if np.nanmin(data) > 30 and np.nanmax(data) < 220:
        return float(np.nanmean(data))
    return np.nan


def fallback_crop(frame_bgr):
    """Center-crop fallback when face not detected."""
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    s = min(h, w) // 3
    roi = frame_bgr[cy - s : cy + s, cx - s : cx + s, ::-1]
    return cv2.resize(
        roi.astype("float32") / 255.0, (36, 36), interpolation=cv2.INTER_AREA
    )


# ── Main ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model_run = load_model(os.path.join(MODEL_DIR, "model.onnx"))
    init_state = load_state(os.path.join(MODEL_DIR, "state.json"))
    landmarker = create_face_landmarker()

    results = []

    for subj in SUBJECTS:
        print(f"\n{'='*50}")
        print(f"Processing {subj}...")
        print(f"{'='*50}")

        vid_path = os.path.join(BASE_DIR, subj, "vid.avi")
        gt_path = os.path.join(BASE_DIR, subj, "ground_truth.txt")

        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        state = {k: np.array(v, dtype=np.float32) for k, v in init_state.items()}
        dt = np.float32(1.0 / fps)

        bvp_signal = []
        frame_idx = 0
        skipped = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_img = crop_face(frame, landmarker)
            if face_img is None:
                skipped += 1
                face_img = fallback_crop(frame)

            bvp_val, state = model_run(face_img, state, dt)
            bvp_signal.append(float(bvp_val))
            frame_idx += 1

            if frame_idx % 200 == 0:
                print(f"  Frame {frame_idx}/{total_frames}...")

        cap.release()
        bvp_signal = np.array(bvp_signal)
        print(f"  Extracted {len(bvp_signal)} frames ({skipped} face-detect misses)")

        pred_hr = get_hr_welch(bvp_signal, sr=fps, hr_min=HR_MIN, hr_max=HR_MAX)
        gt_hr = load_gt_hr(gt_path)
        mae = abs(pred_hr - gt_hr)

        results.append(
            {"subject": subj, "pred_hr": pred_hr, "gt_hr": gt_hr, "mae": mae, "fps": fps}
        )

        print(f"  FPS: {fps:.2f}")
        print(f"  Pred HR: {pred_hr:.2f} bpm")
        print(f"  GT HR:   {gt_hr:.2f} bpm")
        print(f"  MAE:     {mae:.2f} bpm")

        # ── Plots ───────────────────────────────────────────────────
        t = np.arange(len(bvp_signal)) / fps
        freqs, psd = welch(
            bvp_signal, fs=fps, nfft=int(1e5 / fps),
            nperseg=min(len(bvp_signal) - 1, 256),
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(t, bvp_signal, linewidth=0.5)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("BVP")
        axes[0].set_title(f"{subj} — ME-rPPG BVP signal")

        axes[1].plot(freqs * 60, psd)
        axes[1].axvline(pred_hr, color="r", linestyle="--", label=f"Pred {pred_hr:.1f}")
        axes[1].axvline(gt_hr, color="g", linestyle="--", label=f"GT {gt_hr:.1f}")
        axes[1].set_xlim(HR_MIN, HR_MAX)
        axes[1].set_xlabel("Heart Rate (BPM)")
        axes[1].set_ylabel("PSD")
        axes[1].set_title(f"{subj} — PSD")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{subj}_baseline.png"), dpi=150)
        plt.close()

    # ── Summary ─────────────────────────────────────────────────────
    pred_hrs = np.array([r["pred_hr"] for r in results])
    gt_hrs = np.array([r["gt_hr"] for r in results])
    errors = np.abs(pred_hrs - gt_hrs)

    mae_mean = np.nanmean(errors)
    rmse = np.sqrt(np.nanmean(errors**2))
    valid = ~np.isnan(pred_hrs) & ~np.isnan(gt_hrs)
    if np.sum(valid) >= 2:
        r, p = pearsonr(pred_hrs[valid], gt_hrs[valid])
    else:
        r, p = np.nan, np.nan

    print(f"\n{'='*50}")
    print("BASELINE RESULTS (ME-rPPG)")
    print(f"{'='*50}")
    print(f"{'Subject':<12} {'Pred HR':>10} {'GT HR':>10} {'MAE':>10}")
    print("-" * 44)
    for res in results:
        print(f"{res['subject']:<12} {res['pred_hr']:>10.2f} {res['gt_hr']:>10.2f} {res['mae']:>10.2f}")
    print("-" * 44)
    print(f"{'Mean MAE':<12} {'':>10} {'':>10} {mae_mean:>10.2f} bpm")
    print(f"{'RMSE':<12} {'':>10} {'':>10} {rmse:>10.2f} bpm")
    print(f"{'Pearson r':<12} {'':>10} {'':>10} {r:>10.4f}")
    print(f"\nPlots saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
