#!/usr/bin/env python3
"""
Stage 3 — Validation: run ME-rPPG on both original and de-identified
videos, compare accuracy metrics.
"""

import os
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
import onnxruntime as ort

# ── Config ──────────────────────────────────────────────────────────────
SUBJECTS = ["subject1", "subject3", "subject4"]
BASE_DIR = "data/DATASET_2"
OUT_DIR = "outputs/validation"
MODEL_DIR = "ME-rPPG"
FACE_MODEL = os.path.join(MODEL_DIR, "face_landmarker.task")

HR_MIN = 40
HR_MAX = 180


# ── Helpers ─────────────────────────────────────────────────────────────
def load_state(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model(path):
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
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def crop_face(frame_bgr, landmarker):
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
    roi = frame_bgr[ymin:ymax, xmin:xmax, ::-1]
    roi = roi.astype("float32") / 255.0
    return cv2.resize(roi, (36, 36), interpolation=cv2.INTER_AREA)


def fallback_crop(frame_bgr):
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    s = min(h, w) // 3
    roi = frame_bgr[cy - s : cy + s, cx - s : cx + s, ::-1]
    return cv2.resize(
        roi.astype("float32") / 255.0, (36, 36), interpolation=cv2.INTER_AREA
    )


def get_hr_welch(bvp_signal, sr=30, hr_min=40, hr_max=180):
    if len(bvp_signal) < 64:
        return np.nan
    freqs, psd = welch(
        bvp_signal, fs=sr, nfft=int(1e5 / sr), nperseg=min(len(bvp_signal) - 1, 256)
    )
    mask = (freqs > hr_min / 60.0) & (freqs < hr_max / 60.0)
    if not np.any(mask):
        return np.nan
    return float(freqs[mask][np.argmax(psd[mask])] * 60.0)


def load_gt_hr(gt_path):
    data = np.loadtxt(gt_path)
    if data.ndim == 2 and data.shape[0] >= 2:
        return float(np.nanmean(data[1]))
    data = data.flatten()
    if np.nanmin(data) > 30 and np.nanmax(data) < 220:
        return float(np.nanmean(data))
    return np.nan


def run_rppg_on_video(vid_path, model_run, init_state, landmarker, label=""):
    """Run ME-rPPG on a video, return (bvp_signal, fps)."""
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    state = {k: np.array(v, dtype=np.float32) for k, v in init_state.items()}
    dt = np.float32(1.0 / fps)
    bvp = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_img = crop_face(frame, landmarker)
        if face_img is None:
            face_img = fallback_crop(frame)
        val, state = model_run(face_img, state, dt)
        bvp.append(float(val))
        idx += 1
        if idx % 300 == 0:
            print(f"    {label} frame {idx}/{total}...")

    cap.release()
    return np.array(bvp), fps


# ── Main ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model_run = load_model(os.path.join(MODEL_DIR, "model.onnx"))
    init_state = load_state(os.path.join(MODEL_DIR, "state.json"))
    landmarker = create_face_landmarker()

    all_results = []

    for subj in SUBJECTS:
        print(f"\n{'='*55}")
        print(f"  {subj}")
        print(f"{'='*55}")

        gt_hr = load_gt_hr(os.path.join(BASE_DIR, subj, "ground_truth.txt"))

        # Original
        print(f"  [Original]")
        bvp_orig, fps = run_rppg_on_video(
            os.path.join(BASE_DIR, subj, "vid.avi"),
            model_run, init_state, landmarker, "orig"
        )
        hr_orig = get_hr_welch(bvp_orig, sr=fps)

        # De-identified
        vid_deid = os.path.join(BASE_DIR, subj, "vid_deid.avi")
        if not os.path.exists(vid_deid):
            print(f"  WARNING: {vid_deid} not found")
            hr_deid = np.nan
            bvp_deid = np.array([])
        else:
            print(f"  [De-identified]")
            bvp_deid, _ = run_rppg_on_video(
                vid_deid, model_run, init_state, landmarker, "deid"
            )
            hr_deid = get_hr_welch(bvp_deid, sr=fps)

        mae_o = abs(hr_orig - gt_hr)
        mae_d = abs(hr_deid - gt_hr)

        all_results.append({
            "subject": subj, "gt_hr": gt_hr,
            "hr_orig": hr_orig, "hr_deid": hr_deid,
            "mae_orig": mae_o, "mae_deid": mae_d, "fps": fps,
        })

        print(f"  GT HR:       {gt_hr:.2f} bpm")
        print(f"  Original HR: {hr_orig:.2f} bpm  (MAE: {mae_o:.2f})")
        print(f"  De-ID HR:    {hr_deid:.2f} bpm  (MAE: {mae_d:.2f})")

        # ── Comparison plots ────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"{subj} — Original vs De-Identified", fontsize=14)

        t_o = np.arange(len(bvp_orig)) / fps
        axes[0, 0].plot(t_o, bvp_orig, linewidth=0.5)
        axes[0, 0].set_title("Original BVP")
        axes[0, 0].set_xlabel("Time (s)")

        f_o, p_o = welch(bvp_orig, fs=fps, nfft=int(1e5 / fps),
                         nperseg=min(len(bvp_orig) - 1, 256))
        axes[0, 1].plot(f_o * 60, p_o)
        axes[0, 1].axvline(hr_orig, color="r", ls="--", label=f"Pred {hr_orig:.1f}")
        axes[0, 1].axvline(gt_hr, color="g", ls="--", label=f"GT {gt_hr:.1f}")
        axes[0, 1].set_xlim(HR_MIN, HR_MAX)
        axes[0, 1].set_title("Original PSD")
        axes[0, 1].legend()

        if len(bvp_deid) > 0:
            t_d = np.arange(len(bvp_deid)) / fps
            axes[1, 0].plot(t_d, bvp_deid, linewidth=0.5, color="orange")
            axes[1, 0].set_title("De-ID BVP")
            axes[1, 0].set_xlabel("Time (s)")

            f_d, p_d = welch(bvp_deid, fs=fps, nfft=int(1e5 / fps),
                             nperseg=min(len(bvp_deid) - 1, 256))
            axes[1, 1].plot(f_d * 60, p_d, color="orange")
            axes[1, 1].axvline(hr_deid, color="r", ls="--", label=f"Pred {hr_deid:.1f}")
            axes[1, 1].axvline(gt_hr, color="g", ls="--", label=f"GT {gt_hr:.1f}")
            axes[1, 1].set_xlim(HR_MIN, HR_MAX)
            axes[1, 1].set_title("De-ID PSD")
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{subj}_comparison.png"), dpi=150)
        plt.close()

    # ── Summary ─────────────────────────────────────────────────────
    hr_o = np.array([r["hr_orig"] for r in all_results])
    hr_d = np.array([r["hr_deid"] for r in all_results])
    gts = np.array([r["gt_hr"] for r in all_results])

    def metrics(preds, gts):
        v = ~np.isnan(preds) & ~np.isnan(gts)
        if np.sum(v) < 2:
            return np.nan, np.nan, np.nan
        e = np.abs(preds[v] - gts[v])
        r, _ = pearsonr(preds[v], gts[v])
        return np.mean(e), np.sqrt(np.mean(e**2)), r

    mae_o, rmse_o, r_o = metrics(hr_o, gts)
    mae_d, rmse_d, r_d = metrics(hr_d, gts)

    print(f"\n{'='*65}")
    print("  VALIDATION RESULTS — Original vs De-Identified")
    print(f"{'='*65}")
    hdr = f"{'Subject':<12} {'GT':>6} {'Orig':>6} {'MAE_O':>7} {'DeID':>6} {'MAE_D':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        print(f"{r['subject']:<12} {r['gt_hr']:>6.1f} {r['hr_orig']:>6.1f} "
              f"{r['mae_orig']:>7.2f} {r['hr_deid']:>6.1f} {r['mae_deid']:>7.2f}")

    print(f"\n  {'Metric':<20} {'Original':>12} {'De-Identified':>15}")
    print(f"  {'-'*47}")
    print(f"  {'MAE (BPM)':<20} {mae_o:>12.2f} {mae_d:>15.2f}")
    print(f"  {'RMSE (BPM)':<20} {rmse_o:>12.2f} {rmse_d:>15.2f}")
    print(f"  {'Pearson r':<20} {r_o:>12.4f} {r_d:>15.4f}")

    delta = mae_d - mae_o
    print(f"\n  MAE degradation: {delta:+.2f} bpm")
    if abs(delta) < 3:
        print("  ✅ Minimal impact on rPPG accuracy")
    elif delta < 5:
        print("  ⚠️  Moderate impact")
    else:
        print("  ❌ Significant impact")

    print(f"\nPlots saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
