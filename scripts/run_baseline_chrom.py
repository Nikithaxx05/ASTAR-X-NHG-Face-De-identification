import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

SUBJECTS = ["subject1", "subject3", "subject4"]
BASE_DIR = "data/DATASET_2"
OUT_DIR = "outputs/baseline_chrom"

LOW_BPM = 40
HIGH_BPM = 180
MAX_SECONDS = 30  # process first 30s to keep it fast


def bandpass(x, fs, low_bpm=40, high_bpm=180, order=3):
    low = (low_bpm / 60.0) / (fs / 2.0)
    high = (high_bpm / 60.0) / (fs / 2.0)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def hr_from_signal(x, fs, low_bpm=40, high_bpm=180):
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs * 10)))
    low_hz = low_bpm / 60.0
    high_hz = high_bpm / 60.0
    mask = (f >= low_hz) & (f <= high_hz)
    if not np.any(mask):
        return np.nan, f, pxx
    f_band = f[mask]
    p_band = pxx[mask]
    peak_hz = f_band[np.argmax(p_band)]
    return float(peak_hz * 60.0), f, pxx


def load_gt(gt_path: str):
    data = np.loadtxt(gt_path)
    data = np.array(data).squeeze()
    return data


def chrom_bvp(frames_rgb: np.ndarray):
    RGB = frames_rgb.reshape(frames_rgb.shape[0], -1, 3).mean(axis=1).astype(np.float64)
    R, G, B = RGB[:, 0], RGB[:, 1], RGB[:, 2]
    Rn = (R - R.mean()) / (R.std() + 1e-8)
    Gn = (G - G.mean()) / (G.std() + 1e-8)
    Bn = (B - B.mean()) / (B.std() + 1e-8)
    X = 3 * Rn - 2 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn
    alpha = (np.std(X) / (np.std(Y) + 1e-8))
    return X - alpha * Y


def extract_face_roi_frames(video_path: str, max_seconds=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    max_frames = int(max_seconds * fps)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frames = []
    face_box = None
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret or i >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if face_box is None or i % 15 == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
                face_box = (x, y, w, h)

        if face_box is not None:
            x, y, w, h = face_box
            pad = int(0.15 * w)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)
            roi = frame[y0:y1, x0:x1]
        else:
            H, W = frame.shape[:2]
            roi = frame[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4]

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Resize to fixed size so all frames have same shape
        roi_rgb = cv2.resize(roi_rgb, (128, 128), interpolation=cv2.INTER_AREA)
        frames.append(roi_rgb)
        i += 1

    cap.release()
    return np.array(frames, dtype=np.uint8), float(fps)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    errors = []

    for subj in SUBJECTS:
        subj_dir = os.path.join(BASE_DIR, subj)
        vid = os.path.join(subj_dir, "vid.avi")
        gt_path = os.path.join(subj_dir, "ground_truth.txt")

        frames, fps = extract_face_roi_frames(vid, MAX_SECONDS)
        bvp = chrom_bvp(frames)
        bvp_f = bandpass(bvp, fps, LOW_BPM, HIGH_BPM)

        pred_hr, f, pxx = hr_from_signal(bvp_f, fps, LOW_BPM, HIGH_BPM)

        gt = load_gt(gt_path)

        # If GT looks like HR series (40..200), take mean; else estimate from waveform
        if np.nanmin(gt) > 30 and np.nanmax(gt) < 220:
            gt_hr = float(np.nanmean(gt))
        else:
            gt_f = bandpass(gt, fps, LOW_BPM, HIGH_BPM)
            gt_hr, _, _ = hr_from_signal(gt_f, fps, LOW_BPM, HIGH_BPM)

        err = abs(pred_hr - gt_hr)
        errors.append(err)

        print(f"\n{subj}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Pred HR: {pred_hr:.2f} bpm")
        print(f"  GT HR:   {gt_hr:.2f} bpm")
        print(f"  Abs Error: {err:.2f} bpm")

        # Save plots
        t = np.arange(len(bvp_f)) / fps
        plt.figure(figsize=(10, 3))
        plt.plot(t, bvp_f)
        plt.xlabel("Time (s)")
        plt.ylabel("BVP (filtered)")
        plt.title(f"{subj} - CHROM baseline BVP")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{subj}_bvp.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(10, 3))
        plt.plot(f, pxx)
        plt.xlim(0, 5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(f"{subj} - CHROM baseline PSD")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{subj}_psd.png"), dpi=160)
        plt.close()

    print("\n=== Summary ===")
    print(f"Mean Abs Error: {np.mean(errors):.2f} bpm")
    print(f"Std Abs Error:  {np.std(errors):.2f} bpm")
    print(f"Plots saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
