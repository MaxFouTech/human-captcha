"""
CAPTCHA attack script — tests robustness against automated solving.

Connects to the running service, fetches captchas, and applies multiple
analysis techniques to extract human-readable text images.

Techniques:
  1. Temporal variance — per-pixel variance across frames reveals text mask
  2. Change rate — count binary state transitions per pixel
  3. Motion direction segmentation — optical flow separates H/V motion
  4. Temporal difference accumulation — sum of |frame[i] - frame[i-1]|
  5. Combined signal — weighted combination of all techniques

Usage:
    uv run --no-project --with numpy --with pillow --with httpx --with opencv-python-headless attack.py
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image


OUTPUT_DIR = Path(__file__).parent / "attack_results"


def fetch_captcha(base_url: str) -> tuple[str, dict, np.ndarray]:
    """Fetch a captcha from the service and return (captcha_id, metadata, frames)."""
    resp = httpx.post(
        f"{base_url}/api/captcha/generate-all",
        headers={"Accept": "application/octet-stream", "Content-Type": "application/json"},
        content=b"{}",
        timeout=30,
    )
    resp.raise_for_status()

    h = resp.headers
    width = int(h["x-frame-width"])
    height = int(h["x-frame-height"])
    total_frames = int(h["x-total-frame-count"])
    frame_packed_size = int(h["x-frame-packed-size"])
    captcha_id = h["x-captcha-id"]

    params_raw = h.get("x-captcha-params", "{}")
    try:
        params = json.loads(params_raw)
    except json.JSONDecodeError:
        params = {}

    metadata = {
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frame_packed_size": frame_packed_size,
        "params": params,
    }

    packed = np.frombuffer(resp.content, dtype=np.uint8)
    frames = np.zeros((total_frames, height, width), dtype=np.uint8)
    for f in range(total_frames):
        offset = f * frame_packed_size
        bits = np.unpackbits(packed[offset : offset + frame_packed_size])[: height * width]
        frames[f] = (bits * 255).reshape(height, width)

    return captcha_id, metadata, frames


def verify_answer(base_url: str, captcha_id: str, text: str, with_hints: bool = False) -> dict:
    """Submit an answer to the service."""
    resp = httpx.post(
        f"{base_url}/api/captcha/verify",
        json={"captcha_id": captcha_id, "text": text, "honeypot": "", "with_hints": with_hints},
        timeout=10,
    )
    return resp.json()


# ---------------------------------------------------------------------------
# Attack techniques — each returns a float32 image (0=background, 1=text signal)
# ---------------------------------------------------------------------------

def attack_variance(frames: np.ndarray) -> np.ndarray:
    """Per-pixel variance across frames. Text region has different variance."""
    f = frames.astype(np.float32) / 255.0
    var = np.var(f, axis=0)
    # Normalize to 0-1
    if var.max() > var.min():
        var = (var - var.min()) / (var.max() - var.min())
    return var


def attack_change_rate(frames: np.ndarray) -> np.ndarray:
    """Count how often each pixel flips between consecutive frames."""
    diffs = np.diff(frames.astype(np.int16), axis=0)
    changes = np.sum(np.abs(diffs) > 0, axis=0).astype(np.float32)
    if changes.max() > changes.min():
        changes = (changes - changes.min()) / (changes.max() - changes.min())
    return changes


def attack_temporal_diff(frames: np.ndarray) -> np.ndarray:
    """Accumulated absolute difference between consecutive frames."""
    acc = np.zeros(frames.shape[1:], dtype=np.float32)
    for i in range(1, len(frames)):
        acc += np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))
    if acc.max() > acc.min():
        acc = (acc - acc.min()) / (acc.max() - acc.min())
    return acc


def attack_motion_direction(frames: np.ndarray) -> np.ndarray:
    """
    Optical flow direction segmentation.
    Background moves horizontally, text moves vertically.
    Ratio of |flow_y| / (|flow_x| + |flow_y|) → high = text, low = background.
    """
    h, w = frames.shape[1], frames.shape[2]
    flow_x_acc = np.zeros((h, w), dtype=np.float64)
    flow_y_acc = np.zeros((h, w), dtype=np.float64)

    n_pairs = min(30, len(frames) - 1)
    step = max(1, (len(frames) - 1) // n_pairs)

    for i in range(0, len(frames) - 1, step):
        f1 = frames[i]
        f2 = frames[i + 1]
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_x_acc += np.abs(flow[:, :, 0])
        flow_y_acc += np.abs(flow[:, :, 1])

    total = flow_x_acc + flow_y_acc + 1e-8
    # High ratio = vertical motion = text
    ratio = flow_y_acc / total
    if ratio.max() > ratio.min():
        ratio = (ratio - ratio.min()) / (ratio.max() - ratio.min())
    return ratio.astype(np.float32)


def attack_combined(frames: np.ndarray) -> np.ndarray:
    """Weighted combination of all individual attack signals."""
    var = attack_variance(frames)
    change = attack_change_rate(frames)
    tdiff = attack_temporal_diff(frames)
    motion = attack_motion_direction(frames)

    # Equal weighting — all signals normalized to 0-1
    combined = 0.25 * var + 0.25 * change + 0.25 * tdiff + 0.25 * motion
    if combined.max() > combined.min():
        combined = (combined - combined.min()) / (combined.max() - combined.min())
    return combined


def _estimate_text_centroid_per_frame(frames: np.ndarray) -> np.ndarray:
    """
    Estimate the text region centroid in each frame using motion direction signal.
    Returns array of shape (n_frames, 2) with (cx, cy) per frame.
    """
    h, w = frames.shape[1], frames.shape[2]
    n = len(frames)

    # First pass: get a rough motion direction map from all frame pairs
    flow_x_acc = np.zeros((h, w), dtype=np.float64)
    flow_y_acc = np.zeros((h, w), dtype=np.float64)
    for i in range(0, n - 1, 2):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_x_acc += np.abs(flow[:, :, 0])
        flow_y_acc += np.abs(flow[:, :, 1])

    total = flow_x_acc + flow_y_acc + 1e-8
    ratio = flow_y_acc / total  # high = text

    # Threshold to get text mask
    thresh = np.percentile(ratio, 70)
    text_mask = ratio > thresh

    # For each frame, compute weighted centroid of the text region
    # Use a sliding window of frames to get per-frame flow and track centroid
    ys, xs = np.where(text_mask)
    if len(xs) == 0:
        # Fallback: center
        return np.tile([w // 2, h // 2], (n, 1)).astype(np.float64)

    # Base centroid from the overall mask
    base_cx = np.mean(xs)
    base_cy = np.mean(ys)

    # Track centroid displacement using optical flow
    centroids = np.zeros((n, 2), dtype=np.float64)
    centroids[0] = [base_cx, base_cy]

    for i in range(1, n):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i - 1], frames[i], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        # Median flow in text region
        if text_mask.sum() > 0:
            dx = np.median(flow[text_mask, 0])
            dy = np.median(flow[text_mask, 1])
        else:
            dx, dy = 0.0, 0.0
        centroids[i] = centroids[i - 1] + [dx, dy]

    return centroids


def _align_frames(frames: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Shift each frame so the text centroid aligns to the mean position."""
    n, h, w = frames.shape
    mean_cx = np.mean(centroids[:, 0])
    mean_cy = np.mean(centroids[:, 1])

    aligned = np.zeros_like(frames)
    for i in range(n):
        dx = mean_cx - centroids[i, 0]
        dy = mean_cy - centroids[i, 1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned[i] = cv2.warpAffine(frames[i], M, (w, h),
                                     borderMode=cv2.BORDER_WRAP)
    return aligned


def attack_motion_direction_enhanced(frames: np.ndarray) -> np.ndarray:
    """
    Enhanced motion direction attack:
    1. Use ALL frame pairs, not just 30
    2. Try multiple frame gaps (1, 2, 3) for stronger signal
    3. Apply sharpening and contrast enhancement
    4. Morphological cleanup
    """
    h, w = frames.shape[1], frames.shape[2]
    flow_x_acc = np.zeros((h, w), dtype=np.float64)
    flow_y_acc = np.zeros((h, w), dtype=np.float64)

    # Multiple frame gaps for richer signal
    for gap in [1, 2, 3]:
        for i in range(0, len(frames) - gap):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i + gap], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_x_acc += np.abs(flow[:, :, 0])
            flow_y_acc += np.abs(flow[:, :, 1])

    total = flow_x_acc + flow_y_acc + 1e-8
    ratio = flow_y_acc / total

    if ratio.max() > ratio.min():
        ratio = (ratio - ratio.min()) / (ratio.max() - ratio.min())

    # Convert to uint8 for OpenCV processing
    img = (ratio * 255).clip(0, 255).astype(np.uint8)

    # Apply CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Sharpen with unsharp mask
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 2.0, blurred, -1.0, 0)

    # Morphological closing to fill gaps in letters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Normalize back to 0-1
    result = img.astype(np.float32) / 255.0
    if result.max() > result.min():
        result = (result - result.min()) / (result.max() - result.min())
    return result


def attack_motion_compensated(frames: np.ndarray) -> np.ndarray:
    """
    Motion-compensated attack:
    1. Estimate text centroid trajectory across frames
    2. Align all frames to cancel position movement
    3. Run enhanced motion direction attack on aligned frames
    """
    centroids = _estimate_text_centroid_per_frame(frames)
    aligned = _align_frames(frames, centroids)
    return attack_motion_direction_enhanced(aligned)


# ---------------------------------------------------------------------------
# Attack: Temporal PCA — source separation via principal components
# ---------------------------------------------------------------------------

def attack_temporal_pca(frames: np.ndarray) -> np.ndarray:
    """
    Treat each pixel as a time series across frames.
    PCA decomposes into principal components — text and background
    should separate into different components due to different motion patterns.
    Returns the best component (highest text signal).
    """
    n_frames, h, w = frames.shape
    # Each pixel is a sample, each frame is a feature
    # Matrix: (h*w) x n_frames
    data = frames.reshape(n_frames, h * w).T.astype(np.float64)
    data -= data.mean(axis=0)

    # SVD (more stable than covariance for PCA)
    U, S, Vt = np.linalg.svd(data, full_matrices=False)

    # Try the first several components — one should capture text vs background
    best_signal = None
    best_contrast = 0.0
    for k in range(min(10, len(S))):
        component = U[:, k].reshape(h, w)
        # Normalize
        c = component - component.min()
        if c.max() > 0:
            c = c / c.max()
        # Score: how much contrast exists in the center vs edges
        # (text is typically centered)
        margin = 10
        center = c[margin:-margin, margin:-margin] if h > 2*margin and w > 2*margin else c
        edge_top = c[:margin, :]
        edge_bot = c[-margin:, :]
        contrast = abs(center.mean() - 0.5 * (edge_top.mean() + edge_bot.mean()))
        if contrast > best_contrast:
            best_contrast = contrast
            best_signal = c.astype(np.float32)

    if best_signal is None:
        best_signal = np.zeros((h, w), dtype=np.float32)
    return best_signal


# ---------------------------------------------------------------------------
# Attack: Phase correlation tracking + aligned motion direction
# ---------------------------------------------------------------------------

def _phase_correlation_shift(f1: np.ndarray, f2: np.ndarray) -> tuple[float, float]:
    """Compute sub-pixel shift between two frames using phase correlation."""
    f1f = np.fft.fft2(f1.astype(np.float64))
    f2f = np.fft.fft2(f2.astype(np.float64))
    cross = f1f * np.conj(f2f)
    cross /= np.abs(cross) + 1e-8
    corr = np.abs(np.fft.ifft2(cross))

    # Find peak
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak

    # Handle wraparound
    h, w = f1.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return float(dx), float(dy)


def _estimate_centroids_phase_corr(frames: np.ndarray) -> np.ndarray:
    """
    Track text region displacement using phase correlation.
    More robust to binary noise than optical flow.
    """
    n, h, w = frames.shape

    # Get text mask from motion direction signal
    motion_sig = attack_motion_direction(frames)
    thresh = np.percentile(motion_sig, 70)
    text_mask = motion_sig > thresh

    ys, xs = np.where(text_mask)
    if len(xs) == 0:
        return np.tile([w // 2, h // 2], (n, 1)).astype(np.float64)

    # Crop a region around the text for phase correlation
    # (global phase correlation captures background motion, not text)
    min_x, max_x = max(0, xs.min() - 20), min(w, xs.max() + 20)
    min_y, max_y = max(0, ys.min() - 20), min(h, ys.max() + 20)

    base_cx = (min_x + max_x) / 2.0
    base_cy = (min_y + max_y) / 2.0

    centroids = np.zeros((n, 2), dtype=np.float64)
    centroids[0] = [base_cx, base_cy]

    for i in range(1, n):
        # Crop text region from consecutive frames
        crop1 = frames[i-1, min_y:max_y, min_x:max_x]
        crop2 = frames[i, min_y:max_y, min_x:max_x]
        if crop1.size == 0 or crop2.size == 0:
            centroids[i] = centroids[i-1]
            continue
        dx, dy = _phase_correlation_shift(crop1, crop2)
        centroids[i] = centroids[i-1] + [dx, dy]

    return centroids


def attack_phase_corr_aligned(frames: np.ndarray) -> np.ndarray:
    """
    Phase-correlation based motion compensation:
    1. Track text centroid with phase correlation (robust to binary noise)
    2. Align all frames
    3. Run motion direction on aligned frames
    """
    centroids = _estimate_centroids_phase_corr(frames)
    aligned = _align_frames(frames, centroids)
    return attack_motion_direction(aligned)


# ---------------------------------------------------------------------------
# Attack: Lemniscate trajectory fitting + alignment
# ---------------------------------------------------------------------------

def _score_alignment(frames: np.ndarray, offsets_x: np.ndarray, offsets_y: np.ndarray) -> float:
    """
    Score how well a set of per-frame offsets aligns the text.
    Higher = better alignment (measured by motion direction signal contrast).
    """
    n, h, w = frames.shape
    aligned = np.zeros_like(frames)
    for i in range(n):
        dx = -offsets_x[i]
        dy = -offsets_y[i]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned[i] = cv2.warpAffine(frames[i], M, (w, h), borderMode=cv2.BORDER_WRAP)

    # Quick motion direction signal — use fewer pairs for speed
    flow_x_acc = np.zeros((h, w), dtype=np.float64)
    flow_y_acc = np.zeros((h, w), dtype=np.float64)
    for i in range(0, n - 1, 4):
        flow = cv2.calcOpticalFlowFarneback(
            aligned[i], aligned[i + 1], None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_x_acc += np.abs(flow[:, :, 0])
        flow_y_acc += np.abs(flow[:, :, 1])

    total = flow_x_acc + flow_y_acc + 1e-8
    ratio = flow_y_acc / total

    # Score = variance of the ratio image (higher = more contrast = better separation)
    return float(np.var(ratio))


def attack_lemniscate_fit(frames: np.ndarray) -> np.ndarray:
    """
    Brute-force lemniscate trajectory parameters:
    - X = ax * sin(2t + phase)
    - Y = ay * sin(t + phase)
    For each candidate, shift frames to cancel trajectory, then measure
    signal contrast. Best parameters = correct trajectory → aligned frames.
    """
    n_frames, h, w = frames.shape
    t = np.linspace(0, 2 * np.pi * 0.4, n_frames)  # 0.4 matches the speed factor

    best_score = -1.0
    best_offsets = (np.zeros(n_frames), np.zeros(n_frames))

    # Search over amplitude and phase ranges
    ax_range = np.arange(10, 40, 6)
    ay_range = np.arange(8, 32, 6)
    phase_range = np.arange(0, 2 * np.pi, np.pi / 4)

    total_combos = len(ax_range) * len(ay_range) * len(phase_range)
    print(f"    Lemniscate fit: searching {total_combos} parameter combos...")

    for ax in ax_range:
        for ay in ay_range:
            for phase in phase_range:
                ox = ax * np.sin(2.0 * (t + phase))
                oy = ay * np.sin(t + phase)
                score = _score_alignment(frames, ox, oy)
                if score > best_score:
                    best_score = score
                    best_offsets = (ox.copy(), oy.copy())

    print(f"    Best score: {best_score:.6f}")

    # Align with best parameters and run full motion direction
    aligned = np.zeros_like(frames)
    for i in range(n_frames):
        dx = -best_offsets[0][i]
        dy = -best_offsets[1][i]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned[i] = cv2.warpAffine(frames[i], M, (w, h), borderMode=cv2.BORDER_WRAP)

    return attack_motion_direction_enhanced(aligned)


# ---------------------------------------------------------------------------
# Attack: Sliding window local analysis
# ---------------------------------------------------------------------------

def attack_sliding_window(frames: np.ndarray) -> np.ndarray:
    """
    Instead of accumulating across ALL frames (which smears due to movement),
    use short sliding windows (8 frames) where position hasn't moved much.
    Extract local motion direction signal per window, then combine.
    """
    n_frames, h, w = frames.shape
    window_size = 8
    acc = np.zeros((h, w), dtype=np.float64)
    n_windows = 0

    for start in range(0, n_frames - window_size, 4):  # overlapping windows
        window = frames[start:start + window_size]
        flow_x = np.zeros((h, w), dtype=np.float64)
        flow_y = np.zeros((h, w), dtype=np.float64)

        for i in range(len(window) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                window[i], window[i + 1], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_x += np.abs(flow[:, :, 0])
            flow_y += np.abs(flow[:, :, 1])

        total = flow_x + flow_y + 1e-8
        ratio = flow_y / total
        acc += ratio
        n_windows += 1

    if n_windows > 0:
        acc /= n_windows
    if acc.max() > acc.min():
        acc = (acc - acc.min()) / (acc.max() - acc.min())
    return acc.astype(np.float32)


# ---------------------------------------------------------------------------
# Image output helpers
# ---------------------------------------------------------------------------

def signal_to_image(signal: np.ndarray, invert: bool = False) -> Image.Image:
    """Convert a 0-1 float signal to a grayscale PIL image."""
    img = (signal * 255).clip(0, 255).astype(np.uint8)
    if invert:
        img = 255 - img
    return Image.fromarray(img, mode="L")


def threshold_image(signal: np.ndarray, percentile: float = 30) -> Image.Image:
    """Threshold a signal at a given percentile and return binary image."""
    thresh = np.percentile(signal, percentile)
    binary = (signal < thresh).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def save_attack_results(
    attack_name: str,
    signal: np.ndarray,
    run_dir: Path,
):
    """Save raw signal, inverted, and thresholded versions."""
    prefix = run_dir / attack_name

    # Raw signal (bright = high signal)
    signal_to_image(signal).save(f"{prefix}_raw.png")
    # Inverted (dark text on light background)
    signal_to_image(signal, invert=True).save(f"{prefix}_inverted.png")
    # Thresholded at various percentiles
    for pct in (20, 30, 40):
        threshold_image(signal, pct).save(f"{prefix}_thresh{pct}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ATTACKS = {
    "variance": attack_variance,
    "change_rate": attack_change_rate,
    "temporal_diff": attack_temporal_diff,
    "motion_direction": attack_motion_direction,
    "motion_direction_enhanced": attack_motion_direction_enhanced,
    "motion_compensated": attack_motion_compensated,
    "combined": attack_combined,
    "temporal_pca": attack_temporal_pca,
    "phase_corr_aligned": attack_phase_corr_aligned,
    "lemniscate_fit": attack_lemniscate_fit,
    "sliding_window": attack_sliding_window,
}


def run_single(base_url: str, run_index: int) -> dict:
    """Run all attacks on a single captcha."""
    print(f"\n--- Captcha #{run_index + 1} ---")
    captcha_id, meta, frames = fetch_captcha(base_url)
    print(f"  Fetched: {meta['total_frames']} frames @ {meta['width']}x{meta['height']}, id={captcha_id[:8]}...")

    params = meta.get("params", {})
    if params:
        print(f"  Params: bg_spd={params.get('bg_speed', '?'):.2f}  "
              f"txt_spd_x={params.get('text_speed_x', '?'):.2f}  "
              f"txt_spd_y={params.get('text_speed_y', '?'):.2f}  "
              f"lem_ax={params.get('lemniscate_ax', '?'):.1f}  "
              f"lem_ay={params.get('lemniscate_ay', '?'):.1f}  "
              f"dir={params.get('text_motion_type', '?')}")

    run_dir = OUTPUT_DIR / f"captcha_{run_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save captcha_id and params for analysis
    (run_dir / "captcha_id.txt").write_text(captcha_id)
    (run_dir / "params.json").write_text(json.dumps(params, indent=2))

    # Save a few raw frames for reference
    for i in [0, 1, len(frames) // 2]:
        Image.fromarray(frames[i], mode="L").save(run_dir / f"frame_{i:03d}.png")

    results = {}
    for name, attack_fn in ATTACKS.items():
        t0 = time.time()
        signal = attack_fn(frames)
        elapsed = (time.time() - t0) * 1000
        save_attack_results(name, signal, run_dir)
        print(f"  {name}: {elapsed:.0f}ms")
        results[name] = {"time_ms": elapsed}

    return {"captcha_id": captcha_id, "metadata": meta, "results": results}


def run_interactive(base_url: str, run_index: int) -> dict:
    """
    Interactive attack with hints:
    1. Run attacks and save images
    2. Prompt the operator (human/model) for a guess
    3. Submit with hints enabled, display feedback
    4. Allow refining the guess based on hint colors
    """
    result = run_single(base_url, run_index)
    captcha_id = result["captcha_id"]
    run_dir = OUTPUT_DIR / f"captcha_{run_index:03d}"

    print(f"\n  Attack images saved to: {run_dir}/")
    print(f"  Best signal: motion_direction_enhanced_inverted.png")

    for attempt in range(3):
        guess = input(f"  Attempt {attempt + 1}/3 — enter 6-char guess (or 'skip'): ").strip().upper()
        if guess == "SKIP" or not guess:
            print("  Skipped.")
            break

        resp = verify_answer(base_url, captcha_id, guess, with_hints=True)
        if resp.get("success"):
            print(f"  ✓ CORRECT! '{guess}' is the answer.")
            result["solved"] = True
            result["answer"] = guess
            result["attempts_used"] = attempt + 1
            return result

        hints = resp.get("hints", [])
        remaining = resp.get("attempts_remaining", 0)
        hint_display = []
        for h in hints:
            status = h["status"]
            letter = h["letter"]
            if status == "correct":
                hint_display.append(f"\033[42;30m {letter} \033[0m")  # green bg
            elif status == "present":
                hint_display.append(f"\033[43;30m {letter} \033[0m")  # yellow bg
            else:
                hint_display.append(f"\033[100;37m {letter} \033[0m")  # gray bg
        print(f"  Hints: {''.join(hint_display)}  ({remaining} attempts left)")

        if remaining <= 0:
            print("  No attempts remaining.")
            break

    result["solved"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description="CAPTCHA attack tester")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the CAPTCHA service")
    parser.add_argument("--count", type=int, default=5, help="Number of captchas to test")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode with hints")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Attacking {args.url} — {args.count} captchas")
    print(f"Output: {OUTPUT_DIR}")

    solved = 0
    total = 0
    for i in range(args.count):
        if args.interactive:
            result = run_interactive(args.url, i)
            total += 1
            if result.get("solved"):
                solved += 1
        else:
            run_single(args.url, i)

    if args.interactive:
        print(f"\n=== Results: {solved}/{total} solved ===")
    print(f"\nDone. Results saved to {OUTPUT_DIR}/")
    print("Review the generated images to assess human readability of extracted text.")


if __name__ == "__main__":
    main()
