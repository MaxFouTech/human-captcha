"""
Numba-optimized core functions for high-performance CAPTCHA generation.
"""

import numpy as np
from numba import njit
import random


@njit(cache=True)
def hash_float(n):
    """Generate a pseudo-random float from an integer using bit manipulation."""
    n = (n << 13) ^ n
    n = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff
    return (n / 1073741824.0) - 1.0  # Returns -1.0 to 1.0


@njit(cache=True)
def smoothstep(t):
    """Smooth interpolation curve (6t^5 - 15t^4 + 10t^3)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True)
def perlin_noise_1d(x, seed=0):
    """
    1D Perlin-like coherent noise.
    Returns smooth noise value between -1 and 1 for any input x.
    Uses seed to generate different noise patterns.
    """
    # Integer and fractional parts
    xi = int(np.floor(x))
    xf = x - xi

    # Get gradient values at surrounding integer points
    g0 = hash_float(xi + seed)
    g1 = hash_float(xi + 1 + seed)

    # Compute dot products (in 1D, just multiply by distance)
    d0 = g0 * xf
    d1 = g1 * (xf - 1.0)

    # Interpolate using smoothstep
    t = smoothstep(xf)
    return d0 + t * (d1 - d0)


@njit(cache=True)
def perlin_noise_2d(x, y, seed=0):
    """
    2D Perlin-like coherent noise for more complex patterns.
    Returns smooth noise value between approximately -1 and 1.
    """
    xi = int(np.floor(x))
    yi = int(np.floor(y))
    xf = x - xi
    yf = y - yi

    # Get gradient values at four corners
    g00 = hash_float(xi + yi * 57 + seed)
    g10 = hash_float(xi + 1 + yi * 57 + seed)
    g01 = hash_float(xi + (yi + 1) * 57 + seed)
    g11 = hash_float(xi + 1 + (yi + 1) * 57 + seed)

    # Interpolate
    tx = smoothstep(xf)
    ty = smoothstep(yf)

    nx0 = g00 + tx * (g10 - g00)
    nx1 = g01 + tx * (g11 - g01)

    return nx0 + ty * (nx1 - nx0)


# --- Lemniscate + Curl Noise Motion System ---


@njit(cache=True)
def _grad_2d(hash_val, dx, dy):
    """Gradient dot product for permutation-table Perlin noise.
    Selects one of 4 gradient vectors based on hash_val & 3."""
    h = hash_val & 3
    if h == 0:
        return dx + dy
    elif h == 1:
        return -dx + dy
    elif h == 2:
        return dx - dy
    else:
        return -dx - dy


@njit(cache=True)
def build_perlin_perm_table(seed):
    """Build a permutation table from a 32-bit seed using LCG + Fisher-Yates.
    Returns int32[512] (p[0..255] doubled)."""
    state = np.int64(seed) & np.int64(0xFFFFFFFF)

    # Initialize identity permutation
    p = np.arange(256, dtype=np.int32)

    # Fisher-Yates shuffle using LCG
    for i in range(255, 0, -1):
        state = (state * np.int64(1664525) + np.int64(1013904223)) & np.int64(0xFFFFFFFF)
        j = int(state % np.int64(i + 1))
        p[i], p[j] = p[j], p[i]

    # Double the permutation table
    perm = np.empty(512, dtype=np.int32)
    for i in range(512):
        perm[i] = p[i & 255]

    return perm


@njit(cache=True)
def perlin_noise_2d_perm(x, y, perm):
    """2D Perlin noise using a prebuilt permutation table.
    Uses quintic fade (smoothstep) and 4-corner gradient interpolation.
    Returns float in approximately [-1, 1]."""
    xi = int(np.floor(x)) & 255
    yi = int(np.floor(y)) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)

    # Quintic fade
    u = smoothstep(xf)
    v = smoothstep(yf)

    # Hash corners
    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    # Gradient dot products
    g_aa = _grad_2d(aa, xf, yf)
    g_ba = _grad_2d(ba, xf - 1.0, yf)
    g_ab = _grad_2d(ab, xf, yf - 1.0)
    g_bb = _grad_2d(bb, xf - 1.0, yf - 1.0)

    # Bilinear interpolation
    nx0 = g_aa + u * (g_ba - g_aa)
    nx1 = g_ab + u * (g_bb - g_ab)

    return nx0 + v * (nx1 - nx0)


@njit(cache=True)
def _bounce_fold(v, lo, hi):
    """Reflect a value at boundaries instead of clamping.
    Values outside [lo, hi] bounce back like a ball."""
    r = hi - lo
    if r <= 0.0:
        return (lo + hi) * 0.5
    v = v - lo
    # Modulo into [0, 2*range)
    v = v % (r * 2.0)
    if v < 0.0:
        v += r * 2.0
    # Reflect if past range
    if v > r:
        v = r * 2.0 - v
    return v + lo


@njit(cache=True)
def precompute_lemniscate_positions(cx, cy, min_x, max_x, min_y, max_y,
                                    total_frames, base_ax, base_ay, perm,
                                    noise_offset_x, noise_offset_y,
                                    noise_scale, field_strength, warp_radius):
    """Precompute all frame positions using lemniscate + curl noise.

    Args:
        cx, cy: Center of usable area
        min_x, max_x, min_y, max_y: Boundary limits for text top-left
        total_frames: Number of frames (64)
        base_ax, base_ay: Half-amplitudes of usable area
        perm: int32[512] permutation table from build_perlin_perm_table
        noise_offset_x, noise_offset_y: Shift in noise space [-50, 50]
        noise_scale: Frequency of noise field (0.5)
        field_strength: Curl noise displacement strength (0.15)
        warp_radius: Radius of circular sampling path in noise space (1.0)

    Returns:
        (positions_x, positions_y) as float64[total_frames] arrays
    """
    positions_x = np.empty(total_frames, dtype=np.float64)
    positions_y = np.empty(total_frames, dtype=np.float64)

    usable_w = max_x - min_x
    usable_h = max_y - min_y
    max_drift = min(usable_w, usable_h) * field_strength * 0.5

    EPS = 0.01

    for i in range(total_frames):
        t = (float(i) / float(total_frames)) * 2.0 * np.pi * 0.4

        # Lemniscate-style diagonal motion (both X and Y)
        lx = np.sin(2.0 * t)  # [-1, 1] figure-8 X component (double frequency)
        ly = np.sin(t)         # [-1, 1] Y component

        # Curl noise: sample on circular path in noise space for perturbation
        snx = (warp_radius * np.cos(t) + noise_offset_x) * noise_scale
        sny = (warp_radius * np.sin(t) + noise_offset_y) * noise_scale

        # Finite-difference curl for both X and Y perturbation
        dp_dx = (perlin_noise_2d_perm(snx + EPS, sny, perm)
                 - perlin_noise_2d_perm(snx - EPS, sny, perm)) / (2.0 * EPS)
        dp_dy = (perlin_noise_2d_perm(snx, sny + EPS, perm)
                 - perlin_noise_2d_perm(snx, sny - EPS, perm)) / (2.0 * EPS)
        curl_vx = dp_dy
        curl_vy = -dp_dx

        raw_x = cx + lx * base_ax + curl_vx * max_drift
        raw_y = cy + ly * base_ay + curl_vy * max_drift

        positions_x[i] = _bounce_fold(raw_x, min_x, max_x)
        positions_y[i] = _bounce_fold(raw_y, min_y, max_y)

    return positions_x, positions_y


@njit(cache=True)
def create_noise_array_optimized(height, width, top_margin, bottom_margin, left_margin, right_margin):
    """Numba-optimized noise array generation."""
    arr = np.full((height, width), 255, dtype=np.uint8)
    noise_height = height - (top_margin + bottom_margin)
    noise_width = width - (left_margin + right_margin)

    for y in range(noise_height):
        for x in range(noise_width):
            if random.random() < 0.5:
                arr[top_margin + y, left_margin + x] = 0
            else:
                arr[top_margin + y, left_margin + x] = 255

    return arr


@njit(cache=True)
def pack_binary_data_optimized(frames_bool, frame_count, height, width):
    """Numba-optimized binary packing with bitwise operations."""
    pixels_per_frame = height * width
    packed_size = (pixels_per_frame + 7) // 8
    total_packed_size = frame_count * packed_size

    packed_data = np.zeros(total_packed_size, dtype=np.uint8)

    for frame_idx in range(frame_count):
        frame_offset = frame_idx * packed_size

        for byte_idx in range(packed_size):
            byte_value = 0
            start_pixel = byte_idx * 8

            for bit_idx in range(8):
                pixel_idx = start_pixel + bit_idx
                if pixel_idx < pixels_per_frame:
                    y = pixel_idx // width
                    x = pixel_idx % width

                    if frames_bool[frame_idx, y, x]:
                        byte_value |= (1 << (7 - bit_idx))

            packed_data[frame_offset + byte_idx] = byte_value

    return packed_data


@njit(cache=True)
def apply_lissajous_motion(center_x, center_y, safe_left, safe_right,
                           safe_top, safe_bottom, frame_index, total_frames,
                           amplitude_x, amplitude_y, freq_x, freq_y, phase_x, phase_y,
                           noise_seed=0):
    """
    Lissajous curve motion - smooth periodic motion that naturally loops.

    x(t) = center_x + Ax * sin(ωx * t + φx)
    y(t) = center_y + Ay * sin(ωy * t + φy)

    Where t goes from 0 to 2π over total_frames.
    Using integer frequencies ensures seamless looping.
    Uses Perlin noise for smooth jitter instead of random per-frame noise.
    """
    # Calculate t for this frame (0 to 2π over the animation)
    t = (frame_index / total_frames) * 2.0 * np.pi

    # Calculate position using Lissajous equations
    new_x = center_x + amplitude_x * np.sin(freq_x * t + phase_x)
    new_y = center_y + amplitude_y * np.sin(freq_y * t + phase_y)

    # Add smooth Perlin noise jitter for security (makes tracking harder but stays smooth)
    # Scale frame_index to control noise frequency - lower divisor = faster variation
    noise_time = frame_index / 8.0  # Smooth variation over ~8 frames
    jitter_x = perlin_noise_1d(noise_time, noise_seed) * 1.5
    jitter_y = perlin_noise_1d(noise_time, noise_seed + 1000) * 0.8
    new_x += jitter_x
    new_y += jitter_y

    # Clamp to safe bounds (shouldn't be needed if amplitudes are correct, but safety first)
    margin = 10.0
    new_x = max(safe_left + margin, min(safe_right - margin, new_x))
    new_y = max(safe_top + margin, min(safe_bottom - margin, new_y))

    return new_x, new_y


@njit(cache=True)
def apply_path_motion(center_x, center_y, safe_left, safe_right,
                      safe_top, safe_bottom, frame_index, total_frames,
                      start_x, start_y, end_x, end_y, curve_amount, noise_seed):
    """
    Path-based motion: smooth drift from start to end, then reverse.
    Uses pendulum-style motion (forward 0-32 frames, reverse 32-64 frames).

    Args:
        center_x, center_y: Canvas center position
        safe_*: Boundary constraints
        frame_index: Current frame (0 to total_frames-1)
        total_frames: Total animation frames (typically 64)
        start_x, start_y: Path start position (relative, -1 to 1)
        end_x, end_y: Path end position (relative, -1 to 1)
        curve_amount: Arc curve intensity (0 = linear, >0 = curved)
        noise_seed: Seed for Perlin jitter

    Returns:
        (new_x, new_y) position in pixels
    """
    # Normalize frame to 0-1 progress
    progress = frame_index / total_frames

    # Pendulum motion: 0→0.5 forward, 0.5→1 reverse
    if progress < 0.5:
        t = progress * 2.0  # 0 → 1 during first half
    else:
        t = 2.0 - progress * 2.0  # 1 → 0 during second half

    # Apply smoothstep easing for natural slow-at-endpoints feel
    t_smooth = smoothstep(t)

    # Calculate path dimensions (convert relative to pixels)
    canvas_width = safe_right - safe_left
    canvas_height = safe_bottom - safe_top

    # Convert relative positions to pixel offsets from center
    path_start_px_x = start_x * canvas_width * 0.5
    path_start_px_y = start_y * canvas_height * 0.5
    path_end_px_x = end_x * canvas_width * 0.5
    path_end_px_y = end_y * canvas_height * 0.5

    # Linear interpolation along path
    offset_x = path_start_px_x + (path_end_px_x - path_start_px_x) * t_smooth
    offset_y = path_start_px_y + (path_end_px_y - path_start_px_y) * t_smooth

    # Optional arc curve (perpendicular bulge in the middle)
    if curve_amount != 0.0:
        # Arc peaks at t=0.5 (middle of path)
        arc_intensity = np.sin(t * np.pi) * curve_amount

        # Calculate perpendicular direction
        dx = path_end_px_x - path_start_px_x
        dy = path_end_px_y - path_start_px_y
        path_length = np.sqrt(dx * dx + dy * dy)

        if path_length > 0.001:  # Avoid division by zero
            perp_x = -dy / path_length * arc_intensity * canvas_width * 0.3
            perp_y = dx / path_length * arc_intensity * canvas_height * 0.3
            offset_x += perp_x
            offset_y += perp_y

    # Add small Perlin noise jitter for security (reduced for readability)
    noise_time = frame_index / 12.0  # Slower variation for smoother motion
    jitter_x = perlin_noise_1d(noise_time, noise_seed) * 0.8
    jitter_y = perlin_noise_1d(noise_time, noise_seed + 1000) * 0.5
    offset_x += jitter_x
    offset_y += jitter_y

    # Calculate final position
    new_x = center_x + offset_x
    new_y = center_y + offset_y

    # Clamp to safe bounds
    margin = 10.0
    new_x = max(safe_left + margin, min(safe_right - margin, new_x))
    new_y = max(safe_top + margin, min(safe_bottom - margin, new_y))

    return new_x, new_y


@njit(cache=True)
def apply_physics_motion_optimized(center_x, center_y, velocity_x, velocity_y,
                                   target_x, target_y, safe_left, safe_right,
                                   safe_top, safe_bottom, frame_index, time_offset):
    """
    Gentle Lissajous motion - slow, smooth movement for readable text.
    Uses small amplitudes and low frequency for human-friendly motion.
    Uses Perlin noise for smooth jitter instead of random per-frame noise.
    """
    total_frames = 64.0

    # Derive parameters from time_offset (random per captcha)
    seed_val = time_offset * 0.001

    # MINIMUM MOTION GUARANTEE: Ensure enough movement to defeat attacks
    # HARDENED: Increased amplitude to defeat variance-based attacks
    # Higher amplitude (25-40 px X, 12-20 px Y) makes averaging ineffective
    amplitude_x = 25.0 + 15.0 * (0.5 + 0.5 * np.sin(seed_val * 1.1))  # 25-40 px
    amplitude_y = 12.0 + 8.0 * (0.5 + 0.5 * np.sin(seed_val * 2.3))   # 12-20 px

    # LOW FREQUENCY: freq=1 means ONE gentle cycle over 64 frames
    freq_x = 1.0
    freq_y = 1.0

    # AVOID STATIONARY PHASES: Don't start near peaks/troughs (π/2, 3π/2)
    # where sine derivative is zero and motion is slowest
    # Offset phases to start in "active" zones (where motion is happening)
    raw_phase_x = (seed_val * 3.7) % (2.0 * np.pi)
    raw_phase_y = (seed_val * 5.3) % (2.0 * np.pi)

    # Push phases away from stationary points (π/2 and 3π/2)
    # Add π/4 offset to avoid starting at peaks/troughs
    phase_x = raw_phase_x + 0.785  # +π/4
    phase_y = raw_phase_y + 2.356  # +3π/4 (different offset for Y)

    # Use time_offset as noise seed for Perlin jitter (unique per captcha)
    noise_seed = int(time_offset) % 100000

    new_x, new_y = apply_lissajous_motion(
        center_x, center_y,
        safe_left, safe_right, safe_top, safe_bottom,
        frame_index, total_frames,
        amplitude_x, amplitude_y,
        freq_x, freq_y,
        phase_x, phase_y,
        noise_seed
    )

    return new_x, new_y, 0.0, 0.0


@njit(cache=True)
def generate_wide_noise_optimized(height, total_width, seed_value):
    """Numba-optimized wide noise generation for seamless looping."""
    random.seed(seed_value)
    noise_array = np.zeros((height, total_width), dtype=np.uint8)

    for y in range(height):
        for x in range(total_width):
            if random.random() < 0.5:
                noise_array[y, x] = 0
            else:
                noise_array[y, x] = 255

    return noise_array


@njit(cache=True)
def generate_density_field(height, width, seed, scale=0.05, min_density=0.40, max_density=0.60):
    """
    Generate 2D density field using Perlin noise.
    HARDENED: Creates spatially varying noise density to defeat variance-based attacks.

    Args:
        height, width: Dimensions of the density field
        seed: Random seed for Perlin noise
        scale: Controls "blobiness" (smaller = larger blobs)
        min_density, max_density: Range of density values (0.40-0.60 typical)

    Returns:
        2D float32 array of density values
    """
    density = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Use Perlin noise for smooth spatial variation
            noise_val = perlin_noise_2d(x * scale, y * scale, seed)
            # Map -1..1 to min_density..max_density
            normalized = (noise_val + 1.0) / 2.0
            density[y, x] = min_density + normalized * (max_density - min_density)

    return density


@njit(cache=True)
def generate_wide_noise_with_density(height, total_width, seed_value, density_field, field_width):
    """
    Generate noise with variable density per-pixel.
    HARDENED: Density varies spatially based on density_field.

    Args:
        height, total_width: Dimensions of noise array
        seed_value: Random seed
        density_field: 2D array of density values (0.0-1.0)
        field_width: Width of density field (for tiling)
    """
    random.seed(seed_value)
    noise_array = np.zeros((height, total_width), dtype=np.uint8)

    for y in range(height):
        for x in range(total_width):
            # Tile density field across total_width
            density = density_field[y, x % field_width]
            if random.random() < density:
                noise_array[y, x] = 0
            else:
                noise_array[y, x] = 255

    return noise_array


@njit(cache=True)
def generate_tall_wide_noise_with_density(height, total_height, width, total_width, seed_value,
                                          density_field, field_height, field_width):
    """
    Generate 2D scrollable noise with variable density.
    HARDENED: Density varies spatially for both background and text noise.

    Args:
        height, total_height, width, total_width: Dimensions
        seed_value: Random seed
        density_field: 2D array of density values
        field_height, field_width: Dimensions of density field (for tiling)
    """
    random.seed(seed_value)
    noise_array = np.zeros((total_height, total_width), dtype=np.uint8)

    for y in range(total_height):
        for x in range(total_width):
            # Tile density field across both dimensions
            density = density_field[y % field_height, x % field_width]
            if random.random() < density:
                noise_array[y, x] = 0
            else:
                noise_array[y, x] = 255

    return noise_array


@njit(cache=True)
def extract_visible_portion_optimized(noise_array, offset, width, height, total_width):
    """Numba-optimized extraction of visible portion from wide noise array."""
    visible = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            src_x = (offset + x) % total_width
            visible[y, x] = noise_array[y, src_x]

    return visible


@njit(cache=True)
def generate_tall_wide_noise_optimized(height, total_height, width, total_width, seed_value):
    """
    Generate noise array extended in both X and Y for 2D seamless scrolling.
    HARDENED: Supports vertical scrolling to defeat variance-based attacks.
    """
    random.seed(seed_value)
    noise_array = np.zeros((total_height, total_width), dtype=np.uint8)

    for y in range(total_height):
        for x in range(total_width):
            if random.random() < 0.5:
                noise_array[y, x] = 0
            else:
                noise_array[y, x] = 255

    return noise_array


@njit(cache=True)
def extract_visible_portion_2d_optimized(noise_array, offset_x, offset_y, width, height, total_width, total_height):
    """
    Extract visible portion with 2D wrapping for seamless X and Y scrolling.
    HARDENED: Supports vertical scrolling to defeat variance-based attacks.
    """
    visible = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            src_x = (offset_x + x) % total_width
            src_y = (offset_y + y) % total_height
            visible[y, x] = noise_array[src_y, src_x]

    return visible


@njit(cache=True)
def apply_per_char_motion(frame_index, total_frames, amplitude_x, amplitude_y,
                          freq_x, freq_y, phase_x, phase_y, noise_seed):
    """
    Calculate position offset for a single character's independent motion.
    Uses pendulum-style (forward then reverse) to ensure seamless looping.
    HARDENED Phase 3: Each character moves independently to defeat alignment attacks.

    Args:
        frame_index: Current frame number
        total_frames: Total frames in animation
        amplitude_x, amplitude_y: Motion range (1-3px typical for subtle breathing)
        freq_x, freq_y: Speed multipliers (affects how fast the pendulum swings)
        phase_x, phase_y: Phase offsets (shifts the timing)
        noise_seed: Seed for per-character variation

    Returns:
        (offset_x, offset_y) to add to character's base position
    """
    # Normalize frame to 0-1 progress
    progress = frame_index / total_frames

    # Pendulum motion: 0→0.5 forward, 0.5→1 reverse (ensures return to start)
    if progress < 0.5:
        t = progress * 2.0  # 0 → 1 during first half
    else:
        t = 2.0 - progress * 2.0  # 1 → 0 during second half

    # Apply smoothstep for natural easing
    t_smooth = smoothstep(t)

    # Use phase offsets to create different timing per character
    # Phase shifts the effective progress (wrapping around)
    phase_shift_x = (phase_x / (2.0 * np.pi))  # Normalize to 0-1
    phase_shift_y = (phase_y / (2.0 * np.pi))

    # Apply phase shift to create offset timing
    t_x = t_smooth * freq_x + phase_shift_x * 0.3  # Subtle phase influence
    t_y = t_smooth * freq_y + phase_shift_y * 0.3

    # Clamp to valid range
    t_x = min(1.0, max(0.0, t_x))
    t_y = min(1.0, max(0.0, t_y))

    # Calculate offset: starts at 0, goes to amplitude, returns to 0
    # Use sin curve for smooth motion (0 at start, peak at middle, 0 at end)
    offset_x = amplitude_x * np.sin(t_x * np.pi)
    offset_y = amplitude_y * np.sin(t_y * np.pi)

    # Add very small Perlin noise for security (but loop-friendly)
    # Use sin-wrapped noise time to ensure seamless looping
    noise_phase = np.sin(progress * 2.0 * np.pi)  # -1 to 1, loops seamlessly
    jitter_x = perlin_noise_1d(noise_phase * 2.0, noise_seed) * 0.3
    jitter_y = perlin_noise_1d(noise_phase * 2.0, noise_seed + 1000) * 0.2

    return offset_x + jitter_x, offset_y + jitter_y


@njit(cache=True)
def apply_decoy_to_background(background, decoy_mask, decoy_noise, opacity, height, width, seed):
    """
    Blend static decoy letters into background with given opacity.
    HARDENED Phase 2: Creates faded decoy letters in background.

    Args:
        background: Background noise array (height, width)
        decoy_mask: Boolean mask of decoy letter shapes
        decoy_noise: Noise array to use for decoy pixels
        opacity: Probability of showing decoy (0.0-1.0)
        height, width: Dimensions
        seed: Random seed for probabilistic blending

    Returns:
        Background with decoy letters blended in
    """
    random.seed(seed)
    result = np.copy(background)

    for y in range(height):
        for x in range(width):
            if decoy_mask[y, x]:
                # Probabilistically show decoy based on opacity
                if random.random() < opacity:
                    result[y, x] = decoy_noise[y, x]

    return result


@njit(cache=True)
def apply_decoys_with_varying_opacity(background, decoy_mask, decoy_noise, opacity_map,
                                       height, width, seed):
    """
    Blend decoy letters with per-decoy varying opacity.
    HARDENED Phase 2: Each decoy can have different opacity.

    Args:
        background: Background noise array
        decoy_mask: Boolean mask of decoy shapes
        decoy_noise: Noise for decoy pixels
        opacity_map: 2D array of opacity values (0.0-1.0) for each pixel
        height, width: Dimensions
        seed: Random seed

    Returns:
        Background with decoys blended
    """
    random.seed(seed)
    result = np.copy(background)

    for y in range(height):
        for x in range(width):
            if decoy_mask[y, x]:
                opacity = opacity_map[y, x]
                if random.random() < opacity:
                    result[y, x] = decoy_noise[y, x]

    return result


@njit(cache=True)
def combine_frames_optimized(background, text_noise, text_mask, height, width):
    """Numba-optimized frame combination using boolean mask."""
    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if text_mask[y, x]:
                result[y, x] = text_noise[y, x]
            else:
                result[y, x] = background[y, x]

    return result


def warm_up_jit():
    """Warm up all JIT functions to pre-compile them."""
    print("Warming up Numba JIT functions...")

    # Warm up Perlin noise functions
    _ = hash_float(42)
    _ = smoothstep(0.5)
    _ = perlin_noise_1d(1.5, 0)
    _ = perlin_noise_2d(1.5, 2.5, 0)

    _ = create_noise_array_optimized(100, 100, 10, 10, 10, 10)

    test_frames = np.random.random((2, 50, 50)) > 0.5
    _ = pack_binary_data_optimized(test_frames, 2, 50, 50)

    _ = apply_physics_motion_optimized(100.0, 100.0, 0.1, 0.1, 120.0, 120.0,
                                       50.0, 150.0, 50.0, 150.0, 0, 0.0)

    _ = generate_wide_noise_optimized(50, 100, 42)

    # HARDENED: Warm up density field functions
    test_density = generate_density_field(50, 50, 42)
    _ = generate_wide_noise_with_density(50, 100, 42, test_density, 50)
    _ = generate_tall_wide_noise_with_density(50, 100, 50, 100, 42, test_density, 50, 50)

    # HARDENED: Warm up 2D noise functions
    _ = generate_tall_wide_noise_optimized(50, 100, 50, 100, 42)

    test_noise = np.random.randint(0, 256, (50, 100), dtype=np.uint8)
    _ = extract_visible_portion_optimized(test_noise, 10, 50, 50, 100)

    # HARDENED: Warm up 2D extraction
    test_noise_2d = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    _ = extract_visible_portion_2d_optimized(test_noise_2d, 10, 10, 50, 50, 100, 100)

    test_bg = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    test_text = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    test_mask = np.random.random((50, 50)) > 0.5
    _ = combine_frames_optimized(test_bg, test_text, test_mask, 50, 50)

    # HARDENED Phase 2: Warm up decoy functions
    test_decoy_mask = np.random.random((50, 50)) > 0.7
    test_decoy_noise = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    _ = apply_decoy_to_background(test_bg, test_decoy_mask, test_decoy_noise, 0.4, 50, 50, 42)

    # HARDENED Phase 3: Warm up per-char motion function
    _ = apply_per_char_motion(0, 64, 5.0, 3.0, 1.0, 1.0, 0.0, 0.5, 42)

    # Path-based motion warmup
    _ = apply_path_motion(100.0, 75.0, 10.0, 190.0, 10.0, 140.0,
                          0, 64.0, -0.3, -0.3, 0.3, 0.3, 0.0, 42)

    # Lemniscate + curl noise motion warmup
    _ = _grad_2d(7, 0.5, 0.5)
    test_perm = build_perlin_perm_table(42)
    _ = perlin_noise_2d_perm(1.5, 2.5, test_perm)
    _ = _bounce_fold(105.0, 10.0, 100.0)
    _, _ = precompute_lemniscate_positions(
        300.0, 75.0, 60.0, 540.0, 30.0, 120.0,
        64, 25.0, 12.0, test_perm,
        0.0, 0.0, 0.5, 0.15, 1.0
    )

    print("Numba JIT warm-up complete!")
