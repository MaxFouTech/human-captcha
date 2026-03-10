"""
CAPTCHA generator using motion-based differential perception.
"""

import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .optimized_core import (
    create_noise_array_optimized,
    pack_binary_data_optimized,
    generate_wide_noise_optimized,
    generate_tall_wide_noise_optimized,
    generate_density_field,
    generate_wide_noise_with_density,
    generate_tall_wide_noise_with_density,
    extract_visible_portion_optimized,
    extract_visible_portion_2d_optimized,
    apply_decoy_to_background,
    apply_per_char_motion,
    combine_frames_optimized,
    perlin_noise_1d,
    build_perlin_perm_table,
    precompute_lemniscate_positions,
)


# Font paths
POSSIBLE_FONT_PATHS = [
    Path(__file__).parent / "fonts" / "Verdana Bold.ttf",
    Path("/System/Library/Fonts/Supplemental/Verdana Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Verdana.ttf"),
    Path("/Library/Fonts/Verdana Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("C:/Windows/Fonts/verdanab.ttf"),
]

REGISTERED_FONT_PATH = None
for fpath in POSSIBLE_FONT_PATHS:
    if fpath.exists():
        REGISTERED_FONT_PATH = str(fpath)
        print(f"CAPTCHA: Found font at: {fpath}")
        break
if not REGISTERED_FONT_PATH:
    print("CAPTCHA: Warning: Could not find Verdana Bold; will use system default.")


# In-memory storage for CAPTCHA challenges
active_captchas: dict = {}


# CAPTCHA configuration
CAPTCHA_CONFIG = {
    "width": 600,
    "height": 150,
    "pixelSize": 1,
    "text_size": 91,
    "margins": {
        "top": 0.20,
        "bottom": 0.20,
        "left": 0.10,
        "right": 0.10
    },
    "expirationTime": 5 * 60 * 1000,  # 5 minutes in ms
    "chars": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "challengeLength": 6,
    "fps": 60,
    "frameCount": 64,
    # Background motion - randomized per captcha
    "motionSpeed": {"min": 1.5, "max": 3.0},
    # Text motion X
    "textMotionSpeed": {"min": 0.8, "max": 1.8},
    # Text motion Y
    "textMotionSpeedY": {"min": 1.5, "max": 2.2},
    # Pixel flip rate — disrupts optical flow estimation
    "pixel_flip_rate": 0.0,
    # Per-frame jitter
    "frame_jitter": {"x": 0.3, "y": 0.2},
    "initial_position_range": {
        "x": [-28, 28],
        "y": [-14, 14]
    },
    # Speed variation per frame (HARDENED: was 0.05)
    "speed_variation": 0.20,
    "dynamic_position": True,
    "position_speed": {"min": 0.1, "max": 0.3},
    "position_range": {"min": 7, "max": 21},
    "motion_patterns": ["linear", "circular", "zigzag", "random_walk"],
    # NEW: Random motion direction for text noise
    "text_motion_directions": ["vertical"],
    # HARDENED: Feature toggles for anti-attack measures
    "dynamic_density": True,      # Phase 1: Variable noise density
    "decoy_letters": False,       # Phase 2: Static decoy letters (disabled for comparison)
    "per_char_motion": False,     # Phase 3: Per-character motion (disabled for comparison)
    # Dynamic density settings
    "density_scale": 0.05,        # Perlin noise scale (smaller = larger blobs)
    "density_min": 0.40,          # Minimum density (40%)
    "density_max": 0.60,          # Maximum density (60%)
    # Decoy letter settings
    "decoy_count": {"min": 4, "max": 6},  # Number of decoy letters
    "decoy_opacity": {"min": 0.30, "max": 0.50},  # Decoy visibility (faded)
    "decoy_size_ratio": {"min": 0.7, "max": 1.0},  # Decoy size relative to real text
    # Per-character motion settings (reduced for readability, prevents overlap)
    "per_char_amplitude_x": {"min": 0.5, "max": 2.0},  # X motion range per char (subtle breathing)
    "per_char_amplitude_y": {"min": 0.3, "max": 1.5},  # Y motion range per char (subtle breathing)
    "per_char_freq_x": {"min": 0.8, "max": 1.2},  # X frequency variation
    "per_char_freq_y": {"min": 0.8, "max": 1.2},  # Y frequency variation
    # Lemniscate + curl noise motion
    "motion_type": "lemniscate_curl",
    "noise_scale": 0.5,
    "field_strength": 0.15,
    "warp_radius": 1.0,
    "noise_offset_range": 50,
    "lemniscate_amplitude_x": {"min": 28, "max": 38},
    "lemniscate_amplitude_y": {"min": 22, "max": 32},
}


def apply_physics_motion(center_x, center_y, safe_bounds, frame_index, animation_state):
    """Look up precomputed lemniscate+curl noise position for this frame."""
    if animation_state and 'frame_positions_x' in animation_state:
        return (
            animation_state['frame_positions_x'][frame_index],
            animation_state['frame_positions_y'][frame_index],
        )
    return float(center_x), float(center_y)


def create_text_mask(noise_width, noise_height, text, text_size, pixel_size, frame_index=0, animation_state=None):
    """Create text mask for the CAPTCHA."""
    top_margin = int(noise_height * CAPTCHA_CONFIG["margins"]["top"])
    bottom_margin = int(noise_height * CAPTCHA_CONFIG["margins"]["bottom"])
    left_margin = int(noise_width * CAPTCHA_CONFIG["margins"]["left"])
    right_margin = int(noise_width * CAPTCHA_CONFIG["margins"]["right"])

    safe_left = left_margin
    safe_right = noise_width - right_margin
    safe_top = top_margin
    safe_bottom = noise_height - bottom_margin

    try:
        font = ImageFont.truetype(REGISTERED_FONT_PATH, text_size) if REGISTERED_FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    pil_img = Image.new('L', (noise_width, noise_height), 0)
    draw = ImageDraw.Draw(pil_img)

    center_x = noise_width // 2
    center_y = noise_height // 2

    if CAPTCHA_CONFIG["dynamic_position"]:
        safe_bounds = (safe_left, safe_right, safe_top, safe_bottom)
        center_x, center_y = apply_physics_motion(center_x, center_y, safe_bounds, frame_index, animation_state)

    draw.text((center_x, center_y), text, font=font, fill=255, anchor="mm", align="center")

    # Draw rectangle border around the text
    bbox = draw.textbbox((center_x, center_y), text, font=font, anchor="mm")
    padding = 10
    rect_left = bbox[0] - padding
    rect_top = bbox[1] - padding
    rect_right = bbox[2] + padding
    rect_bottom = bbox[3] + padding
    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], outline=255, width=4)

    mask_array = np.array(pil_img, dtype=np.uint8)
    text_mask = mask_array > 128

    return text_mask


def create_per_char_masks(noise_width, noise_height, text, text_size, pixel_size,
                          frame_index, animation_state, char_motion_params):
    """
    Create text mask with per-character independent motion.
    HARDENED Phase 3: Each character moves independently to defeat alignment attacks.

    Args:
        noise_width, noise_height: Canvas dimensions
        text: CAPTCHA text string (6 chars)
        text_size: Font size
        pixel_size: Pixel scaling factor
        frame_index: Current frame number
        animation_state: Global animation state dict
        char_motion_params: List of 6 dicts with per-char Lissajous params

    Returns:
        Boolean mask with all characters rendered at their independent positions
    """
    top_margin = int(noise_height * CAPTCHA_CONFIG["margins"]["top"])
    bottom_margin = int(noise_height * CAPTCHA_CONFIG["margins"]["bottom"])
    left_margin = int(noise_width * CAPTCHA_CONFIG["margins"]["left"])
    right_margin = int(noise_width * CAPTCHA_CONFIG["margins"]["right"])

    safe_left = left_margin
    safe_right = noise_width - right_margin
    safe_top = top_margin
    safe_bottom = noise_height - bottom_margin

    try:
        font = ImageFont.truetype(REGISTERED_FONT_PATH, text_size) if REGISTERED_FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    pil_img = Image.new('L', (noise_width, noise_height), 0)
    draw = ImageDraw.Draw(pil_img)

    # Get global text center position (from existing motion system)
    center_x = noise_width // 2
    center_y = noise_height // 2

    if CAPTCHA_CONFIG["dynamic_position"]:
        safe_bounds = (safe_left, safe_right, safe_top, safe_bottom)
        center_x, center_y = apply_physics_motion(center_x, center_y, safe_bounds, frame_index, animation_state)

    # Calculate total text width and character positions
    total_frames = CAPTCHA_CONFIG["frameCount"]
    char_widths = []
    for char in text:
        bbox = font.getbbox(char)
        char_widths.append(bbox[2] - bbox[0])

    total_text_width = sum(char_widths)
    char_spacing = 8  # Gap between characters (increased to prevent overlap with per-char motion)

    # Starting x position (left edge of first character)
    start_x = center_x - total_text_width // 2 - (len(text) - 1) * char_spacing // 2

    # Render each character with independent motion
    current_x = start_x
    for i, char in enumerate(text):
        # Get per-char motion offset
        params = char_motion_params[i]
        offset_x, offset_y = apply_per_char_motion(
            frame_index, total_frames,
            params['amplitude_x'], params['amplitude_y'],
            params['freq_x'], params['freq_y'],
            params['phase_x'], params['phase_y'],
            params['noise_seed']
        )

        # Calculate character position with offset
        char_x = current_x + char_widths[i] // 2 + offset_x
        char_y = center_y + offset_y

        # Clamp to safe bounds
        char_x = max(safe_left + 10, min(safe_right - 10, char_x))
        char_y = max(safe_top + 5, min(safe_bottom - 5, char_y))

        # Draw character
        draw.text((char_x, char_y), char, font=font, fill=255, anchor="mm")

        # Move to next character position
        current_x += char_widths[i] + char_spacing

    mask_array = np.array(pil_img, dtype=np.uint8)
    text_mask = mask_array > 128

    return text_mask


def generate_per_char_motion_params(num_chars):
    """
    Generate random per-character motion parameters.
    HARDENED Phase 3: Creates independent motion params for each character.

    Args:
        num_chars: Number of characters (typically 6)

    Returns:
        List of dicts with per-char motion parameters
    """
    params = []
    for _ in range(num_chars):
        params.append({
            'amplitude_x': random.uniform(
                CAPTCHA_CONFIG["per_char_amplitude_x"]["min"],
                CAPTCHA_CONFIG["per_char_amplitude_x"]["max"]
            ),
            'amplitude_y': random.uniform(
                CAPTCHA_CONFIG["per_char_amplitude_y"]["min"],
                CAPTCHA_CONFIG["per_char_amplitude_y"]["max"]
            ),
            'freq_x': random.uniform(
                CAPTCHA_CONFIG["per_char_freq_x"]["min"],
                CAPTCHA_CONFIG["per_char_freq_x"]["max"]
            ),
            'freq_y': random.uniform(
                CAPTCHA_CONFIG["per_char_freq_y"]["min"],
                CAPTCHA_CONFIG["per_char_freq_y"]["max"]
            ),
            'phase_x': random.uniform(0, 2 * np.pi),
            'phase_y': random.uniform(0, 2 * np.pi),
            'noise_seed': random.randint(0, 100000),
        })
    return params


def generate_decoy_letter_mask(noise_width, noise_height, text_size, decoy_params):
    """
    Generate mask with static decoy letters at fixed positions.
    HARDENED Phase 2: Decoy letters confuse vision models.

    Args:
        noise_width, noise_height: Dimensions of the mask
        text_size: Base text size (decoys may be slightly smaller)
        decoy_params: List of dicts with 'char', 'x', 'y', 'size_ratio'

    Returns:
        Boolean mask with decoy letter shapes
    """
    try:
        font_base = ImageFont.truetype(REGISTERED_FONT_PATH, text_size) if REGISTERED_FONT_PATH else ImageFont.load_default()
    except Exception:
        font_base = ImageFont.load_default()

    pil_img = Image.new('L', (noise_width, noise_height), 0)
    draw = ImageDraw.Draw(pil_img)

    for decoy in decoy_params:
        # Scale font size for this decoy
        decoy_size = int(text_size * decoy['size_ratio'])
        try:
            font = ImageFont.truetype(REGISTERED_FONT_PATH, decoy_size) if REGISTERED_FONT_PATH else font_base
        except Exception:
            font = font_base

        # Draw decoy letter at fixed position
        draw.text((decoy['x'], decoy['y']), decoy['char'], font=font, fill=255, anchor="mm")

    mask_array = np.array(pil_img, dtype=np.uint8)
    decoy_mask = mask_array > 128

    return decoy_mask


def generate_decoy_params(noise_width, noise_height, real_text_bbox=None):
    """
    Generate random decoy letter parameters.
    HARDENED Phase 2: Creates parameters for static decoy letters.

    Args:
        noise_width, noise_height: Canvas dimensions
        real_text_bbox: Optional (left, top, right, bottom) of real text area to avoid

    Returns:
        List of decoy parameter dicts
    """
    num_decoys = random.randint(
        CAPTCHA_CONFIG["decoy_count"]["min"],
        CAPTCHA_CONFIG["decoy_count"]["max"]
    )

    chars = CAPTCHA_CONFIG["chars"]
    margin_x = int(noise_width * 0.05)
    margin_y = int(noise_height * 0.15)

    decoy_params = []
    for _ in range(num_decoys):
        # Random position avoiding edges
        x = random.randint(margin_x, noise_width - margin_x)
        y = random.randint(margin_y, noise_height - margin_y)

        # Random character
        char = random.choice(chars)

        # Random size (slightly smaller than real text)
        size_ratio = random.uniform(
            CAPTCHA_CONFIG["decoy_size_ratio"]["min"],
            CAPTCHA_CONFIG["decoy_size_ratio"]["max"]
        )

        # Random opacity for this decoy
        opacity = random.uniform(
            CAPTCHA_CONFIG["decoy_opacity"]["min"],
            CAPTCHA_CONFIG["decoy_opacity"]["max"]
        )

        decoy_params.append({
            'char': char,
            'x': x,
            'y': y,
            'size_ratio': size_ratio,
            'opacity': opacity,
        })

    return decoy_params


def pack_binary_data(frames_array):
    """Pack binary frame data using Numba JIT."""
    bool_array = (frames_array == 255)
    frame_count, height, width = bool_array.shape
    return pack_binary_data_optimized(bool_array, frame_count, height, width).tobytes()


def unpack_frame_size(width, height):
    """Calculate size of a packed frame in bytes."""
    return (width * height + 7) // 8


def generate_captcha_text():
    """Generate random CAPTCHA text."""
    chars = CAPTCHA_CONFIG["chars"]
    length = CAPTCHA_CONFIG["challengeLength"]
    return ''.join(random.choice(chars) for _ in range(length))


def generate_captcha(seed: int | None = None, mode: str = "hardened", text: str | None = None):
    """
    Generate a CAPTCHA with all frames at once.

    Args:
        seed: Optional random seed for reproducibility. If provided, the same seed
              will generate the same captcha (useful for debugging).
        mode: "static" (no motion, clear text), "moving" (motion, no frame hiding),
              or "hardened" (motion + hide text every other frame).

    Returns tuple: (captcha_id, captcha_text, metadata, all_frames_packed)
    """
    start_time = time.time()

    # Generate or use provided seed for reproducibility
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    generation_seed = seed  # Store for metadata

    captcha_text = text.upper() if text else generate_captcha_text()
    captcha_id = str(uuid.uuid4())

    # Animation parameters - HARDENED with randomization
    # Randomize motion speeds (was fixed values)
    base_motion_speed = random.uniform(
        CAPTCHA_CONFIG["motionSpeed"]["min"],
        CAPTCHA_CONFIG["motionSpeed"]["max"]
    )
    base_text_speed = random.uniform(
        CAPTCHA_CONFIG["textMotionSpeed"]["min"],
        CAPTCHA_CONFIG["textMotionSpeed"]["max"]
    )
    base_text_speed_y = random.uniform(
        CAPTCHA_CONFIG["textMotionSpeedY"]["min"],
        CAPTCHA_CONFIG["textMotionSpeedY"]["max"]
    )

    # Random text motion direction (was always opposite to background)
    text_motion_dir = random.choice(CAPTCHA_CONFIG["text_motion_directions"])

    # Generate a noise seed for this captcha (used for Perlin noise)
    noise_seed = random.randint(0, 100000)

    animation_state = {
        'speed_multiplier': 1.0 + random.uniform(
            -CAPTCHA_CONFIG["speed_variation"],
            CAPTCHA_CONFIG["speed_variation"]
        ),
        'motion_direction': random.choice([-1, 1]),
        'text_direction': None,
        # NEW: Text motion direction type
        'text_motion_type': text_motion_dir,
        # Perlin noise seed for smooth jitter
        'noise_seed': noise_seed,
        # Jitter amplitude (still configurable)
        'frame_jitter_x': CAPTCHA_CONFIG["frame_jitter"]["x"],
        'frame_jitter_y': CAPTCHA_CONFIG["frame_jitter"]["y"],
    }

    # Set text direction based on motion type
    if text_motion_dir == "horizontal":
        animation_state['text_direction_x'] = random.choice([-1, 1])
        animation_state['text_direction_y'] = 0
    elif text_motion_dir == "vertical":
        animation_state['text_direction_x'] = 0
        animation_state['text_direction_y'] = random.choice([-1, 1])
    elif text_motion_dir == "diagonal_up":
        animation_state['text_direction_x'] = random.choice([-1, 1])
        animation_state['text_direction_y'] = -1
    else:  # diagonal_down
        animation_state['text_direction_x'] = random.choice([-1, 1])
        animation_state['text_direction_y'] = 1

    # Ensure text noise never scrolls in the same horizontal direction as background
    if animation_state['text_direction_x'] == animation_state['motion_direction']:
        animation_state['text_direction_x'] *= -1
    animation_state['text_direction'] = animation_state['text_direction_x']
    animation_state['effective_motion_speed'] = base_motion_speed * animation_state['speed_multiplier']
    animation_state['effective_text_speed'] = base_text_speed
    # HARDENED: Y-axis motion
    animation_state['effective_text_speed_y'] = base_text_speed_y
    animation_state['text_direction_y_scroll'] = random.choice([-1, 1])  # Random Y scroll direction

    pixel_size = CAPTCHA_CONFIG["pixelSize"]
    noise_width = CAPTCHA_CONFIG["width"] // pixel_size
    noise_height = CAPTCHA_CONFIG["height"] // pixel_size

    extra_width = noise_width
    total_width = noise_width + extra_width
    # HARDENED: Add extra height for Y-axis scrolling
    extra_height = noise_height
    total_height = noise_height + extra_height

    # Precompute lemniscate + curl noise text positions
    safe_left = int(noise_width * CAPTCHA_CONFIG["margins"]["left"])
    safe_right = noise_width - int(noise_width * CAPTCHA_CONFIG["margins"]["right"])
    safe_top = int(noise_height * CAPTCHA_CONFIG["margins"]["top"])
    safe_bottom = noise_height - int(noise_height * CAPTCHA_CONFIG["margins"]["bottom"])

    motion_seed = random.randint(0, 2**32 - 1)
    perm_table = build_perlin_perm_table(motion_seed)
    noise_offset_x = random.uniform(
        -CAPTCHA_CONFIG["noise_offset_range"],
        CAPTCHA_CONFIG["noise_offset_range"]
    )
    noise_offset_y = random.uniform(
        -CAPTCHA_CONFIG["noise_offset_range"],
        CAPTCHA_CONFIG["noise_offset_range"]
    )
    base_amplitude_x = random.uniform(
        CAPTCHA_CONFIG["lemniscate_amplitude_x"]["min"],
        CAPTCHA_CONFIG["lemniscate_amplitude_x"]["max"]
    )
    base_amplitude_y = random.uniform(
        CAPTCHA_CONFIG["lemniscate_amplitude_y"]["min"],
        CAPTCHA_CONFIG["lemniscate_amplitude_y"]["max"]
    )
    start_frame = random.randint(0, CAPTCHA_CONFIG["frameCount"] - 1)

    frame_positions_x, frame_positions_y = precompute_lemniscate_positions(
        float(noise_width // 2), float(noise_height // 2),
        float(safe_left), float(safe_right),
        float(safe_top), float(safe_bottom),
        CAPTCHA_CONFIG["frameCount"],
        base_amplitude_x, base_amplitude_y,
        perm_table,
        noise_offset_x, noise_offset_y,
        CAPTCHA_CONFIG["noise_scale"],
        CAPTCHA_CONFIG["field_strength"],
        CAPTCHA_CONFIG["warp_radius"],
    )

    animation_state['frame_positions_x'] = frame_positions_x
    animation_state['frame_positions_y'] = frame_positions_y
    animation_state['start_frame'] = start_frame
    animation_state['motion_seed'] = motion_seed
    animation_state['lemniscate_ax'] = base_amplitude_x
    animation_state['lemniscate_ay'] = base_amplitude_y

    # HARDENED Phase 1: Generate density fields for variable noise density
    if CAPTCHA_CONFIG["dynamic_density"]:
        background_density = generate_density_field(
            noise_height, noise_width,
            random.randint(0, 100000),
            CAPTCHA_CONFIG["density_scale"],
            CAPTCHA_CONFIG["density_min"],
            CAPTCHA_CONFIG["density_max"]
        )
        text_density = generate_density_field(
            noise_height, noise_width,
            random.randint(0, 100000),
            CAPTCHA_CONFIG["density_scale"],
            CAPTCHA_CONFIG["density_min"],
            CAPTCHA_CONFIG["density_max"]
        )
        # SINGLE NOISE LAYER: both background and text use the same noise texture
        # Text visibility comes from speed difference, not texture difference
        # This defeats flow-based segmentation since both regions have identical texture
        shared_noise = generate_tall_wide_noise_with_density(
            noise_height, total_height, noise_width, total_width, random.randint(0, 2**31),
            background_density, noise_height, noise_width
        )
        background_noise = shared_noise
        text_noise = shared_noise
    else:
        # Fallback to fixed 50% density — still single layer
        shared_noise = generate_tall_wide_noise_optimized(
            noise_height, total_height, noise_width, total_width, random.randint(0, 2**31)
        )
        background_noise = shared_noise
        text_noise = shared_noise

    # HARDENED Phase 2: Generate static decoy letters
    decoy_mask = None
    decoy_noise = None
    decoy_opacity = 0.4
    decoy_seed = random.randint(0, 100000)
    if CAPTCHA_CONFIG["decoy_letters"]:
        decoy_params = generate_decoy_params(noise_width, noise_height)
        decoy_mask = generate_decoy_letter_mask(
            noise_width, noise_height,
            CAPTCHA_CONFIG["text_size"],
            decoy_params
        )
        # Average opacity across all decoys
        decoy_opacity = sum(d['opacity'] for d in decoy_params) / len(decoy_params)
        # Generate separate noise for decoys
        if CAPTCHA_CONFIG["dynamic_density"]:
            decoy_density = generate_density_field(
                noise_height, noise_width,
                random.randint(0, 100000),
                CAPTCHA_CONFIG["density_scale"],
                CAPTCHA_CONFIG["density_min"],
                CAPTCHA_CONFIG["density_max"]
            )
            decoy_noise = generate_wide_noise_with_density(
                noise_height, total_width, random.randint(0, 2**31),
                decoy_density, noise_width
            )
        else:
            decoy_noise = generate_wide_noise_optimized(noise_height, total_width, random.randint(0, 2**31))

    # HARDENED Phase 3: Generate per-character motion parameters
    char_motion_params = None
    if CAPTCHA_CONFIG["per_char_motion"]:
        char_motion_params = generate_per_char_motion_params(len(captcha_text))

    # Generate all frames in a single pass
    all_frames = []
    background_motion = 0
    text_motion = 0
    text_motion_y = 0
    current_bg_offset = 0
    current_text_offset = 0
    current_text_offset_y = 0

    for frame_index in range(CAPTCHA_CONFIG["frameCount"]):
        # Smooth speed variation using Perlin noise
        noise_time = frame_index / 6.0
        speed_noise = perlin_noise_1d(noise_time, animation_state['noise_seed'] + 5000)
        speed_multiplier_wave = 1.0 + speed_noise * 0.15  # ±15% variation

        background_motion += animation_state['effective_motion_speed'] * speed_multiplier_wave
        text_motion += animation_state['effective_text_speed'] * speed_multiplier_wave
        text_motion_y += animation_state['effective_text_speed_y'] * speed_multiplier_wave

        # Smooth jitter using Perlin noise
        jitter_noise_x = perlin_noise_1d(noise_time, animation_state['noise_seed'] + 3000)
        jitter_noise_y = perlin_noise_1d(noise_time, animation_state['noise_seed'] + 4000)
        jitter_x = jitter_noise_x * animation_state['frame_jitter_x'] * 2.0
        jitter_y = jitter_noise_y * animation_state['frame_jitter_y'] * 2.0

        while background_motion >= 1:
            current_bg_offset = (current_bg_offset + animation_state['motion_direction']) % total_width
            background_motion -= 1
        while text_motion >= 1:
            current_text_offset = (current_text_offset + animation_state['text_direction']) % total_width
            text_motion -= 1
        while text_motion_y >= 1:
            current_text_offset_y = (current_text_offset_y + animation_state['text_direction_y_scroll']) % total_height
            text_motion_y -= 1

        jittered_text_offset_x = int(current_text_offset + jitter_x) % total_width
        jittered_text_offset_y = int(current_text_offset_y + jitter_y) % total_height

        visible_background = extract_visible_portion_2d_optimized(
            background_noise, current_bg_offset, 0,
            noise_width, noise_height, total_width, total_height
        )

        # HARDENED Phase 2: Apply static decoy letters to background
        if decoy_mask is not None and decoy_noise is not None:
            visible_decoy_noise = extract_visible_portion_2d_optimized(
                decoy_noise, current_bg_offset, 0,
                noise_width, noise_height, total_width, total_height
            )
            visible_background = apply_decoy_to_background(
                visible_background, decoy_mask, visible_decoy_noise,
                decoy_opacity, noise_height, noise_width, decoy_seed + frame_index
            )

        # HARDENED: Use 2D extraction for text noise
        visible_text = extract_visible_portion_2d_optimized(
            text_noise, jittered_text_offset_x, jittered_text_offset_y,
            noise_width, noise_height, total_width, total_height
        )

        # Create text shape mask
        if mode == "static":
            # Static: text centered, no position movement
            shape_mask = create_text_mask(
                noise_width, noise_height,
                captcha_text,
                CAPTCHA_CONFIG["text_size"],
                pixel_size,
                0, None  # frame_index=0, no animation_state = no lemniscate
            )
        elif char_motion_params is not None:
            shape_mask = create_per_char_masks(
                noise_width, noise_height,
                captcha_text,
                CAPTCHA_CONFIG["text_size"],
                pixel_size,
                frame_index,
                animation_state,
                char_motion_params
            )
        else:
            shape_mask = create_text_mask(
                noise_width, noise_height,
                captcha_text,
                CAPTCHA_CONFIG["text_size"],
                pixel_size,
                frame_index,
                animation_state
            )

        if mode == "hardened" and frame_index % 2 != 0:
            # Hardened: hide text every other frame — breaks optical flow
            normal_frame = visible_background.copy()
        else:
            normal_frame = combine_frames_optimized(
                visible_background, visible_text, shape_mask, noise_height, noise_width
            )

        all_frames.append(normal_frame)

    all_frames_array = np.array(all_frames)
    all_frames_packed = pack_binary_data(all_frames_array)

    # Store CAPTCHA
    captcha = {
        "id": captcha_id,
        "text": captcha_text,
        "created_at": int(time.time() * 1000),
        "expires_at": int(time.time() * 1000) + CAPTCHA_CONFIG["expirationTime"],
        "used": False,
        "attempts": 0,
        "max_attempts": 3,
    }
    active_captchas[captcha_id] = captcha

    frame_packed_size = unpack_frame_size(noise_width, noise_height)

    total_generation_time = (time.time() - start_time) * 1000
    print(f"CAPTCHA generated in {total_generation_time:.2f}ms")

    metadata = {
        "width": noise_width,
        "height": noise_height,
        "total_frame_count": CAPTCHA_CONFIG["frameCount"],
        "frame_packed_size": frame_packed_size,
        "frame_interval": 1000 / CAPTCHA_CONFIG["fps"],
        "generation_time": total_generation_time,
        "seed": generation_seed,
        "motion_seed": animation_state.get('motion_seed'),
        "start_frame": animation_state.get('start_frame', 0),
        # Animation parameters for analysis
        "params": {
            "bg_speed": animation_state.get('effective_motion_speed'),
            "bg_direction": animation_state.get('motion_direction'),
            "text_speed_x": animation_state.get('effective_text_speed'),
            "text_speed_y": animation_state.get('effective_text_speed_y'),
            "text_direction_x": animation_state.get('text_direction_x'),
            "text_direction_y": animation_state.get('text_direction_y'),
            "text_motion_type": animation_state.get('text_motion_type'),
            "lemniscate_ax": animation_state.get('lemniscate_ax'),
            "lemniscate_ay": animation_state.get('lemniscate_ay'),
            "jitter_x": animation_state.get('frame_jitter_x'),
            "jitter_y": animation_state.get('frame_jitter_y'),
            "dir_seed": animation_state.get('noise_seed', 0) + 9000,
        },
    }

    return captcha_id, captcha_text, metadata, all_frames_packed


def _compute_hints(answer: str, guess: str) -> list[dict]:
    """
    Compute per-letter hints (Wordle-style).
    Returns list of {"letter": str, "status": "correct"|"present"|"absent"}.
    """
    hints = []
    answer_chars = list(answer)
    guess_chars = list(guess)
    n = min(len(answer_chars), len(guess_chars))

    # Track which answer chars are still available for "present" matching
    available = list(answer_chars)

    # First pass: mark correct positions
    for i in range(n):
        if guess_chars[i] == answer_chars[i]:
            hints.append({"letter": guess_chars[i], "status": "correct"})
            available[i] = None  # consumed
        else:
            hints.append({"letter": guess_chars[i], "status": None})  # placeholder

    # Second pass: mark present/absent
    for i in range(n):
        if hints[i]["status"] is not None:
            continue
        if guess_chars[i] in available:
            hints[i]["status"] = "present"
            available[available.index(guess_chars[i])] = None  # consume
        else:
            hints[i]["status"] = "absent"

    return hints


def verify_captcha(captcha_id: str, user_input: str, with_hints: bool = False) -> dict:
    """
    Verify a CAPTCHA answer.
    Returns dict with 'success', 'error', 'status_code' keys.
    If with_hints=True, includes per-letter hints on incorrect guesses.
    """
    user_input = user_input.upper().strip()

    if not captcha_id or not user_input:
        return {"success": False, "error": "Missing captcha_id or text", "status_code": 400}

    captcha = active_captchas.get(captcha_id)
    if not captcha:
        return {"success": False, "error": "Invalid or expired CAPTCHA", "status_code": 404}

    # Check expiration
    current_time = int(time.time() * 1000)
    if current_time > captcha["expires_at"]:
        del active_captchas[captcha_id]
        return {"success": False, "error": "CAPTCHA has expired", "status_code": 410}

    # Check attempts
    if captcha["attempts"] >= captcha["max_attempts"]:
        del active_captchas[captcha_id]
        return {"success": False, "error": "Too many attempts", "status_code": 429}

    captcha["attempts"] += 1

    # Check if already used
    if captcha["used"]:
        return {"success": False, "error": "CAPTCHA already used", "status_code": 409}

    # Verify
    if user_input == captcha["text"]:
        captcha["used"] = True
        return {"success": True, "message": "CAPTCHA verified successfully", "status_code": 200}
    else:
        remaining = captcha["max_attempts"] - captcha["attempts"]
        result = {
            "success": False,
            "error": "Incorrect CAPTCHA text",
            "attempts_remaining": remaining,
            "status_code": 200,
        }
        if with_hints:
            result["hints"] = _compute_hints(captcha["text"], user_input)
        if captcha["attempts"] >= captcha["max_attempts"]:
            del active_captchas[captcha_id]
        return result


def cleanup_expired_captchas():
    """Remove expired CAPTCHAs from memory."""
    current_time = int(time.time() * 1000)
    expired = [cid for cid, c in active_captchas.items() if current_time > c["expires_at"]]
    for cid in expired:
        del active_captchas[cid]
    return len(expired)
