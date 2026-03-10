"""Generate a demo GIF of the static captcha for the README."""

import sys
sys.path.insert(0, ".")

from captcha.generator import generate_captcha, CAPTCHA_CONFIG
from captcha.optimized_core import warm_up_jit
from PIL import Image

TARGET_FPS = 50  # GIF max reliable rate (20ms minimum delay)
PIXEL_SIZE = 2

def main():
    warm_up_jit()

    # Generate a static captcha with a fixed seed for reproducibility
    captcha_id, text, metadata, packed = generate_captcha(seed=42, mode="static", text="HELLO")

    width = metadata["width"]
    height = metadata["height"]
    frame_count = metadata["total_frame_count"]
    frame_packed_size = metadata["frame_packed_size"]

    # Unpack binary frames
    import numpy as np
    packed_bytes = np.frombuffer(packed, dtype=np.uint8)
    frames = []
    for f in range(frame_count):
        frame = np.zeros((height, width), dtype=np.uint8)
        offset = f * frame_packed_size
        for y in range(height):
            for x in range(width):
                bit_pos = y * width + x
                byte_pos = offset + (bit_pos >> 3)
                bit_offset = 7 - (bit_pos & 7)
                frame[y, x] = ((packed_bytes[byte_pos] >> bit_offset) & 1) * 255
        frames.append(frame)

    # Subsample frames to target FPS
    source_fps = CAPTCHA_CONFIG["fps"]
    step = max(1, round(source_fps / TARGET_FPS))
    selected = frames[::step]

    # Colorize: black pixels -> dark green (#5A7052), white -> white
    pil_frames = []
    for frame in selected:
        rgb = np.zeros((height * PIXEL_SIZE, width * PIXEL_SIZE, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if frame[y, x] == 0:
                    color = (90, 112, 82)  # dark green
                else:
                    color = (255, 255, 255)
                y0, x0 = y * PIXEL_SIZE, x * PIXEL_SIZE
                rgb[y0:y0+PIXEL_SIZE, x0:x0+PIXEL_SIZE] = color
        pil_frames.append(Image.fromarray(rgb))

    # Duplicate frames to extend the animation (loop source frames 5x)
    extended = pil_frames * 5

    # Save as optimized GIF
    duration = int(1000 / TARGET_FPS)
    extended[0].save(
        "demo.gif",
        save_all=True,
        append_images=extended[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )

    size_kb = __import__("os").path.getsize("demo.gif") / 1024
    print(f"Saved demo.gif ({len(selected)} frames @ {TARGET_FPS}fps, {size_kb:.0f}KB)")
    print(f"Captcha text: {text}")

if __name__ == "__main__":
    main()
