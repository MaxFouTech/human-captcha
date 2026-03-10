# Motion CAPTCHA

> **Experimental** — This is a research project, not a production-ready solution. The hardened mode resists all attacks we've tested so far, but that doesn't mean it's unbreakable.

An experimental CAPTCHA that exploits **persistence of vision** to distinguish humans from bots. Text is hidden in animated binary noise — each frame looks like random static, but the human visual system integrates motion over time and reveals the text. A bot seeing individual frames (or even all frames) sees only noise — cracking it requires running targeted analysis scripts with knowledge of the underlying technique.

<p align="center">
  <img src="demo.gif" alt="Can you read the hidden word?" width="600">
  <br>
  <em>You can read this thanks to persistence of vision. To a vision model, it's just noise.</em>
</p>

## How it works

Text and background are both random binary noise, scrolling in different directions. The human visual system automatically separates the two layers through differential motion perception — the same principle that makes cinema work.

## The demo

The web demo shows three modes side by side:

| Mode | Text position | Frame hiding | Security |
|------|--------------|--------------|----------|
| **Static** | Fixed center | No | A vision-only agent sees noise, but targeted scripts can extract the text since it never moves |
| **Moving** | Lemniscate path | No | Harder — but crackable via sliding window optical flow attack |
| **Hardened** | Lemniscate path | Every other frame | Resists all tested attacks |

## Attacks we tried

We built an attack pipeline and tested multiple techniques against the moving variant:

- **Temporal variance** — per-pixel variance across frames
- **Optical flow segmentation** — separate horizontal vs vertical motion
- **Sliding window flow ratio** — `flow_y / (flow_x + flow_y)` over short frame windows. Most effective attack — consistently cracks the moving variant
- **Phase correlation alignment** — align frames via frequency domain
- **Temporal PCA** — principal components of pixel time series

### Hardening attempts

The sliding window attack needs consistent text in consecutive frame pairs. We tried many approaches before finding one that works:

| Approach | Effect |
|----------|--------|
| Speed tuning | No impact — direction ratio still separates layers |
| Perlin-based direction variation | Inconsistent — depends on noise seed |
| Deterministic direction alternation | No improvement |
| Single shared noise texture | No impact — flow measures motion, not texture |
| Background direction rotation | Marginal improvement |
| Pixel flips (3-15%) | Effective at 15% but hurts human readability |
| **Frame hiding (every other frame)** | **Defeats the attack while remaining human-readable** |

Frame hiding breaks every consecutive frame pair, dropping attack accuracy from 6/6 to ~1/6 correct characters.

## Run it

```bash
uv sync
uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Run the attack

```bash
uv run --no-project \
  --with opencv-python-headless --with numpy --with httpx --with pillow \
  python attack.py --count 5
```

Results are saved to `attack_results/` with extracted images for each technique.
