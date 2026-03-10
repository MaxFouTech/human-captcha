# Motion CAPTCHA

An experimental CAPTCHA that exploits **persistence of vision** — a property of human perception that AI cannot replicate.

<p align="center">
  <img src="demo.gif" alt="Can you read the hidden word?" width="600">
  <br>
  <em>Can you read it? Your brain can. An algorithm can't.</em>
</p>

## How it works

Text and background are both random binary noise, scrolling in different directions. Each individual frame is indistinguishable from random static. But the human visual system integrates motion over time, automatically separating the two layers — revealing the text through differential motion perception.

This is the same principle that makes cinema work: your brain fuses rapidly changing frames into coherent moving objects.

## The demo

The web demo shows three difficulty levels side by side:

| Mode | Text position | Frame hiding | Crackable? |
|------|--------------|--------------|------------|
| **Static** | Fixed center | No | Trivially — text never moves |
| **Moving** | Lemniscate path | No | Yes — sliding window attack |
| **Hardened** | Lemniscate path | Every other frame | Not yet |

## Attacks we tried

We built an automated attack pipeline and tested multiple techniques:

- **Temporal variance** — per-pixel variance across frames. Defeated by shared noise texture.
- **Optical flow segmentation** — separate horizontal vs vertical motion. Works on the moving variant.
- **Sliding window flow ratio** — compute `flow_y / (flow_x + flow_y)` over short frame windows. The most effective attack — consistently cracks the moving variant on the first try.
- **Phase correlation alignment** — align frames via frequency domain. Partially effective.
- **Temporal PCA** — principal components of pixel time series. Noisy results.

### What defeated the sliding window

The sliding window attack needs consistent text pixels in consecutive frame pairs to estimate optical flow. The hardened mode hides text on odd frames, breaking every frame pair. Result: attack accuracy dropped from 6/6 to ~1/6 correct characters.

| Approach | Effect |
|----------|--------|
| Speed tuning | No impact — direction ratio still separates layers |
| Perlin-based direction variation | Inconsistent — depends on noise seed |
| Deterministic direction alternation | No improvement |
| Single shared noise texture | No impact — flow measures motion, not texture |
| Background direction rotation | Marginal improvement |
| Pixel flips (3-15%) | Effective at 15% but hurts human readability |
| **Frame hiding (every other frame)** | **Defeats the attack while remaining human-readable** |

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

## Status

Experimental. This is a research project exploring whether persistence of vision can serve as a reliable human-verification primitive. The hardened mode resists all attacks we've tested so far, but that doesn't mean it's unbreakable.
