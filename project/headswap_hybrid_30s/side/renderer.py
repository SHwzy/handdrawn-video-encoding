"""Independent two-layer side close-up. Source frames [154, 182), 30 fps.

API: render(local_frame, variant=0) -> uint8 BGR (540,960,3).
No source video, optical flow, patch filling or per-frame character synthesis.
The active viewport uses Python slice [53:487] (rows 53 through 486).
"""
from functools import lru_cache
from pathlib import Path
import math
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 960, 540
ACTIVE_TOP, ACTIVE_BOTTOM = 53, 487
ACTIVE_HEIGHT = ACTIVE_BOTTOM - ACTIVE_TOP
FRAME_COUNT = 28
BG_RIGHT_TRAVEL = 96.0
FG_SCALE = 0.485
FG_BOB_AMPLITUDE = 5.0

def _read(path, flags):
    data = np.fromfile(str(path), np.uint8)
    result = cv2.imdecode(data, flags)
    if result is None:
        raise RuntimeError(f"Cannot load asset {path}")
    return result

@lru_cache(maxsize=1)
def _assets():
    background = _read(ROOT / "raw/background_v1.png", cv2.IMREAD_COLOR)
    bh, bw = background.shape[:2]
    background = cv2.resize(background, (round(bw * ACTIVE_HEIGHT / bh), ACTIVE_HEIGHT), interpolation=cv2.INTER_AREA)
    foreground = _read(ROOT / "raw/foreground_v1.png", cv2.IMREAD_UNCHANGED)
    if foreground.ndim != 3 or foreground.shape[2] != 4:
        raise RuntimeError("Foreground must preserve genuine RGBA alpha.")
    a = foreground[:, :, 3:4].astype(np.float32) / 255.0
    # Premultiply before every filter/transform; zero-alpha RGB cannot halo.
    foreground = np.concatenate((foreground[:, :, :3].astype(np.float32) * a, a), axis=2)
    return background, foreground

def motion(local_frame):
    f = float(np.clip(local_frame, 0.0, FRAME_COUNT - 1))
    progress = f / (FRAME_COUNT - 1)
    return {
        "background_right_px": BG_RIGHT_TRAVEL * progress,
        "foreground_y_px": FG_BOB_AMPLITUDE * math.sin(2.0 * math.pi * progress),
    }

def render(local_frame, variant=0):
    # variant is an integration-compatible reserved parameter.
    background, foreground = _assets()
    m = motion(local_frame)
    excess = background.shape[1] - WIDTH
    if excess < BG_RIGHT_TRAVEL + 2:
        raise RuntimeError("Insufficient independent panorama coverage.")
    start_crop_x = excess / 2.0 + BG_RIGHT_TRAVEL / 2.0
    crop_x = start_crop_x - m["background_right_px"]
    bg_matrix = np.float32([[1, 0, -crop_x], [0, 1, 0]])
    bg = cv2.warpAffine(background, bg_matrix, (WIDTH, ACTIVE_HEIGHT),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    fg_y = (ACTIVE_HEIGHT - foreground.shape[0] * FG_SCALE) / 2.0 + m["foreground_y_px"]
    fg_matrix = np.float32([[FG_SCALE, 0, 0], [0, FG_SCALE, fg_y]])
    fg = cv2.warpAffine(foreground, fg_matrix, (WIDTH, ACTIVE_HEIGHT),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = fg[:, :, 3:4]
    active = np.clip(np.rint(fg[:, :, :3] + bg.astype(np.float32) * (1.0 - alpha)), 0, 255).astype(np.uint8)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[ACTIVE_TOP:ACTIVE_BOTTOM] = active
    return frame

if __name__ == "__main__":
    import json
    qa = ROOT / "qa"
    qa.mkdir(exist_ok=True)
    frames = [render(i) for i in range(FRAME_COUNT)]
    for i in (0, 7, 14, 20, 27):
        cv2.imencode(".png", frames[i])[1].tofile(str(qa / f"frame_{i:02d}.png"))
    contact = np.concatenate([
        np.concatenate([frames[0], frames[7]], axis=1),
        np.concatenate([frames[20], frames[27]], axis=1)
    ], axis=0)
    cv2.imencode(".jpg", contact, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(str(qa / "contact.jpg"))
    bg, fg = _assets()
    excess = bg.shape[1] - WIDTH
    report = {
        "frames": FRAME_COUNT, "fps": 30, "size": [WIDTH, HEIGHT],
        "source_interval_exclusive": [154, 182],
        "active_rows_exclusive": [ACTIVE_TOP, ACTIVE_BOTTOM],
        "panorama_render_size": [bg.shape[1], bg.shape[0]],
        "background_right_travel_px": BG_RIGHT_TRAVEL,
        "background_min_horizontal_margin_px": (excess - BG_RIGHT_TRAVEL) / 2.0,
        "foreground_uniform_scale": FG_SCALE,
        "foreground_bob_amplitude_px": FG_BOB_AMPLITUDE,
        "foreground_min_vertical_margin_px": (fg.shape[0] * FG_SCALE - ACTIVE_HEIGHT)/2.0 - FG_BOB_AMPLITUDE,
        "all_frames_bgr_uint8": all(f.shape == (HEIGHT, WIDTH, 3) and f.dtype == np.uint8 for f in frames),
        "black_bars_correct": all(not f[:ACTIVE_TOP].any() and not f[ACTIVE_BOTTOM:].any() for f in frames),
        "no_fully_black_pixels_in_active_viewport": all(np.all(np.any(f[ACTIVE_TOP:ACTIVE_BOTTOM] > 0, axis=2)) for f in frames),
        "alpha_has_transparency": bool(fg[:,:,3].min() == 0 and fg[:,:,3].max() == 1),
        "lyrics": "Intentionally absent; root can overlay separately.",
        "variant": "Reserved; same deterministic two-layer renderer for every variant.",
        "motion": [motion(i) for i in range(FRAME_COUNT)]
    }
    (qa / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    video = cv2.VideoWriter(str(qa / "preview_silent.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (WIDTH, HEIGHT))
    for frame in frames: video.write(frame)
    video.release()
    print(json.dumps({k:v for k,v in report.items() if k != "motion"}, ensure_ascii=False, indent=2))

