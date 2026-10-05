"""Deterministic layered running shots. No source frames are read at render time.

API: render(local_frame, variant=0) -> uint8 BGR 960x540.
Frames 0..26 map to source [71,98) or [235,262) at 30 fps.
"""
from functools import lru_cache
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 960, 540
ACTIVE_TOP, ACTIVE_BOTTOM = 53, 487
ACTIVE_HEIGHT = ACTIVE_BOTTOM - ACTIVE_TOP
FPS = 30
FRAME_COUNT = 27
PARAMETERS = (
    dict(character_scale=0.47, hip=(580.0, 204.0), bg_x=-60.0,
         bg_y=-95.0, bg_speed=1.15, fg_scale=0.80, fg_x=-68.0,
         fg_y=-110.0, fg_speed=10.0, bob_amplitude=3.0, phase=0),
    dict(character_scale=0.65, hip=(675.0, 250.0), bg_x=-88.0,
         bg_y=-95.0, bg_speed=1.15, fg_scale=0.82, fg_x=-45.0,
         fg_y=-130.0, fg_speed=10.5, bob_amplitude=3.5, phase=2),
)
SPRITE_HIP = (875.0, 435.0)

def _read_rgb(name):
    return np.asarray(Image.open(ROOT / 'assets' / name).convert('RGB'))

def _read_premultiplied(name, scale):
    """Premultiply before all resampling to prevent hidden RGB bleeding."""
    rgba = np.asarray(Image.open(ROOT / 'assets' / name).convert('RGBA'), dtype=np.float32) / 255.0
    rgba[..., :3] *= rgba[..., 3:4]
    size = (round(rgba.shape[1] * scale), round(rgba.shape[0] * scale))
    return cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)

@lru_cache(maxsize=2)
def _assets(variant):
    p = PARAMETERS[variant]
    bg = _read_rgb('background.png')
    # A complete generated background is uniformly scaled and cropped, never patched.
    bg = cv2.resize(bg, (1110, round(bg.shape[0] * 1110 / bg.shape[1])), interpolation=cv2.INTER_AREA)
    poses = [_read_premultiplied(name, p['character_scale']) for name in ('pose_a.png', 'pose_b.png')]
    fg = _read_premultiplied('foreground.png', p['fg_scale'])
    return bg, poses, fg

def _over(canvas, premultiplied_rgba, x, y):
    """Translate only; compose a scaled cached RGBA layer over float RGB."""
    matrix = np.array([[1.0, 0.0, float(x)], [0.0, 1.0, float(y)]], dtype=np.float32)
    layer = cv2.warpAffine(premultiplied_rgba, matrix, (WIDTH, ACTIVE_HEIGHT),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))
    canvas *= 1.0 - layer[..., 3:4]
    canvas += layer[..., :3]

def state(local_frame, variant=0):
    """Inspect analytic motion without rendering, using shot-local frame indices."""
    variant = int(variant)
    if variant not in (0, 1):
        raise ValueError('variant must be 0 or 1')
    f = max(0.0, min(float(local_frame), FRAME_COUNT - 1.0))
    p = PARAMETERS[variant]
    phase = f + p['phase']
    pose = (int(phase) // 4) % 2
    bob = -p['bob_amplitude'] * math.sin(2.0 * math.pi * phase / 8.0)
    return dict(frame=f, variant=variant, pose=pose, bob=bob,
                bg_x=p['bg_x'] - p['bg_speed'] * f,
                fg_x=p['fg_x'] - p['fg_speed'] * f,
                character_x=p['hip'][0] - SPRITE_HIP[0] * p['character_scale'],
                character_y=p['hip'][1] - SPRITE_HIP[1] * p['character_scale'] + bob)

def render(local_frame, variant=0):
    """Return BGR uint8 frame including black letterbox bars at y<53 and y>=487."""
    s = state(local_frame, variant)
    p = PARAMETERS[s['variant']]
    bg, poses, fg = _assets(s['variant'])
    matrix = np.array([[1.0, 0.0, s['bg_x']], [0.0, 1.0, p['bg_y']]], dtype=np.float32)
    rgb = cv2.warpAffine(bg, matrix, (WIDTH, ACTIVE_HEIGHT), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 255))
    canvas = rgb.astype(np.float32) / 255.0
    _over(canvas, poses[s['pose']], s['character_x'], s['character_y'])
    _over(canvas, fg, s['fg_x'], p['fg_y'])
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[ACTIVE_TOP:ACTIVE_BOTTOM] = np.clip(canvas[..., ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return frame

if __name__ == '__main__':
    from verify import main
    main()
