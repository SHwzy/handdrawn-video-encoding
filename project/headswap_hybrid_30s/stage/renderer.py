"""Stage crowd reveal: source [372, 427), 30fps, independent rigid layers.

render(source_frame) -> uint8 BGR (540,960,3). No optical flow/morph/video input.
The rear plate contains city + pink Anon + stake/chains/log base as one group.
Front left and right anonymous spectators are generated independent RGBA layers.
"""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 960, 540
ACTIVE_TOP, ACTIVE_BOTTOM = 53, 487
ACTIVE_HEIGHT = ACTIVE_BOTTOM - ACTIVE_TOP
SOURCE_START, SOURCE_END = 372, 427
FRONT_SCALE = 0.55

def _read(path, flag):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise RuntimeError(f"Cannot load {path}")
    return image

def _premultiplied(image):
    if image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError("Spectators require genuine RGBA.")
    a = image[:,:,3:4].astype(np.float32) / 255.0
    return np.concatenate([image[:,:,:3].astype(np.float32) * a, a], axis=2)

@lru_cache(maxsize=1)
def _assets():
    background = _read(ROOT / "raw/background_v1.png", cv2.IMREAD_COLOR)
    left = _premultiplied(_read(ROOT / "raw/left_spectator_v1.png", cv2.IMREAD_UNCHANGED))
    right = _premultiplied(_read(ROOT / "raw/right_spectator_v1.png", cv2.IMREAD_UNCHANGED))
    return background, left, right

def motion(source_frame):
    f = float(np.clip(source_frame, SOURCE_START, SOURCE_END - 1))
    if f <= 399:
        u = (f - SOURCE_START) / 27.0
        # Quick initial reveal decelerates into the first composition.
        e = 1.0 - (1.0 - u) ** 1.6
        return {"background_zoom": 1.025 + 0.016*u, "background_crop_y": 94.0 - 32.0*u,
                "left_center_x": 235.0 - 10.0*e, "left_y": -310.0 + 420.0*e,
                "right_center_x": 770.0 + 30.0*e, "right_y": -350.0 + 425.0*e,
                "segment": 0}
    v = (f - 400.0) / 26.0
    # Frame 400 has a deliberate small edit/impulse, then restrained separation.
    return {"background_zoom": 1.053 + 0.015*v, "background_crop_y": 65.0 - 3.0*v,
            "left_center_x": 214.0 - 40.0*v, "left_y": 122.0 + 48.0*v,
            "right_center_x": 808.0 + 42.0*v, "right_y": 87.0 + 48.0*v,
            "segment": 1}

def _render_layers(source_frame):
    background, left, right = _assets()
    m = motion(source_frame)
    s = (WIDTH / background.shape[1]) * m["background_zoom"]
    tx = WIDTH / 2.0 - background.shape[1] * s / 2.0
    rear_matrix = np.float32([[s, 0, tx], [0, s, -m["background_crop_y"]]])
    rear = cv2.warpAffine(background, rear_matrix, (WIDTH, ACTIVE_HEIGHT),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    overlays = []
    # Raw silhouette head-center anchors determined on generated assets, not old-frame masks.
    for fg, anchor_x, center_x, y in [
        (left, 530.0, m["left_center_x"], m["left_y"]),
        (right, 574.0, m["right_center_x"], m["right_y"]),
    ]:
        matrix = np.float32([[FRONT_SCALE, 0, center_x - FRONT_SCALE*anchor_x],
                             [0, FRONT_SCALE, y]])
        overlays.append(cv2.warpAffine(fg, matrix, (WIDTH, ACTIVE_HEIGHT),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return rear, overlays, rear_matrix

def render(source_frame):
    rear, overlays, _ = _render_layers(source_frame)
    active = rear.astype(np.float32)
    for fg in overlays:
        active = fg[:,:,:3] + active * (1.0 - fg[:,:,3:4])
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[ACTIVE_TOP:ACTIVE_BOTTOM] = np.clip(np.rint(active), 0, 255).astype(np.uint8)
    return frame

if __name__ == "__main__":
    import json
    qa = ROOT / "qa"
    qa.mkdir(exist_ok=True)
    frames = [render(f) for f in range(SOURCE_START, SOURCE_END)]
    sample_ids = [372, 385, 399, 400, 413, 426]
    labeled = []
    for f in sample_ids:
        frame = frames[f - SOURCE_START]
        cv2.imencode(".png", frame)[1].tofile(str(qa / f"frame_{f:04d}.png"))
        sample = frame.copy()
        cv2.putText(sample, f"source {f}", (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        labeled.append(sample)
    contact = np.concatenate([np.concatenate(labeled[i:i+2], axis=1) for i in (0,2,4)], axis=0)
    cv2.imencode(".jpg", contact, [cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(qa / "contact.jpg"))
    writer = cv2.VideoWriter(str(qa / "preview_silent.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (WIDTH, HEIGHT))
    for frame in frames: writer.write(frame)
    writer.release()
    bg, left, right = _assets()
    margins = []
    head_visible = []
    body_visible = []
    for f in range(SOURCE_START, SOURCE_END):
        rear, overlays, matrix = _render_layers(f)
        s, tx, ty = float(matrix[0,0]), float(matrix[0,2]), float(matrix[1,2])
        margins.append({"source_frame":f, "left":-tx, "right":bg.shape[1]*s+tx-WIDTH,
                        "top":-ty, "bottom":bg.shape[0]*s+ty-ACTIVE_HEIGHT})
        # Visibility boxes identify central rear character in the generated plate;
        # they do not create or modify compositing masks.
        combined_alpha = 1 - (1-overlays[0][:,:,3])*(1-overlays[1][:,:,3])
        for bounds, target in [((789,239,853,311),head_visible), ((766,311,876,726),body_visible)]:
            x1,y1,x2,y2 = bounds
            x1,x2 = int(x1*s+tx), int(x2*s+tx)
            y1,y2 = max(0,int(y1*s+ty)), min(ACTIVE_HEIGHT,int(y2*s+ty))
            target.append(float((1-combined_alpha[y1:y2,x1:x2]).mean()))
    report = {
        "source_interval_exclusive":[SOURCE_START,SOURCE_END], "frames":len(frames), "fps":30,
        "size":[WIDTH,HEIGHT], "active_rows_exclusive":[ACTIVE_TOP,ACTIVE_BOTTOM],
        "frame400_change":"12px foreground downward edit impulse, increased rear zoom, then slower separation",
        "foreground_uniform_scale":FRONT_SCALE,
        "all_frames_bgr_uint8":all(f.shape==(HEIGHT,WIDTH,3) and f.dtype==np.uint8 for f in frames),
        "black_bars_correct":all(not f[:ACTIVE_TOP].any() and not f[ACTIVE_BOTTOM:].any() for f in frames),
        "no_fully_black_active_pixels":all(np.all(np.any(f[ACTIVE_TOP:ACTIVE_BOTTOM]>0,axis=2)) for f in frames),
        "rear_coverage_min_margin_px": min(min(m[k] for k in ("left","right","top","bottom")) for m in margins),
        "left_rgba_alpha_range":[float(left[:,:,3].min()),float(left[:,:,3].max())],
        "right_rgba_alpha_range":[float(right[:,:,3].min()),float(right[:,:,3].max())],
        "central_pink_head_min_visible_fraction":min(head_visible),
        "central_body_box_min_visible_fraction":min(body_visible),
        "lyrics":"Absent; parent may overlay",
        "outline_only_spectators":"Optional outline-only crowd omitted; retained near crowd is the two anonymous dark silhouettes",
        "motion":[dict(source_frame=f,**motion(f)) for f in range(SOURCE_START,SOURCE_END)],
        "rear_margins":margins,
    }
    (qa / "checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("motion","rear_margins")},ensure_ascii=False,indent=2))

