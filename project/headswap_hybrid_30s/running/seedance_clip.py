"""Selected Seedance 2.0 running take, cut to the source shot's 27 frames."""
from functools import lru_cache
from pathlib import Path
import cv2

VIDEO = Path(__file__).resolve().parents[2] / 'seedance_running_trial' / 'seedance20_running_0p9s.mp4'


@lru_cache(maxsize=1)
def frames():
    cap = cv2.VideoCapture(str(VIDEO))
    result = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result.append(frame)
    cap.release()
    if len(result) != 27 or any(frame.shape != (540, 960, 3) for frame in result):
        raise ValueError(f'Expected 27 frames of 960x540 in {VIDEO}; got {len(result)}')
    return result


def render(local_frame, variant=0):
    if not 0 <= int(local_frame) < 27:
        raise ValueError(local_frame)
    return frames()[int(local_frame)].copy()
