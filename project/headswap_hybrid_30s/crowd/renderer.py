"""A single continuous panorama and explicit camera windows; no optical flow."""
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
raw=cv2.imdecode(np.fromfile(ROOT/'panorama.png',dtype=np.uint8),cv2.IMREAD_COLOR)
plate=cv2.resize(raw,(round(raw.shape[1]*434/raw.shape[0]),434),interpolation=cv2.INTER_AREA)
max_x=plate.shape[1]-960
def render(source_frame):
    if source_frame<290:
        t=(source_frame-262)/27; q=1.0-.18*t
    elif source_frame<304:
        t=(source_frame-290)/13; q=.54-.12*t
    else:
        t=(source_frame-304)/13;q=.14-.14*t
    x=q*max_x
    # Subpixel camera translation, applied to a static plate without deformation.
    active=cv2.warpAffine(plate,np.float32([[1,0,-x],[0,1,0]]),(960,434),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    out=np.zeros((540,960,3),dtype=np.uint8);out[53:487]=active
    return out
