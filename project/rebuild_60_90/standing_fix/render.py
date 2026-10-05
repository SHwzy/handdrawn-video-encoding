"""Standing shot source [2034,2102): shared camera and grounded contact."""
from pathlib import Path
import sys,importlib.util,json,subprocess,hashlib
import cv2,numpy as np
from PIL import Image
P=Path(__file__).resolve().parent; ROOT=P.parents[1]
spec=importlib.util.spec_from_file_location('base',P.parent/'render.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
OUT=P/'output';OUT.mkdir(exist_ok=True)
rows=json.loads((P/'motion.json').read_text())
# The scale/rotation estimates are subpixel noise: this shot is a vertical pan.
nodes=[0,3,6,10,15,21,28,36,44,51,59,67]
dy=np.array([rows[k]['matrix'][1][2] for k in nodes])
Y,X=np.mgrid[:434,:960].astype(np.float32)
def camera(n):return float(np.interp(n-2034,nodes,dy))
def frame(n):
    shift=camera(n);wy=Y-shift
    # Separate sea and sand depths. Map the existing independent plate onto
    # the planned ground view, keeping the shoreline behind the contact plane.
    sy=np.interp(wy,[-200,100,275,310,510],[0,350,730,790,845]).astype(np.float32)
    sx=(X*1859/959).astype(np.float32)
    plate=base.asset('sea_v1')[:,:,:3]*255
    bg=cv2.remap(plate,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    assert sy.min()>=0 and sy.max()<=845
    # Ground shadows are projected ellipses at actual sole anchors, not
    # painted onto the sprite; the pan drives all contact layers identically.
    for cx,cy,rx,ry,opacity in [(262,386,58,7,.20),(700,386,56,8,.24),
                                 (245,384,17,2.5,.16),(291,384,17,2.5,.16),(697,384,27,3,.22)]:
        a=opacity*np.exp(-.5*((X-cx)/rx)**2-.5*((wy-cy)/ry)**2)
        bg=bg*(1-a[:,:,None])+np.array([107,91,104])*a[:,:,None]
    bg=base.layer(bg,'sakiko_stand_v1',(580,1512),(713,384+shift),.325)
    bg=base.layer(bg,'anon_stand_v1',(470,1520),(280,384+shift),.333)
    bg=base.caption(bg,'standing',True)
    out=np.zeros((540,960,3),np.uint8);out[53:487]=np.clip(bg,0,255)
    return cv2.cvtColor(out,cv2.COLOR_RGB2BGR)
def main():
    src=cv2.VideoCapture(str(base.SRC));old=cv2.VideoCapture(str(P.parent/'output/approved_60_88p8_v4.mp4'))
    selected=[2034,2040,2055,2070,2085,2101];sheets=[]
    cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','960x540','-r','30','-i','pipe:0','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p',str(OUT/'standing_silent.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for n in range(2034,2102):
        a=frame(n);proc.stdin.write(a.tobytes())
        if n in selected:
            src.set(1,n);_,s=src.read();old.set(1,n-1800);_,b=old.read()
            row=np.hstack([cv2.resize(x,(480,270)) for x in [s,b,a]])
            cv2.putText(row,f'{n} source / baseline / fixed',(12,22),0,.65,(255,255,255),1)
            sheets.append(row);cv2.imencode('.jpg',a)[1].tofile(str(P/f'fixed_{n}.jpg'))
    proc.stdin.close();assert proc.wait()==0
    cv2.imencode('.jpg',np.vstack(sheets))[1].tofile(str(P/'comparison.jpg'))
    src.release();old.release()
if __name__=='__main__':main()
