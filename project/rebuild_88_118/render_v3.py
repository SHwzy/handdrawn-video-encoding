from pathlib import Path
import cv2,numpy as np,subprocess,functools,argparse
from PIL import Image
import render_v2 as prev
base=prev.base;P=prev.P;V=P/'v3';O=P/'output'
oldasset=base.asset;oldcrowd=base.crowd
@functools.lru_cache(None)
def replacement(name):
 a=np.array(Image.open(V/'assets'/f'{name}.png').convert('RGBA')).astype(np.float32)
 a[:,:,3]=np.clip((a[:,:,3]-1)*255/252,0,255);a[:,:,:3]*=a[:,:,3:4]/255
 return a
def crowd(n):
 base.asset=lambda name:replacement('witch' if name=='witch_v2' else name) if name in ('witch_v2','crowd_bg','crowd_vivid') else oldasset(name)
 try:return oldcrowd(n)
 finally:base.asset=oldasset
@functools.lru_cache(None)
def clip():
 c=cv2.VideoCapture(str(V/'h3/jump_v3.mp4'));frames=[]
 while True:
  ok,f=c.read()
  if not ok:break
  frames.append(cv2.cvtColor(cv2.resize(f,(960,540)),cv2.COLOR_BGR2RGB).astype(np.float32))
 return frames
def jump(n):
 j=round(np.interp(n,[3479,3487,3490,3495,3505],[0,17,24,35,55]))
 return clip()[j].copy()
base.crowd=crowd;base.jump=jump
def render():
 proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(O/'silent_v3.mp4')],stdin=subprocess.PIPE)
 cap=cv2.VideoCapture(str(next(P.parent.glob('*.mp4'))));cap.set(1,2664)
 for n in range(2664,3540):
  ok,orig=cap.read();assert ok
  f=base.frame(n) if (3061<=n<3073) or (3101<=n<3506) else cv2.cvtColor(cv2.resize(orig,(960,540),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
  proc.stdin.write(f.tobytes())
  if n%150==0:print(n,flush=True)
 proc.stdin.close();assert proc.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(O/'silent_v3.mp4'),'-ss','88.8','-i',str(next(P.parent.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t','29.2','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(O/'segment_88p8_118_v3.mp4')],check=True)
def preview():
 ns=[3101,3204,3318,3479,3483,3487,3490,3493,3495,3500,3505]
 sheet=np.zeros((4*294,1440,3),np.uint8)
 for i,n in enumerate(ns):
  f=base.frame(n);Image.fromarray(f).save(V/'qa'/f'f{n}.png')
  y=i//3*294;x=i%3*480;sheet[y+24:y+294,x:x+480]=cv2.resize(f,(480,270));cv2.putText(sheet,str(n),(x+5,y+18),0,.5,(255,255,255),1)
 Image.fromarray(sheet).save(V/'qa/preview.jpg')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');a=ap.parse_args()
 if a.preview:preview()
 else:render()
