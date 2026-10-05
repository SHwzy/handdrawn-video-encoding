from pathlib import Path
import json,cv2,numpy as np,subprocess,functools,argparse
from PIL import Image
import render as base
P=Path(__file__).resolve().parent;V=P/'v2';O=P/'output';QA=V/'qa';ROOT=P.parent
W,H=960,540;X,Y=base.X,base.Y
stable=json.loads((V/'camera_stable.json').read_text());mouthreg=json.loads((V/'mouth_registration.json').read_text())
original_cam=base.cam
def camera(name,n):
 if name in ('witch','crowd'):
  s,cx,cy=[np.interp(n,stable['frames'],np.array(stable['values'])[:,i]) for i in range(3)]
  return base.mat(s,cx-480*s,cy-270*s)
 return original_cam(name,n)
base.cam=camera
@functools.lru_cache(None)
def image(name):return np.array(Image.open(V/'assets'/f'{name}.png').convert('RGB')).astype(np.float32)
@functools.lru_cache(None)
def clip(name):
 cap=cv2.VideoCapture(str(V/'h3'/f'{name}_h3_v2.mp4'));out=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  out.append(cv2.cvtColor(cv2.resize(f,(960,540)),cv2.COLOR_BGR2RGB).astype(np.float32))
 return out
def mouth(n):
 j=round((n-3427)/51*55);src_cam=camera('mouth',n)
 reg=base.mat(.64,177,-133)
 bg=base.warp(image('mouth_smile'),base.mul(src_cam,reg),border=(243,242,243))
 M=np.array(mouthreg[j]['matrix'],np.float32)
 patch=base.warp(clip('mouth')[j],base.mul(src_cam,M),border=(243,242,243))
 center=base.mul(src_cam,reg)@np.array([474,520,1])
 dist=((X-center[0])/25)**2+((Y-center[1])/17)**2;ring=(dist>2)&(dist<3.5)
 patch+=np.median(bg[ring],axis=0)-np.median(patch[ring],axis=0)
 return base.paper(base.mix(bg,patch,np.exp(-dist**2)))
def jump(n):
 # Three H3 Guide frames lock high opening / lowered center / leftward raised reach.
 # Time mapping aligns observed low point and extended-arm arrival to source frames.
 j=round(np.interp(n,[3479,3481,3483,3485,3487,3489,3491,3493,3494,3505],[0,5,10,15,22,27,31,36,40,55]))
 return clip('jump')[j].copy()
def cage(n):
 dy=-(n-3330)*.99;cb=base.mat(1.3,-144,-155+dy*.89)
 bg=base.plate('cage_bg',cb);ghost=base.layer('witch_v2',.42,157,25+dy)
 gm=ghost[:,:,3:4]/255*np.clip((bg.mean(2)-160)/45,0,1)[:,:,None]*((X>330)&(X<640))[:,:,None]
 bg=bg*(1-gm)+np.array([231,94,137])*gm
 bg=base.over(bg,base.layer('cage_maid',.485,231,-35+dy))
 if n<3375:return bg
 t=n-3375
 top=np.interp(n,[3375,3380,3390,3400,3426],[247,178,74,38,18])+.13*X+7*np.sin(X/170+t*.025)
 bot=np.interp(n,[3375,3380,3390,3400,3426],[264,288,324,335,353])+.13*X+5*np.sin(X/220+t*.02)
 m=np.clip(Y-top,0,1)*np.clip(bot-Y,0,1)
 # Measured source displacement relative to f3400. Fixed scale; no spurious rotation.
 dx=np.interp(n,[3375,3382,3389,3396,3403,3410,3417,3426],[-47,-24.4,-11.7,-3.2,1.6,7.1,10.1,14.1])
 alternate=((n-3375)//2)%2
 a=image('closeup_eye_b') if alternate else base.asset('closeup')[:,:,:3]
 s=.87*960/a.shape[1];M=base.mat(s,35+dx,-21+np.interp(t,[0,20,51],[20,0,-4]))
 close=base.paper(base.warp(a,M,border=(242,241,242)))
 return base.mix(bg,close,m)
base.mouth=mouth;base.jump=jump;base.cage=cage
def frame(n):return base.frame(n)
def previews():
 ns=[3105,3112,3119,3204,3375,3382,3389,3400,3426,3433,3445,3457,3475,3479,3483,3487,3491,3494,3505]
 sheet=np.zeros((((len(ns)+2)//3)*294,1440,3),np.uint8)
 for i,n in enumerate(ns):
  f=frame(n);Image.fromarray(f).save(QA/f'f{n}.png');x=i%3*480;y=i//3*294
  sheet[y+24:y+294,x:x+480]=cv2.resize(f,(480,270));cv2.putText(sheet,f'{n} | {(n-2664)/30:.3f}s',(x+5,y+18),0,.5,(255,255,255),1)
 Image.fromarray(sheet).save(QA/'preview_v2.jpg')
def render():
 proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(O/'silent_v2.mp4')],stdin=subprocess.PIPE)
 cap=cv2.VideoCapture(str(next(ROOT.glob('*.mp4'))));cap.set(1,2664)
 for n in range(2664,3540):
  ok,orig=cap.read();assert ok
  f=frame(n) if (3061<=n<3073) or (3101<=n<3506) else cv2.cvtColor(cv2.resize(orig,(960,540),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
  proc.stdin.write(f.tobytes())
  if n%150==0:print('frame',n,flush=True)
 proc.stdin.close();assert proc.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(O/'silent_v2.mp4'),'-ss','88.8','-i',str(next(ROOT.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t','29.2','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(O/'segment_88p8_118_v2.mp4')],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
 if args.preview:previews()
 else:render()
