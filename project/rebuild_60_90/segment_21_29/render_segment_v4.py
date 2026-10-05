from pathlib import Path
import sys,importlib.util,json,functools,math,subprocess,argparse
import cv2,numpy as np
from PIL import Image
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(P.parent))
import render as base
sp=importlib.util.spec_from_file_location('prior',P.parent/'segment_05_21/render_sequence_v5.py');prior=importlib.util.module_from_spec(sp);sp.loader.exec_module(prior)
W,H,AH=960,540,434;OUT=P/'output';QA=P/'qa_v4';QA.mkdir(exist_ok=True)
raw=json.loads((P/'motion_raw.json').read_text());reg=json.loads((P/'registration.json').read_text())
Y,X=np.mgrid[:AH,:W].astype(np.float32)

def camera(n,name):
 rows=raw[name]['rows'];good=[r for r in rows if r.get('inliers',0)>=10 and .75<r.get('scale',0)<2.2 and abs(r.get('angle',99))<.4 and 460<r.get('cx',0)<500]
 ns=[r['frame'] for r in good]
 d={k:float(np.interp(n,ns,[r[k] for r in good])) for k in ['scale','angle','cx','cy']}
 if name=='grab' and n<2559:d.update(scale=1.,cx=480.,cy=float(np.interp(n,[2547,2559],[470,339])))
 if name=='tear' and n<2610:d.update(scale=1.,cx=480.,cy=float(np.interp(n,[2601,2610],[50,222])))
 return d

def cmatrix(n,name,background=False):
 c=camera(n,name);s=1+(c['scale']-1)*(.65 if background else 1)
 m=cv2.getRotationMatrix2D((480,270),c['angle'],s);m[:,2]+=np.array([c['cx']-480,c['cy']-270])
 return m

@functools.lru_cache(None)
def sprite(name):
 file='hug_extended.png' if name=='hug' else name+'.png'
 p=np.array(Image.open(P/'assets'/file).convert('RGBA')).astype(np.float32)/255;p[:,:,:3]*=p[:,:,3:4]
 return cv2.resize(p,(960,540),interpolation=cv2.INTER_AREA)

def mul(a,b):return (np.vstack([a,[0,0,1]])@np.vstack([b,[0,0,1]]))[:2]
def char(n,mode=0):
 p=sprite('hug').copy();alpha=p[:,:,3:4];rgb=p[:,:,:3]/np.maximum(alpha,.001)
 if mode==1:
  p[:,:,:3]=np.where((rgb.mean(2)<.3)[:,:,None],rgb,rgb*.68)*alpha
 elif mode==2:
  yy,xx=np.mgrid[:540,:960];dark=(rgb.mean(2)<.43)[:,:,None]
  dots=((((xx%12)-6)**2+((yy%12)-6)**2)<4)[:,:,None]
  motif=((np.abs(xx%72-36)<2)|(np.abs(yy%72-36)<2)|(np.abs(np.abs(xx%72-36)-np.abs(yy%72-36))<2))[:,:,None]
  maid=np.where((yy<245)[:,:,None],np.array([238,111,70]),np.array([233,199,101]))/255*(1-.24*dots)
  witch=np.array([222,73,124])/255*(1-.34*motif)
  p[:,:,:3]=np.where(dark,maid,witch)*alpha
 m=mul(cmatrix(n,'hug'),np.array(reg['hug']['matrix']));m[1,2]-=53
 return cv2.warpAffine(p,m,(W,AH),flags=cv2.INTER_LINEAR)

@functools.lru_cache(None)
def gridplate(kind,dark=False):
 h,w=1400,1800;bg=np.full((h,w,3),(110,110,121) if dark else (237,235,238),np.float32)
 color=(213,211,220) if dark else (91,91,103);step=140
 for y in range(-step,h+step,step):
  for x in range(-step,w+step,step):
   if kind%3==0:
    cv2.circle(bg,(x+70,y+70),70,color,1,cv2.LINE_AA);cv2.rectangle(bg,(x+20,y+20),(x+120,y+120),color,1,cv2.LINE_AA)
   elif kind%3==1:
    cv2.polylines(bg,[np.int32([[x+70,y],[x+140,y+70],[x+70,y+140],[x,y+70]])],True,color,1,cv2.LINE_AA);cv2.rectangle(bg,(x+28,y+28),(x+112,y+112),color,1,cv2.LINE_AA)
   else:
    for dx,dy in [(35,0),(-35,0),(0,35),(0,-35)]:cv2.circle(bg,(x+70+dx,y+70+dy),45,color,1,cv2.LINE_AA)
 return bg

def plate(n,name,kind=0,dark=False):
 p=gridplate(kind,dark);c=camera(n,name);s=(1+(c['scale']-1)*.65)*.83
 m=cv2.getRotationMatrix2D((900,700),c['angle'],s);m[:,2]+=np.array([c['cx'],c['cy']-53])-np.array([900,700])
 return cv2.warpAffine(p,m,(W,AH))

def sea(n):
 c=camera(n,'hug');return base.background('sea_wide_v1',0,zoom=1.12+(c['scale']-1)*.55,dx=c['cx']-480,dy=c['cy']-270,angle=c['angle'])

def errors(n):
 rng=np.random.default_rng(991+(n//4))
 def noise(h,w):return cv2.resize(rng.random((h,w)).astype(np.float32),(W,AH),interpolation=cv2.INTER_CUBIC)
 a=noise(9,18)*.65+noise(30,80)*.23+noise(100,180)*.12
 b=noise(13,24)*.7+noise(55,90)*.3
 palette=np.float32([[160,24,188],[246,33,121],[33,163,105],[50,55,168],[236,167,113]])
 colors=palette[rng.integers(0,5,(8,15))];colors=cv2.resize(colors,(W,AH),interpolation=cv2.INTER_NEAREST);colors=cv2.GaussianBlur(colors,(0,0),9)
 tex=prior.mix(np.full_like(colors,235),colors,np.clip((a-.49)*6,0,.94))
 tex=prior.mix(tex,np.full_like(tex,(38,38,51)),np.clip((b-.59)*6,0,.96))
 grain=rng.random((AH,W,1)).astype(np.float32)
 tex=tex*(.80+.20*grain)
 tex[::2]*=.90
 return tex

def halo(n):
 p=np.full((1400,1800,3),(174,173,182),np.float32);center=(900,505)
 for rad in [140,147,155]:cv2.circle(p,center,rad,(248,247,249),2,cv2.LINE_AA)
 for j in range(26):
  a=j*math.tau/26;r1=148;r2=245+(j%3)*24
  cv2.line(p,(round(center[0]+r1*math.cos(a)),round(center[1]+r1*math.sin(a))),(round(center[0]+r2*math.cos(a)),round(center[1]+r2*math.sin(a))),(249,248,251),3,cv2.LINE_AA)
 rng=np.random.default_rng(56)
 for x,y,r in zip(rng.integers(50,1750,55),rng.integers(0,1400,55),rng.integers(3,17,55)):
  for j in range(5):
   a=j*math.tau/5;cv2.circle(p,(int(x+r*.6*math.cos(a)),int(y+r*.6*math.sin(a))),int(r*.6),(45,46,57),-1,cv2.LINE_AA)
 c=camera(n,'hug');s=1+(c['scale']-1)*.65;m=cv2.getRotationMatrix2D((900,700),c['angle'],s);m[:,2]+=np.array([c['cx'],c['cy']-53])-np.array([900,700])
 return cv2.warpAffine(p,m,(W,AH))

def embrace(n):
 p0=char(n);p1=char(n,1);p2=char(n,2)
 bg0=errors(n);bg1=plate(n,'hug',((n-2443)//12)%3)
 a=prior.reveal(n,2446,2460,81,'left');b=prior.reveal(n,2470,2483,83,'down');c=prior.reveal(n,2498,2518,85)
 bg=prior.mix(bg0,bg1,a);fg=prior.mix(p0,p1,a)
 bg=prior.mix(bg,halo(n),b);fg=prior.mix(fg,p2,prior.reveal(n,2471,2485,83,'down'))
 bg=prior.mix(bg,sea(n),c);fg=prior.mix(fg,p0,prior.reveal(n,2499,2519,85))
 if n>=2525:
  amount=min(1,(n-2524)/2);bg=prior.mix(bg,errors(n),amount)
 result=prior.over(bg,fg)
 if n>=2525:
  flash=[.40,.1,.0,.16][(n-2525)%4];result=prior.mix(result,np.full_like(result,242),flash)
 return result

@functools.lru_cache(None)
def video(name):
 cap=cv2.VideoCapture(str(P/'h3'/f'{name}.mp4'));out=[]
 while True:
  ok,im=cap.read()
  if not ok:break
  out.append(cv2.cvtColor(cv2.resize(im,(W,H)),cv2.COLOR_BGR2RGB).astype(np.float32)/255)
 assert len(out)>40
 return out

def key(rgb):
 # Ordinary green-screen foreground alpha, independent of scene reveal effects.
 excess=rgb[:,:,1]-np.maximum(rgb[:,:,0],rgb[:,:,2]);alpha=1-np.clip((excess-.06)/.35,0,1)
 alpha=cv2.GaussianBlur(alpha,(3,3),.45)
 clean=rgb.copy();clean[:,:,1]=np.minimum(clean[:,:,1],np.maximum(clean[:,:,0],clean[:,:,2])+.025)
 return np.dstack([clean*alpha[:,:,None],alpha]).astype(np.float32)

@functools.lru_cache(None)
def tear_canvas():
 p=np.array(Image.open(P/'assets/tear_user_v4.png').convert('RGBA')).astype(np.float32)/255;p[:,:,:3]*=p[:,:,3:4]
 p=cv2.resize(p,(960,round(p.shape[0]*960/p.shape[1])))
 m=np.array(json.loads((P/'tear_user_v4_registration.json').read_text())['matrix']);m[1,2]+=250
 return cv2.warpAffine(p,m,(960,1100))

def animated_tear(n,idx):
 canvas=tear_canvas().copy()
 # Stabilize the H3 face animation, then restore the measured original camera.
 # A local face patch preserves its eyelid/tear animation; the repaired cel
 # supplies opaque foreground hair and the missing camera overscan.
 rec=json.loads((P/'h3/tear_registration.json').read_text())[idx]
 m=np.array(rec['matrix']);aligned=cv2.warpAffine(video('tear')[idx],m,(960,540))
 yy,xx=np.mgrid[:540,:960]
 mask=np.zeros((540,960),np.float32)
 cv2.fillPoly(mask,[np.int32([[304,145],[391,140],[420,189],[406,258],[356,340],[318,283]])],1.)
 mask=cv2.GaussianBlur(mask,(0,0),4)
 # Keep only skin, eye and tear pixels, never the green screen or hair defect.
 mask*=np.clip(1-(aligned[:,:,1]-np.maximum(aligned[:,:,0],aligned[:,:,2]))*5,0,1)
 # Exclude old pink foreground cloth pixels from the retained face animation.
 mask*=np.clip((aligned[:,:,1]-aligned[:,:,2]+.018)/.025,0,1)
 part=canvas[250:790]
 # Retain only the tear animation on exposed white cheek. New fringe/braids
 # and the white face base must never be replaced by the old colored bob.
 base_rgb=part[:,:,:3]/np.maximum(part[:,:,3:4],.001)
 paper=(base_rgb.min(2)>.86)&((base_rgb.max(2)-base_rgb.min(2))<.045)
 mask*=paper.astype(np.float32)
 neutral=np.clip((aligned.min(2)-.55)/.30,0,1)*.15+.82
 animated_ink=np.repeat(neutral[:,:,None],3,axis=2)
 part[:,:,:3]=prior.mix(part[:,:,:3],animated_ink,mask)
 m=cmatrix(n,'tear');m[:,2]-=m[:,1]*250;m[1,2]-=53
 fg=cv2.warpAffine(canvas,m,(W,AH))
 return prior.over(plate(n,'tear',1,True),fg)

def animated(n,name):
 start,end=(2547,2601) if name=='grab' else (2601,2633);k=n-start;q=k/(end-start-1)
 frames=video(name);idx=min(len(frames)-1,round(q*(len(frames)-1)));fg=key(frames[idx])
 # Generated clips carry the large motion. Restore only measured short jitter here.
 c=camera(n,name);dx=c['cx']-480
 m=cv2.getRotationMatrix2D((480,270),c['angle'],1.008);m[:,2]+=np.array([dx,0])
 fg=cv2.warpAffine(fg,m,(W,H))[53:487]
 bg=plate(n,name,0 if name=='grab' else 1,True)
 result=prior.over(bg,fg)
 if name=='tear':result=animated_tear(n,idx)
 sigma=(3.8 if name=='grab' else 3.2)*max(0,1-k/(12 if name=='grab' else 9))
 if sigma>.1:result=cv2.GaussianBlur(result,(0,0),sigma)
 return result

def frame(n):
 if n>=2633:return cv2.imdecode(np.fromfile(P/'reference'/f'f{n}.png',np.uint8),1)
 if n<2547:bg=embrace(n);shot='embrace';white=False
 else:
  name='grab' if n<2601 else 'tear';bg=animated(n,name);shot='back' if name=='grab' else 'eye';white=True
 return prior.pack(base.caption(bg,shot,white))

def main(preview=False):
 selected=set(range(2443,2664,6))|{2443,2546,2547,2600,2601,2632,2633,2663}
 proc=None;pairs=[]
 if not preview:proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','960x540','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-crf','17','-preset','fast',str(OUT/'silent_v4.mp4')],stdin=subprocess.PIPE)
 for n in range(2443,2664):
  if preview and n not in selected:continue
  im=frame(n);assert im.shape==(540,960,3) and np.isfinite(im).all()
  if proc:proc.stdin.write(im.tobytes())
  if n in selected:
   src=cv2.imdecode(np.fromfile(P/'reference'/f'f{n}.png',np.uint8),1);pair=np.hstack([cv2.resize(src,(480,270)),cv2.resize(im,(480,270))]);pairs.append((n,pair));cv2.imencode('.jpg',im)[1].tofile(str(QA/f'f{n}.jpg'))
 if proc:
  proc.stdin.close();assert proc.wait()==0
  base.ff(['-i',OUT/'silent_v4.mp4','-ss',str(2443/30),'-i',base.SRC,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t',str(221/30),'-movflags','+faststart',OUT/'segment_21_29_v4.mp4'])
  base.ff(['-i',P.parent/'segment_05_21/output/review_00_21_v5.mp4','-i',OUT/'segment_21_29_v4.mp4','-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[v]','-map','[v]','-an','-c:v','libx264','-crf','17',OUT/'full_silent_v4.mp4'])
  base.ff(['-i',OUT/'full_silent_v4.mp4','-ss','60','-i',base.SRC,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t','28.8','-movflags','+faststart',OUT/'review_00_29_v4.mp4'])
  base.ff(['-ss','60','-t','28.8','-i',base.SRC,'-i',OUT/'review_00_29_v4.mp4','-filter_complex','[0:v]scale=960:540,setsar=1[a];[a][1:v]hstack[v]','-map','[v]','-map','1:a:0','-c:v','libx264','-crf','18','-preset','fast','-c:a','copy','-t','28.8','-movflags','+faststart',OUT/'source_vs_review_00_29_v4.mp4'])
 for i in range(0,len(pairs),6):
  batch=pairs[i:i+6];sheet=np.zeros((len(batch)*294,960,3),np.uint8)
  for j,(n,im) in enumerate(batch):
   sheet[j*294+24:(j+1)*294]=im;cv2.putText(sheet,f'{n} | {(n-1800)/30:.3f}s | source / rebuild',(10,j*294+18),0,.5,(255,255,255),1)
  cv2.imencode('.jpg',sheet)[1].tofile(str(QA/f'compare_{i//6}.jpg'))
 print('done',preview)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--preview',action='store_true');main(a.parse_args().preview)
