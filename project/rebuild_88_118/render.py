from pathlib import Path
import cv2,numpy as np,json,subprocess,functools,argparse
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;ROOT=P.parent;W,H=960,540
OUT=P/'output';QA=P/'qa';OUT.mkdir(exist_ok=True);QA.mkdir(exist_ok=True)
T=json.loads((P/'tracks.json').read_text());Y,X=np.mgrid[:H,:W].astype(np.float32)
def source(n):return cv2.cvtColor(cv2.imdecode(np.fromfile(P/'reference'/f'f{n}.jpg',np.uint8),1),cv2.COLOR_BGR2RGB).astype(np.float32)
def mat(s=1,x=0,y=0):return np.float32([[s,0,x],[0,s,y]])
def mul(a,b):return (np.vstack([a,[0,0,1]])@np.vstack([b,[0,0,1]]))[:2].astype(np.float32)
def cam(name,n):
 v=T[name];s,cx,cy=[np.interp(n,v['frames'],np.array(v['values'])[:,i]) for i in range(3)]
 if name.startswith('back'):s=1;cy=270
 if name.startswith('cage'):s=1;cx=480
 return mat(s,cx-480*s,cy-270*s)
@functools.lru_cache(None)
def asset(name):
 a=np.array(Image.open(P/'assets'/f'{name}.png').convert('RGBA')).astype(np.float32)
 a[:,:,3]=np.clip((a[:,:,3]-1)*255/252,0,255);a[:,:,:3]*=a[:,:,3:4]/255
 return a
def warp(a,m,border=0):return cv2.warpAffine(a,m,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=border)
def layer(name,s,x,y,c=None):return warp(asset(name),mul(c if c is not None else mat(),mat(s,x,y)))
def over(bg,p):return bg*(1-p[:,:,3:4]/255)+p[:,:,:3]
def plate(name,c=None):
 a=asset(name);m=mat(960/a.shape[1],0,0)
 return warp(a[:,:,:3],mul(c if c is not None else mat(),m),border=(242,241,242))
def mix(a,b,m):return a*(1-m[:,:,None])+b*m[:,:,None]
def paper(im):
 # Unify empty paper across expanded canvas; retain colored ink and dark halftone.
 lo=im.min(2);sat=im.max(2)-lo
 m=np.clip((lo-224)/13,0,1)*np.clip((23-sat)/13,0,1)
 grain=(np.sin(X*2.7+Y*1.3)+np.sin(Y*2.4-X*.8))*.42
 bg=np.array([243,242,243])+grain[:,:,None]
 return mix(im,bg,m)
def bars(im):
 im[:53]=0;im[487:]=0;return np.clip(im,0,255).astype(np.uint8)

@functools.lru_cache(None)
def video(name):
 cap=cv2.VideoCapture(str(P/'h3'/f'{name}_v1.mp4'));out=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  out.append(cv2.cvtColor(cv2.resize(f,(960,540)),cv2.COLOR_BGR2RGB).astype(np.float32))
 return out

def back(n):
 bg=plate('back_bg',cam('back_wall',n))
 # Head top at source (240,99); skirt exceeds viewport, no bottom edge exposed.
 return over(bg,layer('back',.30,100,84,cam('back_person',n)))

NORMAL=source(3318)
@functools.lru_cache(5)
def crowd_field(n):
 if n<3104:return np.zeros((H,W),np.float32)
 # Only a scalar reveal matte is derived from source evidence. No source RGB enters the render.
 a=source(n);b=warp(NORMAL,cam('crowd',n),border=(150,150,155))
 bright=a.mean(2);base=b.mean(2)
 sat=a.max(2)-a.min(2)
 # Confident semantic seeds: navy clothing is normal; red/mint clothing is vivid.
 # Gray wallpaper becoming white and white witch dress becoming gray add evidence.
 outside=(X<350)|(X>650)
 pos=(((a[:,:,0]>140)&(a[:,:,0]>a[:,:,1]*1.35)&outside)|((a[:,:,1]-a[:,:,0]>12)&(bright>170))|((base>100)&(base<190)&(bright>214)&(sat<30))|((X>415)&(X<550)&(Y>220)&(base>215)&(bright>100)&(bright<190))).astype(np.float32)
 neg=(((bright<80)&outside)|((base>100)&(base<185)&(bright>95)&(bright<185)&(sat<25))|((X>420)&(X<540)&(Y>220)&(bright>225)&(sat<16))).astype(np.float32)
 pos[:56]=0;pos[485:]=0;neg[:56]=0;neg[485:]=0
 pp=cv2.GaussianBlur(pos,(0,0),19);nn=cv2.GaussianBlur(neg,(0,0),19)
 field=pp/(pp+nn+.00001)
 return field
def crowd_mask(n):
 f=(crowd_field(max(3101,n-1))+2*crowd_field(n)+crowd_field(min(3318,n+1)))/4
 return np.clip((f-.5)*18+.5,0,1)

def crowd(n):
 c=cam('crowd',n);a=plate('crowd_bg',c);b=plate('crowd_vivid',c)
 # Separate white dove belongs only to the vivid state, at the observed left side.
 b=over(b,layer('dove',.21,-30,170,c))
 b=over(b,layer('dove',.27,28,410,c))
 p=layer('witch_v2',.42,157,68,cam('witch',n));q=p.copy()
 alpha=p[:,:,3:4]/255;rgb=p[:,:,:3]/np.maximum(alpha,.001)
 # Gray-fill / raspberry-outline state shares exactly the same alpha and geometry.
 ink=np.clip((170-rgb.mean(2))/100,0,1)[:,:,None]
 q[:,:,:3]=(np.array([150,150,155])*(1-ink)+np.array([218,25,91])*ink)*alpha
 a=over(a,p);b=over(b,q);return mix(a,b,crowd_mask(n))

def cage(n):
 # Source observation: person rises 1.00px/frame, foreground barrel about 0.88px/frame.
 dy=-(n-3330)*.99; cb=mat(1.3,-144,-155+dy*.89)
 bg=plate('cage_bg',cb)
 ghost=layer('witch_v2',.42,157,25+dy)
 gm=ghost[:,:,3:4]/255*np.clip((bg.mean(2)-160)/45,0,1)[:,:,None]*((X>330)&(X<640))[:,:,None]
 bg=bg*(1-gm)+np.array([231,94,137])*gm
 # Maid/cage/flower share binding; camera crops forehead intentionally as in source.
 bg=over(bg,layer('cage_maid',.485,231,-35+dy))
 if n<3375:return bg
 t=n-3375
 # Two curved boundaries open a diagonal window into an independently drawn closeup.
 top=np.interp(n,[3375,3380,3390,3400,3426],[247,178,74,38,18])+.13*X+7*np.sin(X/170+t*.025)
 bot=np.interp(n,[3375,3380,3390,3400,3426],[264,288,324,335,353])+.13*X+5*np.sin(X/220+t*.02)
 m=np.clip(Y-top,0,1)*np.clip(bot-Y,0,1)
 close=paper(plate('closeup',mat(.87,35,-21+np.interp(t,[0,20,51],[20,0,-4]))))
 return mix(bg,close,m)

def mouth(n):
 ims=video('mouth');j=int(round((n-3427)/51*55));im=ims[j]
 # Stabilize H3's generated camera, then apply measured source camera once.
 hm=np.array(T['h3_mouth']['rows'][j]['matrix'],np.float32)
 inv=cv2.invertAffineTransform(hm)
 reg=mat(.64,177,-133)
 ext=asset('mouth_extended')[:,:,:3]
 bg=warp(ext,mul(cam('mouth',n),reg),border=(243,242,243))
 hreg=np.array(json.loads((P/'mouth_registration.json').read_text())['matrix'],np.float32)
 patch=warp(im,mul(cam('mouth',n),mul(reg,mul(hreg,inv))),border=(243,242,243))
 center=mul(cam('mouth',n),reg)@np.array([474,520,1])
 dist=((X-center[0])/17)**2+((Y-center[1])/12)**2
 ring=(dist>2)&(dist<3.5)
 patch+=np.median(bg[ring],axis=0)-np.median(patch[ring],axis=0)
 m=np.exp(-dist**2)
 return paper(mix(bg,patch,m))

def jump(n):
 ims=video('jump');j=int(round((n-3479)/26*55));im=ims[j]
 # Generated hop already includes body rise and head/arm action; only framing added.
 # Scale is fixed so the generated vertical bounce is retained, not applied twice.
 x=np.interp(n,[3479,3486,3494,3505],[-130,-160,-245,-350])
 return warp(im,mat(1.48,x,-95),border=(244,243,244))

@functools.lru_cache(None)
def lyric_mask(kind):
 im=Image.new('L',(W,H));d=ImageDraw.Draw(im)
 fontpath=__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc')
 font=ImageFont.truetype(fontpath,10)
 lines={'crowd':['平穏とは消耗を以て代わりに成す','実際はどうも変わりはなく'],'cage':['享楽とは嘘で成る'],'end':['綻ぶ前にここを出ていこうかと']}[kind]
 for j,line in enumerate(lines):
  x=915-j*17;y=175
  for ch in line:d.text((x,y),ch,font=font,fill=210);y+=10
 return np.array(im).astype(np.float32)/255
def frame(n):
 if 3061<=n<3073:im=back(n)
 elif 3101<=n<3319:im=crowd(n)
 elif 3319<=n<3427:im=cage(n)
 elif 3427<=n<3479:im=mouth(n)
 elif 3479<=n<3506:im=jump(n)
 else:return bars(source(n))
 if n>=3101:
  kind='crowd' if n<3319 else 'cage' if n<3427 else 'end'
  m=lyric_mask(kind); color=np.where(cv2.GaussianBlur(im.mean(2),(0,0),7)[:,:,None]>145,25,245)
  im=im*(1-m[:,:,None])+color*m[:,:,None]
 return bars(im)

def previews():
 ns=[3061,3067,3072,3101,3130,3180,3230,3280,3318,3319,3355,3375,3385,3400,3426,3427,3440,3455,3478,3479,3486,3494,3505]
 canvas=np.zeros((((len(ns)+2)//3)*294,1440,3),np.uint8)
 for i,n in enumerate(ns):
  f=frame(n);Image.fromarray(f).save(QA/f'f{n}.png');y=i//3*294;x=i%3*480
  canvas[y+24:y+294,x:x+480]=cv2.resize(f,(480,270));cv2.putText(canvas,str(n)+' | '+str(round(n/30,3)),(x+6,y+18),0,.5,(255,255,255),1)
 Image.fromarray(canvas).save(QA/'preview_sheet.jpg')
 for n in [3130,3180,3230,3280]:Image.fromarray((crowd_mask(n)*255).astype(np.uint8)).save(QA/f'mask{n}.png')

def render():
 target=OUT/'silent_v1.mp4';proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE)
 # Direct reuse reads original video, never JPEG analysis intermediates.
 cap=cv2.VideoCapture(str(next(ROOT.glob('*.mp4'))));cap.set(1,2664)
 for n in range(2664,3540):
  ok,orig=cap.read();assert ok
  if (3061<=n<3073) or (3101<=n<3506):f=frame(n)
  else:f=cv2.cvtColor(cv2.resize(orig,(960,540),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
  proc.stdin.write(f.tobytes())
  if n%150==0:print('frame',n,flush=True)
 proc.stdin.close();assert proc.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(target),'-ss','88.8','-i',str(next(ROOT.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t','29.2','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(OUT/'segment_88p8_118_v1.mp4')],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
 if args.preview:previews()
 else:render()
