from pathlib import Path
import os,sys,json,functools,subprocess,time,argparse,math
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;R=P.parent
os.environ['HANDDRAWN_H3_SR_MANIFEST']=str(R/'render_1080p/h3_sr/replacements.json')
sys.path.insert(0,str(R/'render_1080p'))
import runtime as rt
rt.WORK=P;rt.JOB='detail_fixes_v1'
from render import Film
sys.path.insert(0,str(P))
from detail_loader import ModuleLoader
from fixes import tx,blur,caption,clean_right,mask_text,EDITS
DATA=json.loads((P/'measurements.json').read_text())

def measured(name,n):
 rows=[r for r in DATA[name]['rows'] if r['matrix'] is not None]
 mats=np.array([r['matrix'] for r in rows]);ns=[r['frame'] for r in rows]
 return np.array([[np.interp(n,ns,mats[:,i,j]) for j in range(3)] for i in range(2)],np.float32)
def interp(n,keys):return float(np.interp(n,[a for a,b in keys],[b for a,b in keys]))
def pack(active):
 out=rt.np.screen('zeros',(540,960,3),np.uint8);out[53:487]=np.clip(active,0,255).astype('uint8');return out

class FixedFilm(Film):
 def __init__(self):super().__init__();self.loader=ModuleLoader();self.stills={}
 def mannequin(self,n):
  if self.man is None:super().mannequin(n)
  engine,plates,_=self.man
  self.man=(engine,plates,mask_text(['従いたい心根を吐き出さぬように込めて','胸の中が澱のように濁る'],True)[:,:,None])
  return super().mannequin(n)
 @functools.lru_cache(10)
 def title(self):
  # Thin, oblique and widely spaced lettering; independent high-resolution type.
  d=rt.SCALE;a=Image.new('RGBA',(960*d,434*d));f=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/YuGothL.ttc'),46*d)
  for ch,x in zip('ロウワー',[272,405,543,686]):
   cel=Image.new('RGBA',(110*d,85*d));dr=ImageDraw.Draw(cel);dr.text((7*d,0),ch,font=f,fill=(249,248,250,245))
   cel=cel.transform(cel.size,Image.Transform.AFFINE,(1,-.42,18*d,0,1,0),Image.Resampling.BICUBIC)
   a.alpha_composite(cel,(x*d,164*d))
  v=np.array(a,np.float32)/255;v[:,:,:3]*=v[:,:,3:4]
  return rt.tagged(v,d)
 def cathedral(self,n):
  m=self.modules['a1'].modules['cathedral']
  # Composite first, then a single shared camera. Reference cels retain their native alpha.
  ref=490 if n<698 else 703
  key=('cat',ref)
  if key not in self.stills:
   bg,fg,_=m._assets();mo=m.motion(ref)
   b=m.halftone_background(blur(m.projected_background(ref),1 if n<698 else 2.5),mo['dots']).astype(np.float32)
   if n<698:b=m._overlay(b,self.title())
   s=mo['person_scale'];b=m._overlay(b,fg,np.float32([[s,0,480-470*s],[0,s,mo['sprite_y']]]))
   self.stills[key]=b
  a=self.stills[key].copy()
  name='wide' if n<698 else 'close';M=measured(name,max(n,ref));B=measured(name,ref)
  M=(np.vstack([M,[0,0,1]])@np.linalg.inv(np.vstack([B,[0,0,1]])))[:2]
  z=float(np.hypot(M[0,0],M[0,1]));point=M@np.array([480,270,1]);dx,dy=point-[480,270]
  if n<490:z=1+.055*(1-math.exp(-(n-482)/2.7));dx=0;dy=0
  # Entry is authored separately because mosaic frames have no reliable point matches.
  if n<698:
   if n>=490:z*=1.055
   z=max(1.001,z);a=tx(a,z,dx,dy)
   rng=np.random.default_rng(113)
   physical=rt.raw(a).copy();d=rt.density(a)
   for x,y,v in zip(rng.uniform(0,960,24),rng.uniform(0,434,24),rng.uniform(.1,.35,24)):
    yy=(y-(n-482)*v)%434;xx=x+2*math.sin(n*.024+y)
    cv2.line(physical,(round(xx*d),round(yy*d)),(round((xx+1)*d),round((yy+2)*d)),(210,211,216),max(1,round(d*.45)),cv2.LINE_AA)
   a=rt.tagged(physical,d)
   if n<490:
    cell=round(interp(n,[(482,32),(484,22),(486,12),(489,1)]))
    a=rt.cv.resize(rt.cv.resize(a,(max(1,960//cell),max(1,434//cell)),interpolation=cv2.INTER_AREA),(960,434),interpolation=cv2.INTER_NEAREST)
  else:
   z=max(1,z)*(1+.10*math.exp(-(n-698)/3.2))
   if n>=798:z*=1+.055*((n-797)/10)**2
   a=tx(a,z,dx,dy)
   sig=interp(n,[(698,5.4),(704,4.5),(712,2.7),(723,.2),(738,0),(765,.2),(774,.7),(784,0),(798,.25),(807,.65)])
   a=blur(a,sig)
  return pack(a[:,:,::-1]) # original m returns BGR; cached plates are BGR

 def frame(self,n):
  out=super().frame(n)
  if 71<=n<98 or 235<=n<262:
   # Running take has no internal captions: crop first, then the unified subtitle.
   m=self.modules['a1'];im=rt.cv.cvtColor(m.modules['seedance_running'].render(n-(71 if n<98 else 235)),cv2.COLOR_BGR2RGB)
   active=tx(im[53:487],1.80,0,-80,center=(530,210))
   out=pack(active);out=np.uint8(caption(out,['そう簡単な祈りだった　端から' if n<98 else '今から緞帳が上がるから'],(245,245,245)))
  elif 331<=n<343:
   a=out[53:487];k=n-331
   a=tx(a,1.11,-46*math.exp(-k/3.0)+1.0*math.sin(k*1.4),.7*math.sin(k*1.9))
   out=pack(blur(a,3.2*math.exp(-k/2.6)))
  elif 453<=n<468:
   a=out[53:487];cell=round(interp(n,[(453,48),(454,35),(455,23),(456,12),(457,4),(458,1),(464,1),(467,1)]))
   if cell>1:a=rt.cv.resize(rt.cv.resize(a,(max(1,960//cell),max(1,434//cell)),interpolation=cv2.INTER_AREA),(960,434),interpolation=cv2.INTER_NEAREST)
   if n>=465:a=blur(tx(a,1+.035*((n-464)/3)**2),interp(n,[(465,.3),(467,3.3)]))
   out=pack(a)
  elif 468<=n<482:
   a=out[53:487];d=rt.density(a);raw=rt.raw(a).copy()
   # Short magenta flame flickers confined to the background sides.
   for j,(x,y) in enumerate([(72,335),(855,366),(133,250),(805,175)]):
    phase=(n-468+j*2)//2;h=12+(phase%4)*7
    pts=np.array([[x,y],[x-7,y-h*.3],[x+4,y-h],[x+8,y-h*.35]],np.float32)*d
    cv2.fillPoly(raw,[pts.astype(np.int32)],(191,30,94),cv2.LINE_AA)
   a=rt.tagged(raw,d);a=tx(a,1+.07*max(0,(n-477)/4)**2)
   a=blur(a,interp(n,[(468,3.6),(472,1.4),(475,.15),(477,0),(479,1),(481,3.8)]));out=pack(a)
  elif 482<=n<808:
   out=self.cathedral(n)
  elif 808<=n<863:
   a=out[53:487];sig=interp(n,[(808,2.2),(815,.6),(822,0),(831,.6),(839,.1),(847,.8),(854,.1),(862,.35)])
   out=pack(blur(a,sig))
  elif 863<=n<878:
   a=blur(out[53:487],interp(n,[(863,2.8),(865,1),(867,.15),(872,0)]))
   flash=interp(n,[(863,0),(864,.04),(865,.18),(866,.53),(867,.88),(868,.52),(869,.23),(871,.06),(873,0)])
   out=pack(a*(1-flash)+255*flash)
  elif 1751<=n<1776:
   dx=65 if n==1751 else -65*math.exp(-(n-1752)/4.5)
   out=pack(tx(out[53:487],1.15,dx,0))
  elif 4410<=n<4469:
   # Short exposure mix only inside this shot; don't contaminate the following cut.
   if n>=4412:
    previous=super().frame(n-1);out=np.uint8(out*.72+previous*.28)
  elif 5069<=n<5278:
   M=measured('ground',n);p=M@np.array([480,270,1])
   rows=[r for r in DATA['ground']['rows'] if r['matrix'] is not None]
   ns=np.array([r['frame'] for r in rows]);ps=np.array([np.array(r['matrix'])@np.array([480,270,1]) for r in rows])
   trend=np.array([np.polyval(np.polyfit(ns,ps[:,j],1),n) for j in range(2)])
   dx,dy=np.clip(p-trend,-2.2,2.2)
   out=pack(tx(out[53:487],1.015,float(dx),float(dy)))
  elif 6291<=n<6332:
   from whisper_composite import Whisper
   if not hasattr(self,'whisper_fix'):self.whisper_fix=Whisper()
   out=self.whisper_fix.frame(n)
  # Legacy cels and source-reuse shots have captions baked into their pixels.
  lines=None;clean=False
  if 318<=n<343:lines=['今から緞帳が上がるから']
  elif 343<=n<372 or 427<=n<453:lines=['静かな会場を後にさよなら']
  elif 98<=n<154:lines=['そう簡単な祈りだった　端から'];clean=True
  elif 182<=n<235:lines=['段々と消える感嘆'];clean=True
  elif 1973<=n<2034:lines=['何時まで行こうか','何処まで行けるのか'];clean=True
  elif 2102<=n<2130 or 2172<=n<2192:lines=['定かじゃないなら何を想うの'];clean=True
  elif 2664<=n<2868:lines=['まだ誰も知らない感覚で救われていく'];clean=True
  elif 3933<=n<3972:lines=['僕らが壊れるなら　これ以上無いなら'];clean=True
  elif 4162<=n<4190:lines=['心が守れる様に　奪われない様に'];clean=True
  elif 4292<=n<4313:lines=['互いに託して　身体を預けてよ'];clean=True
  elif 5380<=n<5460:lines=['何時まで続くだろうと同じ様に同じ様に呟く'];clean=True
  elif 5744<=n<5773:lines=['僕らが離れるなら　僕らが迷うなら'];clean=True
  elif 6077<=n<6227:lines=['まだ誰も知らない感覚で'];clean=6094<=n<6120 or 6149<=n<6205
  elif 6227<=n<6291:lines=['僕の記憶するすべてを確かめて']
  if lines:
   if clean:out=clean_right(out,columns=len(lines))
   color=(242,242,242) if float(rt.raw(out)[int(160*rt.SCALE):int(390*rt.SCALE),int(900*rt.SCALE):int(930*rt.SCALE)].mean())<140 else (29,28,35)
   out=np.uint8(caption(out,lines,color))
  return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--samples',nargs='*',type=int);ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=6900);ap.add_argument('--scale',type=int,default=2);ap.add_argument('--name',default='candidate_v1');a=ap.parse_args()
 rt.SCALE=a.scale;rt.JOB=a.name;cv2.setNumThreads(3)
 for folder in ['output','qa','logs']: (P/folder).mkdir(exist_ok=True)
 film=FixedFilm();t=time.time()
 if a.samples:
  for n in a.samples:
   im=rt.raw(film.frame(n));assert im.shape==(540*a.scale,960*a.scale,3),im.shape
   Image.fromarray(im).save(P/'qa'/f'{a.name}_{n}.png');print(n,round(time.time()-t,1),flush=True)
 else:
  target=P/'output'/f'{a.name}.mp4';assert not target.exists()
  args=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{960*a.scale}x{540*a.scale}','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
  proc=subprocess.Popen(args,stdin=subprocess.PIPE)
  try:
   for n in range(a.start,a.end):
    im=rt.raw(film.frame(n));assert im.shape==(540*a.scale,960*a.scale,3) and np.isfinite(im).all(),n
    proc.stdin.write(im.astype('uint8').tobytes())
    if n%30==0:print(n,round(time.time()-t,1),flush=True)
  finally:proc.stdin.close()
  assert proc.wait()==0
 film.loader.save_log();(P/'logs'/f'{a.name}_edits.json').write_text(json.dumps(EDITS,indent=2),encoding='utf-8')
if __name__=='__main__':main()
