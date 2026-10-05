from pathlib import Path
import sys,json,math,functools
import numpy as np,cv2
from PIL import Image
P=Path(__file__).resolve().parent;B=P.parent;R=B.parent
sys.path.insert(0,str(B))
import render_fixes as prior
import fixes
rt=prior.rt
original_edit=fixes.edit_source
original_install=fixes.install

def edit(path,src):
 src=original_edit(path,src)
 if path.relative_to(R).as_posix()=='headswap_hybrid_30_60/render.py':
  src=src.replace('offset=4-dx*q if dx<0 else 4+dx*(1-q)',"offset=4-dx*q if dx<0 else 4+dx*(1-q)\n        if self.s['id']=='E':offset-=.462*4*max(0,n-162)")
 if path.relative_to(R).as_posix()=='rebuild_182_202_agent_20261002_a/render_v4.py':
  src=src.replace("cel('hand_a_'+state+'.png',0,53,width=960","cel('hand_a_'+state+'.png',0,13,width=960")
 if path.relative_to(R).as_posix()=='rebuild_147_182/render.py':
  assert 'dx=np.sin(n*.065)*.8' in src
  src=src.replace('dx=np.sin(n*.065)*.8','dx=0.0 # actor and ground share the same translation')
 return src

def install(path,m):
 original_install(path,m)
 rel=path.relative_to(R).as_posix()
 if rel=='rebuild_182_202_agent_20261002_a/render_v4.py':
  rows=m.RAW['hand_a_s']['rows']
  bad=next(x for x in rows if x['frame']==5973)
  left=next(x for x in rows if x['frame']==5972)['center'];right=next(x for x in rows if x['frame']==5974)['center']
  bad['center']=((np.array(left)+right)/2).tolist()
  oldtrack=m.track
  def track(key,n):
   if key=='hand_a_s' and n<5944:
    q=5944-n;return oldtrack(key,5944)+np.array([12.68*q+.65*q*q,1.25*q])
   return oldtrack(key,n)
  m.track=track
  def dip(n):
   z=np.interp(n,[5992,5995,6000,6012,6025,6040,6066,6073,6076],[2.1,1.72,1.48,1.22,1.10,1.055,1,.99,.79]);c=m.camera(z)
   normal_actor=m.cel('dip_normal.png',-55,24,width=940,c=c,canvas=True)
   sad_actor=m.cel('dip_solo.png',-55,24,width=940,c=c,canvas=True)
   bone_actor=m.cel('dip_bone.png',-55,24,width=940,c=c,canvas=True)
   # The chromatic disturbance belongs exclusively to the background.
   normal_bg=pink_cloud(n) if n<6007 else m.crowd_bg(n)
   normal=m.over(normal_bg,normal_actor)
   sad=m.over(m.pattern('gray_light').copy(),sad_actor)
   bone=m.over(m.pattern('halo').copy(),bone_actor)
   out=m.mix(normal,sad,m.reveal(n,5994,6007,1))
   out=m.mix(out,bone,m.reveal(n,6016,6029,1));out=m.mix(out,normal,m.reveal(n,6047,6063,1))
   if n>=6073:out=m.mix(out,np.full_like(out,246),.30*m.smooth((n-6072)/5))
   return out
  m.dip=dip
fixes.edit_source=edit;fixes.install=install

def pink_cloud(n):
 d=rt.SCALE;y,x=np.mgrid[:540*d,:960*d]/d;t=(n-5992)*.08
 field=np.sin(x*.017+y*.025+t)+.6*np.cos(x*.049-y*.014-t)+.35*np.sin(y*.087+x*.061)
 lo=np.array([49,23,73]);hi=np.array([227,43,134]);f=np.clip((field+1.8)/3.6,0,1)[:,:,None]
 a=lo*(1-f)+hi*f
 grain=np.random.default_rng(5992+n//3).normal(0,8,a.shape[:2])
 return rt.tagged(np.clip(a+grain[:,:,None],0,255).astype('float32'),d)

def mosaic(a,cell):
 if cell<=1:return a
 return rt.cv.resize(rt.cv.resize(a,(max(1,960//cell),max(1,434//cell)),interpolation=cv2.INTER_AREA),(960,434),interpolation=cv2.INTER_NEAREST)

def fire(n):
 d=rt.SCALE;h,w=434*d,960*d;a=np.full((h,w,3),[20,19,28],np.float32);t=(n-452)*.22
 for x,y,r in [(95,130,78),(245,315,86),(730,110,72),(890,315,100),(525,350,65)]:
  for j in range(10):
   ag=j*np.pi/5
   cv2.ellipse(a,(int((x+np.cos(ag)*r*.4)*d),int((y+np.sin(ag)*r*.4)*d)),(int(r*.58*d),int(r*.18*d)),j*36,0,360,(76,71,87),max(1,d//2),cv2.LINE_AA)
 # Separate curled flame silhouettes, animated below the person.
 phase=((n-452)//2)*.55
 paths=[[(0,455),(280,455),(235,396),(160,366),(235,317),(205,270),(125,255),(156,206),(212,180),(151,190),(89,226),(93,283),(158,323),(114,365),(38,374),(0,342)],[(668,452),(978,452),(977,265),(925,294),(880,278),(882,229),(920,194),(864,215),(826,265),(844,317),(907,342),(864,380),(795,393),(753,354),(775,310),(728,337),(710,397)]]
 for j,points in enumerate(paths):
  v=np.asarray(points,float);u=np.arange(len(v));v[:,0]+=10*np.sin(phase+u*.7+j);v[:,1]-=9*np.sin(phase*.8+u*.5)
  # Catmull-Rom closed interpolation retains curling tips without polygon corners.
  dense=[]
  for k in range(len(v)):
   p0,p1,p2,p3=[v[i%len(v)] for i in [k-1,k,k+1,k+2]]
   for tt in np.linspace(0,1,10,endpoint=False):dense.append(.5*((2*p1)+(-p0+p2)*tt+(2*p0-5*p1+4*p2-p3)*tt**2+(-p0+3*p1-3*p2+p3)*tt**3))
  mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[(np.asarray(dense)*d).astype('int32')],255,cv2.LINE_AA)
  opacity=mask.astype(float)/255
  scan=np.where((np.arange(h)//max(d*2,1))%2, .70,1)[:,None]
  col=np.array([231,0,78])[None,None,:]*scan[:,:,None]
  a=a*(1-opacity[:,:,None])+col*opacity[:,:,None]
 return rt.tagged(a,d)

class RevisedFilm(prior.FixedFilm):
 def __init__(self):
  super().__init__();self.local={}
 def asset(self,name):
  key=(name,rt.SCALE)
  if key not in self.local:
   im=Image.open(B/'revision_v6'/'assets'/name).convert('RGBA')
   if name=='title_clean.png':
    ys,xs=np.where(np.asarray(im)[:,:,3]>128);crop=im.crop((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))
    im=Image.new('RGBA',(1920,1080));im.alpha_composite(crop.resize((844,137),Image.Resampling.LANCZOS),(535,457))
   im=im.resize((960*rt.SCALE,540*rt.SCALE),Image.Resampling.LANCZOS)
   a=np.asarray(im,np.float32);a[:,:,3]/=255;a[:,:,:3]*=a[:,:,3:4]
   self.local[key]=rt.tagged(a,rt.SCALE)
  return self.local[key]
 def cathedral(self,n):
  m=self.modules['a1'].modules['cathedral']
  if not getattr(m,'revision_title',False):
   original=m._assets;bg,fg,_=original()
   title=self.asset('title_clean.png')[53:487].copy();title[:,:,:3]/=255
   m._assets=lambda:(bg,fg,title)
   m.apply_shake=lambda a,k:a
   m.revision_title=True
  # Restore the earlier continuous authored camera, replacing noisy raw matches.
  a=rt.cv.cvtColor(m.render(n),cv2.COLOR_BGR2RGB)[53:487]
  if 690<=n<698:a=fixes.tx(a,1+.035*((n-689)/8)**2)
  if n>=698:a=fixes.tx(a,1.008,.35*math.sin((n-698)*.28),.25*math.sin((n-698)*.37))
  raw=rt.raw(a).copy();d=rt.SCALE;rng=np.random.default_rng(113)
  for x,y,vx,vy,r,phase in zip(rng.uniform(0,960,40),rng.uniform(0,434,40),rng.uniform(-.23,.23,40),rng.uniform(.25,.7,40),rng.uniform(.25,.65,40),rng.uniform(0,6.28,40)):
   dt=n-482;xx=(x+vx*dt+3*np.sin(dt*.028+phase))%960;yy=(y-vy*dt)%434
   strength=.13+.12*np.sin(dt*.017+phase)**2
   overlay=raw.copy();cv2.circle(overlay,(round(xx*d),round(yy*d)),max(1,round(r*d)),(235,236,242),-1,cv2.LINE_AA)
   raw=cv2.addWeighted(raw,1-strength,overlay,strength,0)
  return prior.pack(rt.tagged(raw,d))
 def frame(self,n):
  # Source-reuse frames already carry the exact original typography: render it once.
  if any(a<=n<b for a,b in [(98,154),(182,235),(1973,2034),(2102,2130),(2172,2192),(2664,2868),(3933,3972),(4162,4190),(4292,4313),(5380,5460),(5744,5773),(6094,6120),(6149,6205)]):return self.rgb(n)
  if 318<=n<343:
   m=self.module('a1','headswap_hybrid_30s/render.py');start=318 if n<331 else 331;k=n-start
   a=rt.cv.cvtColor(m.read('assets/maid_medium.png' if n<331 else 'assets/maid_down.png'),cv2.COLOR_BGR2RGB)
   dx=-32*math.exp(-k/3.2)+.4*math.sin(k*1.1)
   a=fixes.tx(a,1.15,dx,.5*math.sin(k*.8));a=fixes.blur(a,2.4*math.exp(-k/2.8))
   return np.uint8(fixes.caption(prior.pack(a),['今から緞帳が上がるから'],(245,245,245)))
  if 449<=n<482:
   m=self.module('a1','headswap_hybrid_30s/render.py')
   front=rt.cv.cvtColor(m.read('assets/anon_front.png'),cv2.COLOR_BGR2RGB)
   if n<452:
    a=fixes.tx(front,1.035);a=mosaic(a,{449:2,450:6,451:18}[n])
   elif n<=466:
    fg=self.asset('anon_sharp_alpha.png')[53:487];bg=fire(n)
    a=bg*(1-fg[:,:,3:4])+fg[:,:,:3]
    a=fixes.tx(a,1.035+.001*(n-452));a=mosaic(a,{452:44,453:28,454:14,455:5}.get(n,1))
    if n==466:a=fixes.blur(a,2.1)
   else:
    k=n-467;z=2.08 if n<477 else float(np.interp(n,[477,480,481],[2.08,1.92,1.91]))
    a=fixes.tx(front,z,0,5*min(k,10)/10,center=(480,315))
    if n<477:a=fixes.blur(a,2.2)
   return prior.pack(a)
  if 1736<=n<1751:
   # Cubic Bezier speed envelope across 1735..1738: 5, 107/27, 55/27, 1 times baseline.
   a=prior.Film.frame(self,1743)[53:487]
   k=n-1736;distance=.462*k+.462*((80/27)*min(k,1)+(28/27)*min(max(k-1,0),1))
   return prior.pack(fixes.tx(a,1.035,-5+distance,0))
  if 1751<=n<1776:return prior.Film.frame(self,n)
  a=super().frame(n)
  # Pre-cut speed pulse is applied in ScenePlates before caption composition.
  return a

def main():
 import argparse,subprocess,time
 ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,nargs='*');ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);ap.add_argument('--name',default='trial');ap.add_argument('--scale',type=int,default=2);args=ap.parse_args()
 rt.SCALE=args.scale;rt.WORK=P;rt.JOB=args.name;cv2.setNumThreads(3);film=RevisedFilm()
 if args.samples:
  for n in args.samples:
   a=rt.raw(film.frame(n));assert a.shape==(540*rt.SCALE,960*rt.SCALE,3);Image.fromarray(a.astype('uint8')).save(P/'evidence'/f'{args.name}_{n}.png');print(n,flush=True)
 else:
  f=P/'output'/f'{args.name}.mp4';assert not f.exists()
  p=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{960*rt.SCALE}x{540*rt.SCALE}','-r','30','-i','-','-ss',str(args.start/30),'-i',str(rt.SOURCE),'-map','0:v','-map','1:a','-t',str((args.end-args.start)/30),'-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(f)],stdin=subprocess.PIPE)
  for n in range(args.start,args.end):
   p.stdin.write(rt.raw(film.frame(n)).astype('uint8').tobytes())
   if n%30==0:print(n,flush=True)
  p.stdin.close();assert p.wait()==0
 film.loader.save_log()
if __name__=='__main__':main()
