from pathlib import Path
import json,cv2,numpy as np
import runtime as rt
P=Path(__file__).resolve().parent
class Whisper:
 def __init__(self):
  p=P/'h3/whisper_v4_result.mp4';assert p.exists(),p
  c=cv2.VideoCapture(str(p));self.frames=[]
  while True:
   ok,b=c.read()
   if not ok:break
   self.frames.append(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))
  assert len(self.frames)==56,len(self.frames)
  self.quiet=[]
  for i,im in enumerate(self.frames):
   a=im[162:605];m=(a.max(axis=2)<95).astype('uint8');m[:,:510]=0;m[:,700:]=0;m[140:]=0
   count,labels,stats,centers=cv2.connectedComponentsWithStats(m);parts=[s for s in stats[1:] if s[4]>45]
   height=int(max(parts,key=lambda s:s[4])[3]) if parts else 0
   if height<=35 or i in [0,55]:self.quiet.append(i)
  assert self.quiet
  (P/'h3/whisper_v4_retiming.json').write_text(json.dumps({'accepted_indices':self.quiet,'method':'Allow visible opening up to 35 native pixels, retain exact first/last guide frames. Reject only overshoots; no painting or warping of facial pixels.'},indent=2))
  self.cache={}
 def layer(self,i):
  key=(i,rt.SCALE)
  if key in self.cache:return self.cache[key]
  im=self.frames[i];h,w=im.shape[:2]
  # Crop only the supplied upper/lower bars; preserve the expanded figure proportions.
  a=im[round(h*284/1344):round(h*1058/1344)].astype('float32')
  excess=a[:,:,1]-np.maximum(a[:,:,0],a[:,:,2]);alpha=np.clip((100-excess)/70,0,1)
  # Despill the soft antialiased green boundary without touching opaque colored artwork.
  a[:,:,1]=np.minimum(a[:,:,1],np.maximum(a[:,:,0],a[:,:,2])+2)
  prem=a*alpha[:,:,None];z=434*rt.SCALE/a.shape[0]
  size=(round(a.shape[1]*z),434*rt.SCALE)
  prem=cv2.resize(prem,size,interpolation=cv2.INTER_CUBIC);alpha=cv2.resize(alpha,size,interpolation=cv2.INTER_CUBIC)
  x=round(85*rt.SCALE);prem=prem[:,x:x+960*rt.SCALE];alpha=np.clip(alpha[:,x:x+960*rt.SCALE],0,1)
  assert prem.shape==(434*rt.SCALE,960*rt.SCALE,3),prem.shape
  self.cache[key]=(prem,alpha);return prem,alpha
 def background(self,n):
  d=rt.SCALE;h,w=434*d,960*d;y,x=np.mgrid[:h,:w]/d
  noise=np.random.default_rng(6320).normal(0,1.1,(h,w)).astype('float32')
  bg=np.full((h,w,3),247,np.float32)+noise[:,:,None]*.25
  if n<6317:return bg
  dark=np.empty_like(bg);dark[:]=[45,47,60];dark+=noise[:,:,None]*1.7
  # Fine geometric prism lattice is a separately drawn background layer.
  for xx in range(-70,1030,78):
   cv2.line(dark,(xx*d,0),(xx*d,h),(141,144,155),max(1,d//2),cv2.LINE_AA)
   for yy in range(-40,500,90):
    pts=np.array([[xx,yy],[xx+39,yy+23],[xx+78,yy],[xx+78,yy+45],[xx+39,yy+68],[xx,yy+45],[xx,yy]],np.int32)*d
    cv2.polylines(dark,[pts],False,(143,145,156),max(1,d//2),cv2.LINE_AA)
  if n>=6319:return dark
  # Two brief speckled ink states precede the full dark switch; no long fade.
  field=(np.sin(x*.029)+np.cos(y*.041)+np.sin((x+y)*.018)+noise*.15)
  reveal=field>(1.4 if n==6317 else .35)
  return np.where(reveal[:,:,None],dark,bg)
 def frame(self,n):
  from fixes import caption,blur,tx
  k=n-6291;requested=round(min(k,39)/39*55);i=min(self.quiet,key=lambda j:abs(j-requested));prem,alpha=self.layer(i)
  a=prem+self.background(n)*(1-alpha[:,:,None]);a=rt.tagged(a,rt.SCALE)
  if k<18:a=blur(tx(a,1.06,0,-15*np.exp(-k/4.2)),3*np.exp(-k/3))
  if n>=6325:
   cell=round(np.interp(n,[6325,6331],[3,24]));a=rt.cv.resize(rt.cv.resize(a,(960//cell,434//cell),interpolation=cv2.INTER_AREA),(960,434),interpolation=cv2.INTER_NEAREST)
  out=rt.np.screen('zeros',(540,960,3),np.uint8);out[53:487]=np.uint8(np.clip(a,0,255))
  return np.uint8(caption(out,['正しくして'],(35,33,41)))
