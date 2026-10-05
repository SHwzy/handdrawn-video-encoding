"""V11 four local repairs. Unchanged frames are copied from V10."""
from pathlib import Path
import sys,json,math,functools,subprocess,argparse,time,hashlib
import cv2,numpy as np
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent;B=P.parent;R=B.parent;A=B/'step1_assets_v9/assets'
sys.path.insert(0,str(B/'revision_v7'));sys.path.insert(0,str(B))
import revision as prior
import fixes
rt=prior.rt
BASE=B/'grounding_v11/output/review_complete_0_230_detail_v11_master.mp4'
RANGES=[(98,154,3),(1973,2102,66),(2227,2335,75),(4313,4338,144),(5069,5278,173),(6120,6149,204)]

@functools.lru_cache(None)
def rgba(name):
 a=np.array(Image.open(A/(name+'.png')).convert('RGBA'),np.float32);a[:,:,3]=np.clip((a[:,:,3]-1)/252,0,1);a[:,:,:3]*=a[:,:,3:4];return a
@functools.lru_cache(None)
def rgb(name):return np.array(Image.open(A/(name+'.png')).convert('RGB'),np.float32)
def fit_native(new,old,alpha_max=1,unit=255):
 a=rgba(new).copy();o=rt.raw(old);h,w=o.shape[:2]
 def box(v,threshold):
  y,x=np.where(v>threshold);return np.array([x.min(),y.min(),x.max()+1,y.max()+1],float)
 ob=box(o[:,:,3],alpha_max*.5);nb=box(a[:,:,3],.5);s=min((ob[2]-ob[0])/(nb[2]-nb[0]),(ob[3]-ob[1])/(nb[3]-nb[1]))
 x=(ob[0]+ob[2]-(nb[0]+nb[2])*s)/2;y=ob[1]-nb[1]*s
 a=cv2.warpAffine(a,np.float32([[s,0,x],[0,s,y]]),(w,h));a[:,:,:3]*=unit/255;a[:,:,3]*=alpha_max
 return rt.tagged(a,rt.density(old))
def replace_art(m,fn,mapping,alpha_max=1,unit=255):
 old=getattr(m,fn);cache={}
 def load(name):
  if name not in mapping:return old(name)
  if name not in cache:cache[name]=fit_native(mapping[name],old(name),alpha_max,unit)
  return cache[name]
 setattr(m,fn,load)

def native_frame(name):
 return rt.tagged(cv2.resize(rgb(name),(960*rt.SCALE,540*rt.SCALE)),rt.SCALE)
def pack(a):return prior.prior.pack(a)
def cap(a,text,color=(240,240,240)):return fixes.caption(a,[text],color).astype('uint8')
def tx(a,s=1,dx=0,dy=0,angle=0):return fixes.tx(a,s,dx,dy,angle)
def rawwarp(a,M,w=960,h=434):
 d=rt.SCALE;M=np.array(M,np.float32).copy();M*=d
 return rt.tagged(cv2.warpAffine(a,M,(w*d,h*d),flags=cv2.INTER_LINEAR),d)
def active_rgba(name,width,x,y):
 a=rgba(name);s=width/a.shape[1];return rawwarp(a,[[s,0,x],[0,s,y]])

@functools.lru_cache(None)
def swap_background():
 a=np.full((1650,2400,3),(218,215,220),np.float32)
 for cy in range(-40,1750,100):
  for cx in range(-40,2500,100):
   for dx,dy in [(0,-18),(18,0),(0,18),(-18,0)]:cv2.circle(a,(cx+dx,cy+dy),27,(230,227,232),1,cv2.LINE_AA)
 return a

oldinstall=fixes.install
def install(path,m):
 oldinstall(path,m);rel=path.relative_to(R).as_posix()
 if rel=='headswap_hybrid_30s/crowd/renderer.py':
  a=rgb('04_panorama_final')[:,:,::-1];s=434/a.shape[0]
  m.plate=rt.tagged(cv2.resize(a,(round(a.shape[1]*s*rt.SCALE),434*rt.SCALE)),rt.SCALE);m.max_x=m.plate.shape[1]-960
 elif rel=='headswap_hybrid_30_60/render.py':
  old=m.load
  mapping={'chairs/assets/reality_v1.png':'16_chairs_reality','chairs/assets/white_v1.png':'16_chairs_white_dry','chase/assets/reality_v1.png':'19_chase_reality','chase/assets/white_v1.png':'19_chase_white','chase/assets/pattern_v1.png':'19_chase_pattern'}
  m.load=lambda path,rgba=False:rt.tagged(rgb(mapping[str(path)]),1) if str(path) in mapping else old(path,rgba)
  oldpre=m.premult_layer
  def pre(path,width=None,height=None):
   if str(path)!='finale/assets/seated_pair_v1.png':return oldpre(path,width,height)
   a=globals()['rgba']('19_eri_kneeling_pair_fix').copy();a[:,:,:3]/=255
   s=width/a.shape[1] if width else height/a.shape[0]
   return rt.tagged(cv2.resize(a,(round(a.shape[1]*s*rt.SCALE),round(a.shape[0]*s*rt.SCALE))),rt.SCALE)
  m.premult_layer=pre
 elif rel=='rebuild_60_90/render.py':
  replace_art(m,'asset',{'dance_pair_v1':'20_dance_monochrome'},1,1)
  oldasset=m.asset;cache={}
  def anchored_asset(name):
   if name!='anon_front_v1':return oldasset(name)
   if name not in cache:
    old=oldasset(name);h,w=rt.raw(old).shape[:2];a=rgba('20_anon_monochrome').copy();a[:,:,:3]/=255
    a=cv2.warpAffine(a,np.float32([[1,0,-325],[0,1,65]]),(max(w,a.shape[1]-325),h));cache[name]=rt.tagged(a,rt.density(old))
   return cache[name]
  m.asset=anchored_asset
 elif rel=='rebuild_60_90/segment_05_21/render_sequence_v5.py':
  oldsea=m.sea
  def sea(n):
   if n<2279 or n>=2335:return oldsea(n)
   a=np.array(Image.open(B/'grounding_v11/assets/beach_grounded.png').convert('RGB'),np.float32)
   angle,z,dx,dy=m.world(n);h,w=a.shape[:2]
   # Same camera scale, angle and centre as the standing/dancing pair.
   relative=np.interp(n,[2270,2280,2295,2310,2335],[.445,.425,.397,.38,.355])/.38
   s=max(960/w,434/h)*1.36*relative
   pan=-95*max(0,(n-2328)/6)**2
   mat=cv2.getRotationMatrix2D((w/2,h/2),angle,s);mat[:,2]+=np.array([506+pan,215])-[w/2,h/2]
   return rawwarp(a,mat)
  m.sea=sea
 elif rel=='rebuild_60_90/segment_21_29/render_segment_v4.py':
  oldsprite=m.sprite
  @functools.lru_cache(None)
  def sprite(name):
   if name!='hug':return oldsprite(name)
   a=np.array(Image.open(B/'grounding_v11/assets/hug_user.png').convert('RGBA'),np.float32)/255;a[:,:,:3]*=a[:,:,3:4]
   s=960*rt.SCALE/a.shape[1];return rt.tagged(cv2.warpAffine(a,np.float32([[s,0,0],[0,s,0]]),(960*rt.SCALE,540*rt.SCALE)),rt.SCALE)
  m.sprite=sprite
 elif rel=='rebuild_147_182/render.py':
  oldart=m.art
  @functools.lru_cache(None)
  def art(name):
   if name!='assets/forest.png':return oldart(name)
   a=np.array(Image.open(B/'grounding_v11/assets/forest_two_legs.png').convert('RGBA'),np.float32);a[:,:,3]=np.clip((a[:,:,3]-1)/252,0,1);a[:,:,:3]*=a[:,:,3:4]
   return rt.tagged(a,1)
  m.art=art
 elif rel=='rebuild_88_118/render.py':replace_art(m,'asset',{'back':'29_sakiko_back_right','cage_maid':'31_local_r2'},255)
 elif rel=='rebuild_118_147/render_v5.py':
  replace_art(m,'art',{'rain_handhold':'33_local_r3'})
  def door(n):
   k=n-4372;dy=float(np.interp(k,[0,1,2,4,6,8,12,18,21],[110,86,72,53,28,29,15,0,-3]))
   cb=m.mat(1.08,-38,dy*.55-15);cf=m.mat(1+k*.0005,-k*.15,dy)
   im=m.plate('door_front_clean_v1.png',cb)
   im=m.over(im,m.fit('backgrounds/door_onlookers_v2.png',(0,169,254,595),cf))
   p=m.art('door_pair');left=p.copy();left[:,p.shape[1]//2:]=0;right=p.copy();right[:,:p.shape[1]//2]=0
   # Fixed relative pose: both actors use the same camera. Preserve the former right actor endpoint.
   end=m.mat(1.0105,-3.15,-3);endbg=m.mat(1.08,-38,-16.65)
   registered=m.mul(cv2.invertAffineTransform(end),m.mul(endbg,m.mat(.38,263,129)))
   im=m.over(im,m.warp(left,m.mul(cf,m.mat(.38,125,129))))
   return m.over(im,m.warp(right,m.mul(cf,registered)))
  m.door=door
  oldcam=m.camera
  def cam(name,n):
   if name not in ['dance_a','dance_b']:return oldcam(name,n)
   start,end=(3972,4162) if name=='dance_a' else (4190,4292);q=np.clip((n-start)/(end-start-1),0,1);e=1-(1-q)**3
   z=(3.5-2.5*e) if name=='dance_a' else (1.38-.36*e)
   angle=(-12.3+13.5*e) if name=='dance_a' else (-3+5*e)
   M=cv2.getRotationMatrix2D((480,270),angle+.22*math.sin(q*math.tau*2.1),z);M[:,2]+=[3.2*math.sin(q*math.tau*2.4),170*(1-e) if name=='dance_a' else 0]
   M[1,2]+=2.1*math.sin(q*math.tau*1.8);return M.astype('float32')
  m.camera=cam
  original_states=m.dance_states
  def states(n,which):
   # Frame-specific generated glitches are backgrounds; apply the identical camera too.
   return original_states(n,which)
  m.dance_states=states
  original_glitches=m.glitches
  m.glitches=lambda n:m.warp(original_glitches(n),cam('dance_a' if n<4162 else 'dance_b',n),border=(136,137,147))
  @functools.lru_cache(None)
  def frames():
   f=B/'step2_v10/seedance/mouth.mp4';assert f.exists(),'H3 result missing'
   c=cv2.VideoCapture(str(f));out=[]
   while True:
    ok,a=c.read()
    if not ok:break
    out.append(rt.tagged(cv2.cvtColor(a,cv2.COLOR_BGR2RGB).astype('float32'),a.shape[1]/1344))
   c.release();return [out[round(x)] for x in np.linspace(8,80,34)]
  m.mouth_frames=frames
  oldmouth=m.mouth
  # Keep a distinct Seedance sample at every edit frame to retain intermediate mouth shapes.
  m.mouth=oldmouth
 elif rel=='rebuild_182_202_agent_20261002_a/render_v4.py':
  replace_art(m,'art',{'dip_normal.png':'46_normal_r2','dip_bone.png':'46_dip_bone_final'})
  original_art=m.art;low=rt.tagged(rgba('40_hood_bowed'),1)
  def forest(n):
   old=m.art;m.art=lambda name:low if name=='hood_full_raw.png' else old(name)
   try:
    q=np.clip((n-5499)/95,0,1);dx=-15+35*float(m.smooth(q))
    a=m.plate('forest_back.png',dx=dx*.6,zoom=1.14)
    a=m.over(a,m.cel('hood_full_raw.png',653+22*q,151,height=334))
    return m.over(a,m.cel('forest_front.png',-70+dx,45,height=460))
   finally:m.art=old
  m.forest=forest
  m.alt_anon=lambda n:native_frame('41_continuity_r2').copy()
  m.alt_sakiko=lambda n:native_frame('43_local_r2').copy()
  oldcrowd=m.crowd
  @functools.lru_cache(1)
  def reach_scene():
   old=m.art;cache={}
   def art(name):
    if name!='anon_full.png':return old(name)
    if name not in cache:cache[name]=fit_native('42_anon_reaching',old(name))
    return cache[name]
   m.art=art
   if hasattr(m.bounds,'cache_clear'):m.bounds.cache_clear()
   try:return oldcrowd(5595+45,False)
   finally:m.art=old;m.bounds.cache_clear()
  m.crowd=lambda n,second=False:tx(reach_scene(),1+.065*(1-math.exp(-(n-5691)/6))) if second else oldcrowd(n,False)
fixes.install=install

class ContentFilm(prior.RevisedFilm):
 def __init__(self):
  super().__init__();self.basecap=cv2.VideoCapture(str(BASE));self.last=-2
 def frozen(self,n):
  if n!=self.last+1:self.basecap.set(1,n)
  ok,b=self.basecap.read();assert ok,n;self.last=n
  if rt.SCALE!=2:b=cv2.resize(b,(960*rt.SCALE,540*rt.SCALE))
  return rt.tagged(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),rt.SCALE)
 def side(self,n):
  m=self.module('a1','headswap_hybrid_30s/render.py').modules['side'];bg,_=m._assets();k=n-154;motion=m.motion(k)
  excess=bg.shape[1]-960;crop=excess/2+48-motion['background_right_px']
  b=rt.cv.warpAffine(bg,np.float32([[1,0,-crop],[0,1,0]]),(960,434))[:,:,::-1].astype('float32')
  f=rgba('02_clean_foreground').copy();ear=rgba('02_earring')[249:1364,321:711];ear=cv2.resize(ear,(64,184));M=cv2.getRotationMatrix2D((32,0),24*math.sin(k*math.tau/27),1)
  M[:,2]+=[309,396];v=cv2.warpAffine(ear,M,(f.shape[1],f.shape[0]));f[:,:,:3]=f[:,:,:3]*(1-v[:,:,3:4])+v[:,:,:3];f[:,:,3]=f[:,:,3]+v[:,:,3]*(1-f[:,:,3])
  y=(434-f.shape[0]*.485)/2+motion['foreground_y_px'];v=rawwarp(f,[[.485,0,0],[0,.485,y]])
  out=v[:,:,:3]+b*(1-v[:,:,3:4])
  # Two short glints on the hanging jewel, local to the ear and screen transform.
  if k in [7,8,17,18]:
   ar=rt.raw(out).copy();d=rt.SCALE;point=M@np.array([32,124,1]);x=round(point[0]*.485*d);yy=round((point[1]*.485+y)*d);sz=(4 if k%2 else 6)*d
   cv2.line(ar,(x-sz,yy),(x+sz,yy),(255,255,255),d,cv2.LINE_AA);cv2.line(ar,(x,yy-sz),(x,yy+sz),(255,255,255),d,cv2.LINE_AA);out=rt.tagged(ar,d)
  return cap(pack(fixes.blur(out,4.2*math.exp(-k/4))), '段々と消える感嘆')
 def stage(self,n):
  m=self.module('a1','headswap_hybrid_30s/render.py').modules['stage'];b,_,_=m._render_layers(400);b=b[:,:,::-1].astype('float32')
  q=(n-372)/54;common=9*(q-.5);b=tx(b,1.014,0,common)
  a=rgba('gen06');s=960/a.shape[1];f=rawwarp(a,[[s,0,0],[0,s,434-a.shape[0]*s+common+4*(q-.5)]])
  out=f[:,:,:3]+b*(1-f[:,:,3:4]);return cap(pack(fixes.blur(out,.7)),'静かな会場を後にさよなら')
 def partner(self,n):
  m=self.module('bopen','rebuild_60_90/opening_00_05/render_v3.py');k=n-1790
  e=(1-math.exp(-k/62))/(1-math.exp(-182/62));angle=24*e;z=1.48-.66*(1-math.exp(-k/48))
  c=cv2.getRotationMatrix2D((480,217),angle,z)
  bgplate=rt.tagged(cv2.resize(rt.raw(m.plates(0)),(2400*rt.SCALE,1650*rt.SCALE)),rt.SCALE)
  # A common world transform drives both plate and actor; post-rotation actor sway only.
  bm=c.copy();bm[:,2]+=c[:,:2]@np.array([-720,-608]);bg=rt.cv.warpAffine(bgplate,bm,(960,434))
  a=rgba('19_close_r4');s=1000/a.shape[1];base=np.array([[s,0,-20],[0,s,-100],[0,0,1]],float)
  sway=np.clip((k-12)/12,0,1);base[0,2]+=sway*1.2*math.sin((k-12)*.10);base[1,2]+=sway*.65*math.sin((k-12)*.075)
  cm=(np.vstack([c,[0,0,1]])@base)[:2];fg=rawwarp(a,cm)
  if n>=1884:
   al=fg[:,:,3:4];ink=np.clip((165-(fg[:,:,:3]/np.maximum(al,.001)).mean(2))/100,0,1)[:,:,None]
   patterned=np.concatenate([(np.array([205,62,115])*(1-ink)+np.array([39,98,91])*ink)*al,al],2)
   rb=m.ribbons(n);re=m.return_reveal(n);fg=fg*(1-rb[:,:,None])+patterned*rb[:,:,None]
   fg=fg*(1-re[:,:,None])+rawwarp(a,cm)*re[:,:,None]
   tm=c.copy();tm[:,:2]*=1.7;tm[:,2]+=c[:,:2]@np.array([480-650*1.7,217-350*1.7]);tex=rt.cv.warpAffine(m.pattern_texture(),tm,(960,434))
   bg=bg*(1-rb[:,:,None])+tex*rb[:,:,None];bg=bg*(1-re[:,:,None])+rt.cv.warpAffine(bgplate,bm,(960,434))*re[:,:,None]
  out=bg*(1-fg[:,:,3:4])+fg[:,:,:3]
  if n>=1941:
   seq=self.module('bseq','rebuild_60_90/segment_05_21/render_sequence_v5.py')
   oldplace=seq.place;oldover=seq.over
   def registered_place(p,anchor,dest,scale,angle=0):
    # Register incoming eyes and joined hands to the outgoing shared pose.
    reg=np.float32([[1.53,0,-109],[0,1.53,-161],[0,0,1]])
    mat=(np.vstack([cm,[0,0,1]])@reg)[:2]
    return rawwarp(rt.raw(p),mat)
   seq.place=registered_place
   incoming_bg=rawwarp(swap_background(),bm)
   seq.over=lambda unused_background,p:oldover(incoming_bg,p)
   try:out=seq.swap_tail(n,out,fg/np.array([255,255,255,1]))
   finally:seq.place=oldplace;seq.over=oldover
  lines=['感じてたものが遠く放たれていた','同じ様で違うなんだか違う']
  return fixes.caption(pack(out),lines,(30,25,30)).astype('uint8')
 def frame(self,n):
  if not any(a<=n<b for a,b,_ in RANGES):return self.frozen(n)
  if 154<=n<182:return self.side(n)
  if 372<=n<427:return self.stage(n)
  if 427<=n<449:
   if not hasattr(self,'head_frames'):
    c=cv2.VideoCapture(str(P/'seedance/head.mp4'));self.head_frames=[]
    while True:
     ok,b=c.read()
     if not ok:break
     self.head_frames.append(cv2.cvtColor(cv2.resize(b,(1920,1080)),cv2.COLOR_BGR2RGB))
    c.release()
   idx=round(np.interp(n,[427,428,445,448],[0,1,20,len(self.head_frames)-1]));source=self.head_frames[idx][101:945]
   d=rt.SCALE;h=434*d;w=round(source.shape[1]*h/source.shape[0]);source=cv2.resize(source,(w,h));x=(w-960*d)//2
   a=np.zeros((540*d,960*d,3),np.uint8);a[53*d:487*d]=source[:,x:x+960*d]
   a=rt.tagged(a,d);a[:,860:]=native_frame('07_k1')[:,860:]
   return a
  if 1790<=n<1973:return self.partner(n)
  if 2130<=n<2172:
   m=self.module('bbase','rebuild_60_90/render.py');old=m.asset
   new=fit_native('21_sakiko_head_up',old('sakiko_stand_v1'),1,1)
   m.asset=lambda name:new if name=='sakiko_stand_v1' else old(name)
   try:return rt.cv.cvtColor(m.frame(n,rt.source_frame(n))[0],cv2.COLOR_BGR2RGB)
   finally:m.asset=old
  if 2192<=n<2200:
   k=n-2192;names=['22_k1','22_k2','22_k3_r2','22_k4_r3'];a=native_frame(names[k//2])[53:487]
   sigma=[0,0,.1,.65,1.1,1.8,2.4,3.2][k]
   if k==1:a=tx(a,1.006,4.8,0)
   if k==7:a=tx(a,1.012,-4.8,2.17)
   return cap(pack(fixes.blur(a,sigma)),'定かじゃないなら何を想うの')
  if 2200<=n<2205:
   a=super().frame(n);a[53:487]=fixes.blur(a[53:487],3.2*(1-(n-2200)/4));return a
  if 3479<=n<3506:
   # Remove the last three source samples and add the requested low-point holds.
   seq=list(range(3479,3486))+[3485]+[3486]*3+list(range(3487,3503));j=seq[n-3479]
   a=self.frozen(j);sigma=float(np.interp(n,[3479,3484,3485,3489,3494,3505],[2.1,.5,0,0,.8,2.1]));return pack(fixes.blur(a[53:487],sigma))
  if 5463<=n<5490:
   m=self.module('f','rebuild_182_202_agent_20261002_a/render_v4.py')
   # Distribute the moving take uniformly, removing three equivalent held tail frames.
   t=(n-5463)/26*(23/26);a=m.action('prone',t);a[53:487]=fixes.blur(a[53:487],float(np.interp(n,[5463,5471,5477,5483,5489],[3.4,2.4,1.4,.55,0])))
   return cap(a,'いま忘れないよう刻まれた空気を',(30,30,35))
  if 6644<=n<6698:
   a=native_frame('50_crowd_unified')[53:487];k=n-6644;z=1+.44*math.exp(-k/12)
   out=tx(a,z);base=rt.cv.warpAffine(rt.cv.cvtColor(rt.source_frame(6634),cv2.COLOR_BGR2RGB)[53:487],np.float32([[.65,0,168],[0,.65,150]]),(960,434))
   # Original red effect alone; screen-space flame is occluded by the new foreground.
   b=rt.raw(base).astype('float32');h,w=b.shape[:2];yy,xx=np.mgrid[:h,:w];r,g,bl=b[:,:,0],b[:,:,1],b[:,:,2]
   mask=np.clip((r-np.maximum(g,bl*.68)-9)/45,0,1)*(r>g+12)*(r>bl+8)
   # The whole head/body silhouette follows the same plate transform.
   hh,ww=rgb('50_crowd_unified').shape[:2];occ=np.zeros((hh,ww),np.float32)
   pts=np.array([[462,941],[484,870],[613,812],[558,755],[512,641],[504,561],[551,479],[568,353],[633,251],[723,193],[810,179],[918,195],[1019,251],[1093,352],[1128,480],[1171,553],[1164,651],[1139,755],[1071,815],[1198,861],[1242,941]],np.int32)
   cv2.fillPoly(occ,[pts],1);occ=rt.tagged(cv2.resize(occ,(960*rt.SCALE,540*rt.SCALE)),rt.SCALE)[53:487];occ=tx(occ,z)
   pix=rt.raw(out);neutral=(pix.max(2)-pix.min(2)<42)&(pix.mean(2)>125)
   mask*=np.clip(1-rt.raw(occ),0,1)*neutral
   out=out*(1-rt.tagged(mask[:,:,None],rt.SCALE))+base*rt.tagged(mask[:,:,None],rt.SCALE)
   return pack(fixes.blur(out,1.8*math.exp(-k/3.5)))
  return super().frame(n)

# Disable every existing generated caption BEFORE scene composition/motion.
# Original-source shots below are passed through with their single original caption.
_saved_mask=fixes.mask_text
fixes.caption=lambda im,*args,**kwargs:im
prior.prior.caption=fixes.caption
def _zero_text(lines,active=False):
 return rt.tagged(np.zeros(((434 if active else 540)*rt.SCALE,960*rt.SCALE),np.float32),rt.SCALE)
fixes.mask_text=_zero_text
prior.prior.mask_text=_zero_text
_previous_install=fixes.install
def subtitle_install(path,m):
 _previous_install(path,m)
 rel=path.relative_to(R).as_posix()
 if rel=='rebuild_202_end_agent_20261002_b/render.py':m.captions=lambda dst,*args,**kw:dst
 if rel=='rebuild_147_182/render.py':m.lyrics=lambda text:_zero_text([text])
 if rel=='rebuild_118_147/render_v5.py':m.lyric_mask=lambda kind:_zero_text([kind])
fixes.install=subtitle_install
class Film(ContentFilm):
 def subtitle_frame(self,n):
  if not any(a<=n<b for a,b,_ in RANGES):return self.frozen(n)
  if 98<=n<154 or 1973<=n<2034:return self.rgb(n)
  out=super().frame(n)
  if 2034<=n<2102:lines=['何時まで行こうか','何処まで行けるのか'];color=(245,241,239)
  elif 2227<=n<2335:lines=['僕らが離れるなら','僕らが迷うなら'];color=(245,241,239) if 2295<=n<2311 else (35,33,41)
  elif 4313<=n<4338:lines=['互いに託して　身体を預けてよ'];color=(29,28,35)
  elif 5069<=n<5278:lines=['君と歌う　君と踊る　君と話す'];color=(245,245,245)
  else:lines=['まだ誰も知らない感覚で'];color=(29,28,35)
  self.clean_frame=out.copy()
  mask=_saved_mask(lines)[:,:,None]
  self.last_subtitle_mask=rt.raw(mask).copy()
  return np.uint8(out*(1-mask)+np.array(color)*mask)
 def frame(self,n):
  fixed=self.subtitle_frame(n)
  if not any(a<=n<b for a,b,_ in RANGES):return fixed
  original=self.frozen(n).copy()
  # Restrict replacement to the complete caption column; retain all other V11 pixels.
  original[53:487,875:950]=fixed[53:487,875:950]
  if 5069<=n<5278:
   if not hasattr(self,'subtitle_hashes'):self.subtitle_hashes={}
   self.subtitle_hashes[str(n)]=hashlib.sha256(self.last_subtitle_mask.tobytes()).hexdigest()
  return original

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--samples',nargs='*',type=int);ap.add_argument('--range-index',type=int);ap.add_argument('--scale',type=int,default=2);args=ap.parse_args()
 rt.SCALE=args.scale;rt.WORK=P;rt.JOB='subtitles12_'+str(args.range_index);cv2.setNumThreads(2)
 for d in ['logs','state','migrated','qa','output']:(P/d).mkdir(exist_ok=True)
 f=Film()
 if args.samples:
  for n in args.samples:
   a=rt.raw(f.frame(n));assert a.shape==(540*args.scale,960*args.scale,3),(n,a.shape);assert np.isfinite(a).all();Image.fromarray(np.clip(a,0,255).astype('uint8')).save(P/'qa'/f'f{n}.png');print(n,flush=True)
 else:
  start,end,issue=RANGES[args.range_index];out=P/'output'/f'patch_{start}_{end}.mp4';assert not out.exists(),out
  p=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-qp','0','-preset','fast','-threads','2',str(out)],stdin=subprocess.PIPE)
  for n in range(start,end):
   a=rt.raw(f.frame(n));assert a.shape==(1080,1920,3) and np.isfinite(a).all(),(n,a.shape)
   a=np.clip(a,0,255).astype('uint8');a[:106]=0;a[974:]=0;p.stdin.write(a.tobytes())
   if n in [start,(start+end)//2,end-1]:Image.fromarray(a).save(P/'qa'/f'f{n}.png')
  p.stdin.close();assert p.wait()==0;print('DONE',start,end,flush=True)
 if hasattr(f,'clean_frame'):Image.fromarray(np.clip(rt.raw(f.clean_frame),0,255).astype('uint8')).save(P/'qa'/'last_clean.png')
 if hasattr(f,'subtitle_hashes'):(P/'logs'/f'{rt.JOB}_subtitle_hashes.json').write_text(json.dumps(f.subtitle_hashes),encoding='utf8')
 f.loader.save_log()
 (P/'logs'/f'{rt.JOB}_reads.json').write_text(json.dumps(rt.READS,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
