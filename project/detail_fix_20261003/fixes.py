"""Scoped corrections; original renderer and approved files remain read-only."""
import functools,json,math
from pathlib import Path
import cv2 as C,numpy as N
from PIL import Image as PI,ImageDraw as PD,ImageFont as PF
import runtime as rt
P=Path(__file__).resolve().parent
EDITS=[]

def replace(src,old,new,label):
 assert old in src,label
 EDITS.append(label)
 return src.replace(old,new)

def edit_source(path,src):
 rel=path.relative_to(rt.ROOT).as_posix()
 if rel=='rebuild_60_90/segment_05_21/render_sequence_v5.py':
  src=replace(src,"bg=sea(n);desat=smooth((n-2199.36)/20)","bg=sea(n);desat=float(2218<=n<=2222)", '23 discrete monochrome flash')
  src=replace(src,"bg=mix(bg,np.full_like(bg,239),desat*.87)","bg=mix(bg,np.full_like(bg,239),float(n>=2212)*.87)",'23 separate white background switch')
  a="  if n<2343:\n   t=smooth((n-2335)/7);off=round(W*t)\n   old=dance(2334);whole=np.hstack([old,bg]);bg=whole[:,off:off+W]"
  b="  if n<2397:\n   t=smooth((n-2389)/7);off=round(W*t)\n   old=closeup(2388,'anon');whole=np.hstack([old,bg]);bg=whole[:,off:off+W]"
  src=replace(src,a,'','25 remove joined scene wipe')
  src=replace(src,b,'','26 remove joined scene wipe')
  src=replace(src,"dest=(506,215)","dest=(506-95*max(0,(n-2328)/6)**2,215)",'25 outgoing dance acceleration')
 if rel=='rebuild_182_202_agent_20261002_a/render_v4.py':
  start=src.index(' # Fast spatial handoffs')
  end=src.index(' text=None',start)
  src=src[:start]+' # Cuts switch scenes immediately; no concatenated-image transition.\n'+src[end:]
  EDITS.append('44 45 remove spatial_handoff blocks')
 if rel=='rebuild_60_90/segment_21_29/render_segment_v4.py':
  src=replace(src,"int(y+r*.6*math.sin(a))","int(y-(n-2443)*2.1+r*.6*math.sin(a))",'28 rising flowers independent of halo')
 if rel=='rebuild_147_182/render_v3.py':
  src=replace(src,"def loops(n,start,end,kind=0):\n t=", "def loops(n,start,end,kind=0):\n n=start+((n-start)//3)*3 if 4410<=n<4469 else n\n t=",'37 ribbon held on threes')
 if rel=='rebuild_88_118/render.py':
  src=replace(src,"def crowd_mask(n):\n f=", "def crowd_mask(n):\n n=3101+((n-3101)//3)*3\n f=",'30 discrete mask clock')
 return src

def tx(im,s=1,dx=0,dy=0,angle=0,center=None):
 """A camera transform with guaranteed source coverage; no invented border pixels."""
 a=rt.raw(im);d=rt.density(im);h,w=a.shape[:2]
 cx,cy=center or (w/2/d,h/2/d)
 # Additional cover only when translation would sample beyond existing artwork.
 need=max(1+2*abs(dx)*d/w,1+2*abs(dy)*d/h)
 z=max(s,need+(.003 if need>1 else 0))
 M=C.getRotationMatrix2D((cx*d,cy*d),angle,z);M[:,2]+=[dx*d,dy*d]
 return rt.tagged(C.warpAffine(a,M,(w,h),flags=C.INTER_LINEAR,borderMode=C.BORDER_CONSTANT),d)

def blur(im,sigma):
 if sigma<=.02:return im
 return rt.tagged(C.GaussianBlur(rt.raw(im),(0,0),sigma*rt.density(im)),rt.density(im))

def mask_text(lines,active=False):
 h=434 if active else 540
 d=rt.SCALE;im=PI.new('L',(960*d,h*d));draw=PD.Draw(im)
 font=PF.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/YuGothL.ttc'),11*d)
 for j,line in enumerate(lines):
  text=line.replace('　',' ').strip();top=max(12,(434-len(text)*12)/2)+(0 if active else 53)
  for k,ch in enumerate(text):draw.text(((908-j*24)*d,(top+k*12)*d),ch,font=font,fill=235)
 return rt.tagged(N.asarray(im).astype('float32')/255,d)

def caption(im,lines,color):
 m=mask_text(lines,im.shape[0]==434)[:,:,None]
 return im*(1-m)+N.asarray(color)*m

def clean_right(im,white=None,columns=1):
 # Remove only compact neutral glyph components in the legacy caption column.
 # This runs in the video renderer; scene pixels outside the glyph matte are retained.
 raw=rt.raw(im).copy();d=rt.density(im);active=im.shape[0]==434
 x0,x1=round((876 if columns>1 else 896)*d),round(956*d);y0,y1=round((100-(53 if active else 0))*d),round((470-(53 if active else 0))*d)
 roi=raw[y0:y1,x0:x1,:3];gray=C.cvtColor(roi,C.COLOR_RGB2GRAY);neutral=roi.max(2).astype(float)-roi.min(2)<36
 kernel=C.getStructuringElement(C.MORPH_ELLIPSE,(max(3,round(15*d))|1,max(3,round(15*d))|1))
 bright=C.morphologyEx(gray,C.MORPH_TOPHAT,kernel)>35
 dark=C.morphologyEx(gray,C.MORPH_BLACKHAT,kernel)>35
 light=bright&(gray>80)&(gray<(225 if active else 256))
 candidate=neutral&(light if white is True else (dark&(gray<200)) if white is False else (light|(dark&(gray<200))))
 # Thin anti-aliased strokes can connect to scene texture. Restrict the matte
 # to the measured legacy type column instead of dropping connected glyphs.
 columns_mask=N.zeros_like(candidate)
 for left,right in ([(878,899),(903,932)] if columns>1 else [(903,932)]):
  columns_mask[:,max(0,round(left*d)-x0):min(candidate.shape[1],round(right*d)-x0)]=True
 if not active:
  count,labels,stats,_=C.connectedComponentsWithStats(candidate.astype('uint8'))
  ids=[j for j,(x,y,w,h,area) in enumerate(stats) if j and w<=20*d and h<=360*d and 2<=area<=6500*d*d and max(w,h)>=4]
  candidate &= N.isin(labels,ids)
 mask=(candidate&columns_mask).astype('uint8')*255
 if mask.any():
  mask=C.dilate(mask,N.ones((max(1,round(d)),max(1,round(d))),N.uint8))
  raw[y0:y1,x0:x1,:3]=C.inpaint(roi,mask,max(1,round(2*d)),C.INPAINT_TELEA)
 return rt.tagged(raw,d)

def install(path,m):
 rel=path.relative_to(rt.ROOT).as_posix()
 if rel=='headswap_hybrid_30s/render.py':
  old=m.lyric
  def lyric(im,text,color):
   # The second running shot has a different line from the first.
   text='今から緞帳が上がるから' if getattr(m,'fix_frame',0)>=235 and getattr(m,'fix_frame',0)<262 else text
   return N.uint8(caption(im,[text],color))
  m.lyric=lyric
  original=m.render_frame
  def render(n,src):
   m.fix_frame=n
   return original(n,src)
  m.render_frame=render
  orig_read=m.read;cleaned={}
  def read(path):
   if path not in cleaned:
    a=orig_read(path)
    if Path(path).stem in ['anon_bowed','anon_front','anon_lifting','maid_floor','maid_medium','maid_down']:
     a=clean_right(a,Path(path).stem in ['maid_medium','maid_down'])
    cleaned[path]=a
   return cleaned[path]
  m.read=read
  for s in m.cfg['shots']:
   if s['start']==453:s.pop('focus',None);s['pixel_entry_frames']=0
   if s['start']==468:s.pop('focus',None)
   if s['start']==863:s.pop('flash_frame',None)
  # Replace overlapping focus later, retain nonoverlapping existing tracks.
  m.effect_tracks=[e for e in m.effect_tracks if not (e['start']<482 and e['end']>453)]
 elif rel=='headswap_hybrid_30_60/render.py':
  m.glyph=lambda lines:mask_text(lines,True)[:,:,None]
  old_init=m.ScenePlates.__init__
  def init(obj,s):
   old_init(obj,s)
   if s['id']=='B':
    old_weights=obj.engine.weights
    obj.engine.weights=lambda n:old_weights((n//3)*3)
  m.ScenePlates.__init__=init
 elif rel=='rebuild_60_90/render.py':
  m.caption=lambda bg,shot,white=False:N.uint8(caption(bg,m.LYRICS.get(shot,[]),(245,241,239) if white else (35,33,41)))
 elif rel=='rebuild_60_90/segment_05_21/render_sequence_v5.py':
  old=m.camera
  def camera(n):
   c=old(n)
   if 2335<=n<2389:
    k=n-2335;c['dx']+=190*math.exp(-k/4.1)-95*max(0,(n-2382)/6)**2
   elif 2389<=n<2443:
    k=n-2389;c['dx']-=190*math.exp(-k/4.1);c['dx']+=105*max(0,(n-2435)/7)**2
   return c
  m.camera=camera
 elif rel=='rebuild_88_118/render.py':
  lines={'crowd':['平穏とは消耗を以て代わりに成す','実際はどうも変わりはなく'],'cage':['享楽とは嘘で成る'],'end':['綻ぶ前にここを出ていこうかと']}
  m.lyric_mask=functools.lru_cache(None)(lambda kind:mask_text(lines[kind]))
 elif rel=='rebuild_118_147/render_v5.py':
  lines={'rain1':'都合の良い願いを同じ様に同じ様に呟く','rain2':'何処から聞こうか　何を見失うか','rain3':'定かじゃないから此処を動けない','dance1':'僕らが壊れるなら　これ以上無いなら','dance2':'その度に何回も逃げ出せる様に','dance3':'心が守れる様に　奪われない様に','end':'互いに託して　身体を預けてよ'}
  m.lyric_mask=functools.lru_cache(None)(lambda kind:mask_text([lines[kind]]))
  old=m.door
  def door(n):
   # Source rises rapidly then settles. Apply at native layers, before viewport crop.
   k=n-4372;dy=float(N.interp(k,[0,1,2,4,6,8,12,18,21],[110,86,72,53,28,29,15,0,-3]))
   cb=m.mat(1.08,-38,dy*.55-15)
   cf=m.mat(1+max(0,k)*.0005,-max(0,k)*.15,dy)
   im=m.plate('door_front_clean_v1.png',cb)
   im=m.over(im,m.fit('backgrounds/door_onlookers_v2.png',(0,169,254,595),cf))
   p=m.art('door_pair');left=p.copy();left[:,p.shape[1]//2:]=0;right=p.copy();right[:,:p.shape[1]//2]=0
   im=m.over(im,m.warp(left,m.mul(cf,m.mat(.38,125,129))))
   return m.over(im,m.warp(right,m.mul(cb,m.mat(.38,263,129))))
  m.door=door
 elif rel in ['rebuild_147_182/render.py','rebuild_182_202_agent_20261002_a/render_v4.py']:
  m.lyrics=functools.lru_cache(None)(lambda text:mask_text([text]))
  if '182_202' in rel:
   old=m.pattern
   def pattern(kind):
    if kind not in ['flowers','halo']:return old(kind)
    n=getattr(m,'fix_frame',5992);d=rt.SCALE
    # Keep halo fixed; animate only the flower population.
    a=rt.np.screen('zeros',(540,960,3),rt.np.float32);a[:]=[31,32,43]
    if kind=='halo':
     for r in [115,122,128]:rt.cv.circle(a,(515,165),r,(246,244,247),1,rt.cv.LINE_AA)
     for j in range(28):
      ag=j*N.pi*2/28;r2=168 if j%3==0 else 147
      rt.cv.line(a,(int(515+117*N.cos(ag)),int(165+117*N.sin(ag))),(int(515+r2*N.cos(ag)),int(165+r2*N.sin(ag))),(246,244,247),2,rt.cv.LINE_AA)
    for x,y,r in [(55,101,7),(222,305,5),(445,420,6),(817,183,7),(735,345,5),(875,468,5),(376,114,4)]:
     y=(y-(n-5992)*1.65+40)%600-40
     for j in range(5):
      ag=j*N.pi*2/5;rt.cv.circle(a,(int(x+N.cos(ag)*r*.65),int(y+N.sin(ag)*r*.65)),int(r*.58),(235,234,239),-1,rt.cv.LINE_AA)
    return a
   m.pattern=pattern
   fn=m.frame
   def frame(n):m.fix_frame=n;return fn(n)
   m.frame=frame
 elif rel=='rebuild_202_end_agent_20261002_b/render.py':
  m.captions=lambda dst,src,darkbg=False:dst
