"""Independent artwork + deterministic layer/camera tracks, source [3540,4410)."""
from pathlib import Path
import cv2,numpy as np,json,subprocess,functools,argparse
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;R=P.parent;W,H=960,540
O=P/'output';Q=P/'qa_v5';Q.mkdir(exist_ok=True);O.mkdir(exist_ok=True)
Y,X=np.mgrid[:H,:W].astype(np.float32)
def smooth(t):
 t=np.clip(t,0,1);return t*t*(3-2*t)
def read(p):return np.array(Image.open(p).convert('RGB')).astype(np.float32)
def save(p,a):Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(p)
def mat(s=1,x=0,y=0):return np.float32([[s,0,x],[0,s,y]])
def mul(a,b):return (np.vstack([a,[0,0,1]])@np.vstack([b,[0,0,1]]))[:2].astype(np.float32)
def warp(a,m,border=0):return cv2.warpAffine(a,m,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=border)
def over(bg,a):return bg*(1-a[:,:,3:4])+a[:,:,:3]
def mix(a,b,m):return a*(1-m[:,:,None])+b*m[:,:,None]
def bars(a):
 a=a.copy();a[:53]=0;a[487:]=0;return np.clip(a,0,255).astype(np.uint8)
@functools.lru_cache(None)
def art(name):
 path=P/(name if '/' in name else 'assets/'+name+'.png')
 a=np.array(Image.open(path).convert('RGBA')).astype(np.float32)
 # Built-in transparent outputs have some 253/254 interiors. Normalize before premultiply.
 a[:,:,3]=np.clip((a[:,:,3]-1)/252,0,1);a[:,:,:3]*=a[:,:,3:4]
 return a
def layer(name,s,x,y,c=None):return warp(art(name),mul(c if c is not None else mat(),mat(s,x,y)))
@functools.lru_cache(None)
def box(name):
 yy,xx=np.where(art(name)[:,:,3]>.25);return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
def fit(name,target,c=None):
 # Alpha cleanup changes the box, but this revised cel keeps its approved registration.
 x,y,x2,y2=[0,0,1024,1536] if name=='rain_lean' and art(name).shape[:2]==(1536,1024) else box(name)
 tx,ty,tx2,ty2=target;s=min((tx2-tx)/(x2-x),(ty2-ty)/(y2-y))
 return layer(name,s,(tx+tx2)/2-(x+x2)*s/2,ty-y*s,c)
@functools.lru_cache(None)
def bg(name):return read(P/'backgrounds'/name)
def plate(name,c=None,angle=0,scale=1):
 a=bg(name);s=max(W/a.shape[1],434/a.shape[0])*scale
 m=cv2.getRotationMatrix2D((a.shape[1]/2,a.shape[0]/2),angle,s)
 m[:,2]+=np.array([480,270])-np.array([a.shape[1]/2,a.shape[0]/2])
 return warp(a,mul(c if c is not None else mat(),m),border=(235,235,239))
RAW=json.loads((P/'motion_raw.json').read_text())
TRACK={}
for name,v in RAW.items():
 rows=[r for r in v['rows'] if r['inliers']>=10 and .7<r.get('scale',0)<3 and abs(r.get('angle',90))<15 and 150<r['center'][0]<750 and 0<r['center'][1]<550]
 if name=='dance_a':
  # Early hands-only reference has no stable shared features. Extrapolation anchored visually.
  rows=[{'frame':3972,'scale':3.5,'angle':-12.3,'center':[480,440],'inliers':0}]+rows
 TRACK[name]=rows
(Q/'camera_used.json').write_text(json.dumps({'note':'Reference-to-frame transforms. inliers>=10, constrained scale/angle/center; unreliable colored silhouette spans interpolated. First dance start manually anchored.','tracks':TRACK},indent=2))
def camera(name,n):
 rows=TRACK[name];ns=[r['frame'] for r in rows]
 vals=[float(np.interp(n,ns,[r[k] for r in rows])) for k in ['scale','angle']]
 center=[float(np.interp(n,ns,[r['center'][j] for r in rows])) for j in [0,1]]
 m=cv2.getRotationMatrix2D((480,270),vals[1],vals[0]);m[:,2]+=np.array(center)-[480,270];return m
@functools.lru_cache(None)
def field(seed):
 rng=np.random.default_rng(seed)
 return cv2.resize(rng.random((9,17)).astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)*.16+cv2.resize(rng.random((24,39)).astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)*.055
def reveal(n,a,b,seed=1,direction=1):
 if n<=a:return np.zeros((H,W),np.float32)
 if n>=b:return np.ones((H,W),np.float32)
 f=(X/W if direction>0 else 1-X/W)+.13*np.sin(Y*.017)+.065*np.sin(Y*.049)+.10*np.sin(X*.018+Y*.021)+field(seed)-.1
 return smooth(((n-a)/(b-a)*1.6-.3-f)/.012+.5)
@functools.lru_cache(None)
def styled(name,mode):
 a=art(name).copy();al=a[:,:,3:4];rgb=a[:,:,:3]/np.maximum(al,.001);hh,ww=al.shape[:2];yy,xx=np.mgrid[:hh,:ww]
 left=xx<ww*.505
 ink=np.clip((205-rgb.mean(2))/150,0,1)
 if mode=='line':
  col=np.where(left[:,:,None],[180,32,90],[24,27,34]);color=np.array([171,171,179])*(1-ink[:,:,None])+col*ink[:,:,None]
  color=np.where(((rgb.mean(2)<60)&~left)[:,:,None],[21,23,31],color)
 elif mode=='solid':
  col=np.where(left[:,:,None],[235,68,139],[240,105,66]);col=np.where(((yy>hh*.45)&~left)[:,:,None],[230,198,104],col)
  dots=((xx%15-7)**2+(yy%15-7)**2)<5
  motifs=(np.abs((xx%106)-53)<3)|(np.abs((yy%106)-53)<3)
  color=np.where(dots[:,:,None],col*.75,col);color=np.where((motifs&left)[:,:,None],color*.84,color)
 else:color=rgb
 a[:,:,:3]=color*al;return a
@functools.lru_cache(None)
def pattern(kind):
 a=np.full((1100,1800,3),[246,245,244],np.float32)
 if kind==0:
  for y in range(-60,1200,150):
   for x in range(-80,1900,150):
    pts=np.array([[x+75,y],[x+150,y+75],[x+75,y+150],[x,y+75]],np.int32)
    cv2.polylines(a,[pts],True,(60,63,70),1,cv2.LINE_AA)
    cv2.rectangle(a,(x+25,y+25),(x+125,y+125),(70,74,81),1,cv2.LINE_AA)
  for x in [60,300,1380,1600]:
   for y in range(0,1100,50):cv2.circle(a,(x+(y//50%2)*22,y),17,(30,35,46),-1,cv2.LINE_AA)
 else:
  for y in range(0,1100,35):
   for x in range(0,1800,35):cv2.circle(a,(x+(y//35%2)*17,y),10,(219,234,233),-1,cv2.LINE_AA)
  cv2.rectangle(a,(0,850),(1800,1100),(245,242,244),-1)
  for y in range(855,1110,75):cv2.line(a,(0,y),(1800,y),(199,142,211),4,cv2.LINE_AA)
  for x in range(-1200,2800,180):cv2.line(a,(900+(x-900)//3,850),(x,1100),(202,145,213),4,cv2.LINE_AA)
 a[::3,::3]*=.91;return a
def pattern_bg(kind,c):
 a=pattern(kind);m=cv2.getRotationMatrix2D((900,550),8,.78);m[:,2]+=np.array([480,270])-[900,550]
 return warp(a,mul(c,m),border=(245,244,244))
@functools.lru_cache(None)
def glyph_bg():
 a=np.full((1000,1700,3),[83,86,101],np.float32)
 for y in range(-140,1100,210):
  for x in range(-200,1800,210):
   for p,q in [((x,y),(x+210,y+210)),((x+210,y),(x,y+210)),((x,y),(x+210,y)),((x,y),(x,y+210))]:cv2.line(a,p,q,(224,224,228),1,cv2.LINE_AA)
   for dx,dy,aa in [(0,0,0),(210,0,90),(210,210,180),(0,210,270)]:cv2.ellipse(a,(x+dx,y+dy),(145,145),0,aa,aa+90,(224,224,228),1,cv2.LINE_AA)
 a[::2,::2]*=.77;return a
def grid(n):return warp(glyph_bg(),mat(.95,-110+(n%100)*.08,-80),border=(82,85,100))
def guests(im,c,kind):
 name='backgrounds/ballroom_guests_pair_v1.png'
 # Fit by person height, never by the broad skirt's width. Anchor actual shoe soles.
 # Scene-space ground points follow the same camera as the architectural floor.
 x0,y0,x1,y1=box(name);sole=(520,1367)
 for gx,gy,height,opacity in [(525,470,285,.44),(155,520,340,.48),(925,435,325,.42)]:
  if kind==2:gy+=70  # The vivid pattern floor starts farther down the background plane.
  s=height/(y1-y0);m=cv2.getRotationMatrix2D(sole,8,s)
  m[:,2]+=np.array([gx,gy])-sole
  p=warp(art(name),mul(c,m))
  if kind==2:p[:,:,:3]=np.array([146,147,155])*p[:,:,3:4]
  p*=opacity;im=over(im,p)
 if kind:
  p=fit(name,(-255,-10,265,1100),c)
  if kind==2:p[:,:,:3]=np.array([160,160,166])*p[:,:,3:4]
  p*=.75;im=over(im,p)
 return im
def glitches(n):
 rng=np.random.default_rng(135+n//3);a=np.full((H,W,3),[136,137,147],np.float32)
 noise=cv2.resize(rng.random((32,55)).astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)
 a+=(noise[:,:,None]-.5)*155
 palette=np.array([[244,28,52],[38,35,237],[189,20,218],[52,231,74],[244,223,191],[28,26,43]],np.float32)
 for j in range(105):
  x=int(rng.integers(-80,960));y=int(rng.integers(50,480));w=int(rng.integers(12,100));h=int(rng.integers(5,65));cv2.rectangle(a,(x,y),(x+w,y+h),tuple(float(v) for v in palette[rng.integers(0,len(palette))]),-1)
 a=cv2.GaussianBlur(a,(0,0),3.2);a[::3]*=.79
 return a
def dance_states(n,which):
 name='dance_pair_a_v4' if which=='a' else 'dance_pair_b_closed_v5';c=camera('dance_'+which,n)
 if which=='a':s=.50;anchor=(765,335);dest=(480,218)
 else:s=.46;anchor=(768,365);dest=(480,197)
 initial=cv2.getRotationMatrix2D(anchor,8,s);initial[:,2]+=np.array(dest)-anchor
 M=mul(c,initial);norm=warp(art(name),M)
 masked=name if which=='a' else 'dance_pair_b_open_registered_v5'
 line=warp(styled(masked,'line'),M);solid=warp(styled(masked,'solid'),M)
 real=guests(plate('ballroom_clean_v1.png',c,angle=8,scale=1.65),c,0)
 pat=guests(pattern_bg(0,c),c,1);vivid=guests(pattern_bg(1,c),c,2)
 return c,[real,pat,vivid],[norm,line,solid]
def dance(n,which):
 c,bgs,ps=dance_states(n,which)
 if which=='a':trans=[(0,1,3974,3984,1),(1,2,3998,4012,-1),(2,0,4030,4043,1),(0,1,4081,4093,1),(1,2,4112,4124,-1),(2,0,4138,4156,1)];gl=[(3972,3983),(4056,4090)]
 else:trans=[(0,1,4191,4203,1),(1,2,4220,4231,-1),(2,0,4247,4264,1)];gl=[(4190,4204),(4272,4292)]
 b=bgs[0];p=ps[0]
 for i,(a,z,start,end,di) in enumerate(trans):
  if n>=end:b=bgs[z];p=ps[z]
  elif n>=start:
   mb=reveal(n,start-2,end-1,11+i,di);mp=reveal(n,start,end,31+i,di)
   b=mix(bgs[a],bgs[z],mb);p=ps[a]*(1-mp[:,:,None])+ps[z]*mp[:,:,None];break
  else:break
 g=np.zeros((H,W),np.float32)
 for a,z in gl:
  if a<=n<z:
   env=min(1,(n-a+4)/5,(z-n)/5);g=np.maximum(g,env*smooth((field(77)+.10*np.sin(X*.021+Y*.025+n*.13)-.04)*9))
 b=mix(b,glitches(n),g*.88)
 im=over(b,p)
 if which=='b' and n>=4272:
  f=smooth((n-4272)/7);veil=smooth((field(19)+.13*np.sin(X*.014+Y*.018+n*.03)-.09)*6)*f*.65
  im=mix(im,np.full_like(im,242),veil)
 return im
@functools.lru_cache(None)
def rain_particles():
 rng=np.random.default_rng(118);return np.c_[rng.uniform(-50,1010,145),rng.uniform(-100,620,145),rng.uniform(4,20,145),rng.uniform(6,39,145),rng.uniform(.2,.65,145)]
def rain_overlay(im,n):
 # Independent one-frame ink streaks: deliberate discontinuity, no slow falling rods.
 rng=np.random.default_rng(118000+n);a=im.copy()
 for i in range(round(int(rng.integers(66,98))*.60)):
  x=float(rng.uniform(-30,990));y=float(rng.uniform(15,500))
  length=float(rng.uniform(12,84))*float(rng.uniform(.20,1.00));width=float(rng.uniform(.45,2.8))
  drift=float(rng.uniform(-.25,-.14))*length
  bend=float(rng.uniform(-1.2,1.2));taper=float(rng.uniform(.12,.5))
  # Source streaks fall upper-right to lower-left, thin trailing top, heavier lower head.
  pts=np.array([[x,y],[x+width*taper,y+length*.12],[x+drift*.55+bend+width*.5,y+length*.55],[x+drift+width*.55,y+length*.96],[x+drift,y+length],[x+drift-width*.65,y+length*.96],[x+drift*.5+bend-width*.45,y+length*.5]],np.int32)
  color=(222,28,86) if i%4 else (157,24,65)
  cv2.fillPoly(a,[pts],color,cv2.LINE_AA)
  if i%5==0:cv2.line(a,(int(x+drift),int(y+length+4)),(int(x+drift-1),int(y+length+9)),color,1,cv2.LINE_AA)
 return im*.20+a*.80
BLACK=[(3632,3635),(3718,3722),(3848,3852),(3862,3865),(3917,3920),(3923,3927),(3931,3933)]
def rain(n):
 im=plate('rain_courtyard_v1.png')
 im=mix(im,np.full_like(im,245),smooth((Y-120)/250)*.40)
 if n<3722:
  if n>=3635:
   group=art('backgrounds/rain_passers_scene1_v1.png')
   for lo,hi,dx in [(1200,group.shape[1],0),(635,1200,50),(0,635,72)]:
    person=group.copy();person[:,:lo]=0;person[:,hi:]=0
    im=over(im,warp(person,mat(.47,dx,52)))
  im=over(im,fit('rain_sakiko_v5',(447,105,617,466)))
 elif n<3848:
  im=over(im,fit('backgrounds/rain_blue_pair_v2.png',(175,68,445,445)))
  im=over(im,fit('rain_anon',(655,73,841,467)))
  im=over(im,fit('rain_sakiko_raised',(500,105,617,466)))
 elif n<3862:
  im=over(im,fit('rain_passing_v5',(355,80,617,467)))
 elif n<3917:
  im=over(im,fit('rain_lean',(355,105,617,467)))
 elif n<3927:im=over(im,fit('rain_handhold',(281,90,617,467)))
 else:im=over(im,fit('rain_run',(160,95,582,467)))
 return rain_overlay(im,n)
@functools.lru_cache(None)
def posters():
 a=read(P/'assets/poster_faceless_a.png');b=read(P/'assets/poster_faceless_b_raw.png');b=cv2.resize(b,(a.shape[1],a.shape[0]))
 # Register B to A on whole image, then allow changes only inside portrait regions.
 sift=cv2.SIFT_create(nfeatures=3000);ka,da=sift.detectAndCompute(cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_RGB2GRAY),None);kb,db=sift.detectAndCompute(cv2.cvtColor(b.astype(np.uint8),cv2.COLOR_RGB2GRAY),None)
 good=[m for m,z in cv2.BFMatcher().knnMatch(db,da,k=2) if m.distance<.7*z.distance]
 M,ins=cv2.estimateAffinePartial2D(np.float32([kb[m.queryIdx].pt for m in good]),np.float32([ka[m.trainIdx].pt for m in good]),method=cv2.RANSAC,ransacReprojThreshold=2)
 b=cv2.warpAffine(b,M,(a.shape[1],a.shape[0]),borderMode=cv2.BORDER_REFLECT)
 yy,xx=np.mgrid[:a.shape[0],:a.shape[1]];mask=np.zeros(a.shape[:2],np.float32)
 for cx,cy,rx,ry in [(.215,.41,.092,.20),(.807,.41,.089,.20),(.52,.88,.082,.19)]:
  d=((xx/a.shape[1]-cx)/rx)**2+((yy/a.shape[0]-cy)/ry)**2;mask=np.maximum(mask,smooth((1-d)*5))
 # Keep paint crosses fixed even within face regions.
 red=(a[:,:,0]>150)&(a[:,:,1]<85)&(a[:,:,2]<145);mask[red]=0
 b=mix(a,b,mask*.75);save(P/'assets/poster_faceless_b_registered.png',b)
 (Q/'poster_registration.json').write_text(json.dumps({'matrix_b_to_a':M.tolist(),'inliers':int(ins.sum()),'note':'B only within 3 portrait ellipses; red crosses fixed.'},indent=2))
 return a,b
def poster(n):
 a=posters()[((n-4313)//3)%2];s=960/a.shape[1];m=mat(s,0,53)
 # native aspect ratio is close to active view, scale y minimally to fill.
 a=cv2.resize(a,(960,434));normalize=cv2.invertAffineTransform(camera('poster',4337))
 return warp(a,mul(mul(camera('poster',n),normalize),mat(1,0,53)),border=(93,92,100))
@functools.lru_cache(None)
def mouth_frames():
 cap=cv2.VideoCapture(str(P/'h3/mouth_v1.mp4'));fs=[]
 while True:
  ok,bgr=cap.read()
  if not ok:break
  fs.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32))
 return fs
def mouth_alpha(im):
 # Generated image contour separates left character from blank paper at right.
 sat=im.max(2)-im.min(2);barrier=((sat>23)|(im.mean(2)<205)).astype(np.uint8)
 barrier=cv2.morphologyEx(barrier,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8));barrier=cv2.dilate(barrier,np.ones((3,3),np.uint8))
 flood=barrier.copy();cv2.floodFill(flood,None,(im.shape[1]-2,im.shape[0]//2),2)
 alpha=(flood!=2).astype(np.float32)
 # Drop isolated noise in the paper; the main silhouette touches the left boundary.
 count,labels,stats,_=cv2.connectedComponentsWithStats(alpha.astype(np.uint8))
 if count>1:alpha=(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.float32)
 return cv2.GaussianBlur(alpha,(3,3),.6)
def mouth(n):
 fs=mouth_frames();idx=round((n-4338)/33*(len(fs)-1));im=fs[idx];al=mouth_alpha(im)
 p=np.dstack([im*al[:,:,None],al])
 # Match source final nose at ~427,160 and lower face crop; compensate large source pan.
 tx=float(np.interp(n,[4338,4343,4348,4353,4358,4363,4368,4371],[-245,-137,-87,-55,-32,-15,0,6]))
 m=mat(.72,tx,20)
 a=layer_from(p,m);result=over(grid(n),a)
 blur=float(np.interp(n,[4338,4345,4351,4361,4371],[3.7,2,1.6,.25,0]))
 if blur>.05:result=cv2.GaussianBlur(result,(0,0),blur)
 return result
def layer_from(p,m):return warp(p,m)
def door(n):
 im=plate('door_front_clean_v1.png');c=mat(1+max(0,n-4372)*.0005,-max(0,n-4372)*.15,0)
 im=over(im,fit('backgrounds/door_onlookers_v2.png',(0,169,254,595),c))
 # Render each individual from the shared independently generated layer, preserving proportions.
 p=art('door_pair');left=p.copy();left[:,p.shape[1]//2:]=0;right=p.copy();right[:,:p.shape[1]//2]=0
 im=over(im,warp(left,mul(c,mat(.38,125,129))))
 im=over(im,warp(right,mul(c,mat(.38,263,129))))
 return im
def close(n):
 im=grid(n);p=art('sakiko_close');s=.50
 # Eye line y~220; shoulders continue below active picture.
 c=camera('close',n);im=over(im,layer('sakiko_close',s,480-p.shape[1]*s/2,220-583*s,c))
 blur=float(np.interp(n,[4394,4398,4403,4407,4409],[3.3,2,1,.15,0]))
 return cv2.GaussianBlur(im,(0,0),max(.01,blur)) if blur>.05 else im
@functools.lru_cache(None)
def lyric_mask(kind):
 lines={'rain1':'都合の良い願いを同じ様に同じ様に呟く','rain2':'何処から聞こうか　何を見失うか','rain3':'定かじゃないから此処を動けない','dance1':'僕らが壊れるなら　これ以上無いなら','dance2':'その度に何回も逃げ出せる様に','dance3':'心が守れる様に　奪われない様に','end':'互いに託して　身体を預けてよ'}
 im=Image.new('L',(W,H));d=ImageDraw.Draw(im);font=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc'),11)
 t=lines[kind];y=160 if kind.startswith('rain') else 180
 for ch in t:d.text((907,y),ch,font=font,fill=220);y+=10.5
 return np.array(im).astype(np.float32)/255
def frame(n):
 if any(a<=n<b for a,b in BLACK):return bars(np.full((H,W,3),[25,24,30],np.float32))
 if 3933<=n<3972 or 4162<=n<4190 or 4292<=n<4313:
  return np.array(Image.open(P/'reference'/f'f{n}.jpg').convert('RGB'))
 if n<3933:im=rain(n);kind='rain1' if n<3722 else 'rain2' if n<3848 else 'rain3'
 elif n<4162:im=dance(n,'a');kind='dance1' if n<4060 else 'dance2'
 elif n<4292:im=dance(n,'b');kind='dance3'
 elif n<4338:im=poster(n);kind='end'
 elif n<4372:im=mouth(n);kind='end'
 elif n<4394:im=door(n);kind='end'
 else:im=close(n);kind='end'
 m=lyric_mask(kind);color=np.where(cv2.GaussianBlur(im.mean(2),(0,0),7)[:,:,None]>160,25,240)
 return bars(im*(1-m[:,:,None])+color*m[:,:,None])
def previews():
 ns=[3540,3650,3750,3885,3921,3927,3972,3985,4000,4018,4048,4070,4098,4127,4159,4190,4205,4230,4260,4285,4313,4325,4338,4353,4368,4380,4394,4409]
 for n in ns:save(Q/f'f{n}.jpg',frame(n))
 for start in range(0,len(ns),12):
  batch=ns[start:start+12];sheet=np.zeros(((len(batch)+1)//2*294,1920,3),np.uint8)
  for j,n in enumerate(batch):
   im=frame(n);src=np.array(Image.open(P/'reference'/f'f{n}.jpg'));pair=np.hstack([src,im]);pair=cv2.resize(pair,(960,270));x=j%2*960;y=j//2*294
   sheet[y+24:y+294,x:x+960]=pair;cv2.putText(sheet,f'{n} | source left / new right',(x+5,y+18),0,.5,(255,255,255),1)
  save(Q/f'compare_{start//12}.jpg',sheet)
def render(start=3540,end=4410,name='segment_118_147_v5'):
 silent=O/(name+'_silent.mp4');p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
 for n in range(start,end):
  p.stdin.write(frame(n).tobytes())
  if n%90==0:print('frame',n,flush=True)
 p.stdin.close();assert p.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(silent),'-ss',str(start/30),'-i',str(next(R.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t',str((end-start)/30),'-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(O/(name+'.mp4'))],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');ap.add_argument('--dance',action='store_true');args=ap.parse_args()
 if args.preview:previews()
 elif args.dance:render(3972,4162,'dance_test_v5')
 else:render()



