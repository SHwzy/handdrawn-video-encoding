"""Source frames [1800,2443): keep accepted material; rebuild marked effect tracks."""
from pathlib import Path
import sys,importlib.util,functools,json,math,subprocess,argparse
import cv2,numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
import render as base
spec=importlib.util.spec_from_file_location('opening',HERE.parent/'opening_00_05/render_v3.py');opening=importlib.util.module_from_spec(spec);spec.loader.exec_module(opening)
W,H,AH=960,540,434;OUT=HERE/'output';QA=HERE/'qa_v5'
for p in [OUT,QA]:p.mkdir(exist_ok=True)
Y,X=np.mgrid[:AH,:W].astype(np.float32)
def smooth(t):return opening.smooth(t)
def mix(a,b,m):
 if np.ndim(m)==2:m=m[:,:,None]
 return a*(1-m)+b*m
def over(bg,p):return bg*(1-p[:,:,3:4])+p[:,:,:3]*255
def warp(p,m):return cv2.warpAffine(p,m,(W,AH),flags=cv2.INTER_LINEAR)
def place(p,anchor,dest,scale,angle=0):
 m=cv2.getRotationMatrix2D(tuple(map(float,anchor)),float(angle),float(scale));m[:,2]+=np.array(dest)-np.array(anchor)
 return warp(p,m)
@functools.lru_cache(None)
def art(name):
 if name.startswith('local:'):
  p=np.array(Image.open(HERE/'assets'/name[6:]).convert('RGBA')).astype(np.float32)/255
  assert p[:,:,3].min()==0 and p[:,:,3].max()>.95
  p[:,:,:3]*=p[:,:,3:4];return p
 return base.asset(name)
@functools.lru_cache(None)
def field(seed):
 rng=np.random.default_rng(seed)
 coarse=cv2.resize(rng.random((7,14)).astype(np.float32),(W,AH),interpolation=cv2.INTER_CUBIC)
 medium=cv2.resize(rng.random((19,41)).astype(np.float32),(W,AH),interpolation=cv2.INTER_CUBIC)
 return (coarse-.5)*.18+(medium-.5)*.055
def reveal(n,a,b,seed=1,direction='right'):
 if n<=a:return np.zeros((AH,W),np.float32)
 if n>=b:return np.ones((AH,W),np.float32)
 q=(n-a)/(b-a)
 if direction=='right':f=X/W+.10*np.sin(Y*.018)+.05*np.sin(Y*.053)
 elif direction=='left':f=1-X/W+.10*np.sin(Y*.016)+.06*np.sin(Y*.043)
 else:f=Y/AH+.13*np.sin(X*.012)+.05*np.sin(X*.037)
 # Folded field creates islands and openings, rather than one broad wavy edge.
 f=f+.10*np.sin(X*.021+Y*.027)+field(seed)*2.2
 return smooth((q*1.65-.32-f)/.007+.5)
def gray(p):
 q=p.copy();l=np.sum(q[:,:,:3]*np.array([.299,.587,.114]),axis=2,keepdims=True);q[:,:,:3]=l;return q
@functools.lru_cache(None)
def styled(name,mode,who):
 p=art(name);a=p[:,:,3:4];rgb=p[:,:,:3]/np.maximum(a,.0001);h,w=a.shape[:2];yy,xx=np.mgrid[:h,:w]
 if who=='pair':left=(xx<w*.57)[:,:,None]
 else:left=np.full((h,w,1),who=='sakiko')
 color=np.where(left,np.array([240,104,69])/255,np.array([226,65,126])/255)
 if mode=='outline':
  ink=np.clip((.72-rgb.mean(axis=2,keepdims=True))*4,0,1)
  out=(.67*(1-ink)+color*ink)
  dark=(rgb.mean(axis=2,keepdims=True)<.28)
  out=np.where(dark,np.array([.06,.055,.07]),out)
 else:
  # Actor-local print motifs follow the same transform as the silhouette.
  dot=(((xx%22)-11)**2+((yy%22)-11)**2<30)[:,:,None]
  u=xx%150-75;v=yy%150-75
  motif=(((np.abs(u)<5)&(np.abs(v)<48))|((np.abs(v)<5)&(np.abs(u)<48))|((np.abs(np.abs(u)-np.abs(v))<5)&(np.abs(u)<35)))[:,:,None]
  yellow=np.array([234,194,89])/255
  orange=np.array([238,103,71])/255
  head=(yy<h*(.60 if who=='sakiko' else .31))[:,:,None]
  maid=np.where(head,orange,yellow)
  maid=maid*(1-.22*dot*(1-head))
  witch=(np.array([224,66,125])/255)*(1-.32*motif)
  out=np.where(left,maid,witch)
 return np.concatenate([out*a,a],axis=2).astype(np.float32)
def edge(p,color=(60,49,64)):
 a=p[:,:,3];d=cv2.dilate(a,np.ones((3,3),np.uint8))-a
 return np.dstack([d[:,:,None]*np.array(color)/255,d]).astype(np.float32)
def camera(n):
 if n<2227:
  return dict(angle=0.,scale=1.,dx=float(np.interp(n,[2200,2205,2210,2214,2220,2226],[188,88,27,0,-13,-17])),dy=float(np.interp(n,[2200,2210,2214,2226],[-55,-13,0,9])))
 if n<2335:
  ns=[2227,2236,2245,2260,2270,2280,2295,2310,2335]
  return dict(angle=float(np.interp(n,ns,[15.8,10.25,5.37,0,-2.6,-4.7,-7.4,-9.2,-11.5])+3),scale=float(np.interp(n,ns,[.83,.67,.56,.48,.445,.425,.397,.38,.355])/.48),dx=26.,dy=-2.)
 k=n-(2335 if n<2389 else 2389);q=k/53
 return dict(angle=(2.5-7*q if n<2389 else 3-6*q),scale=1.04-.065*q,dx=(16-38*q if n<2389 else 12-29*q),dy=-7+17*q)

def world(n):
 c=camera(n)
 return c['angle'],1+(c['scale']-1)*.28,c['dx'],c['dy']

def sea(n):
 angle,z,dx,dy=world(n)
 # Overscan the source plate before sampling; no reflected or blank edges.
 return base.background('sea_wide_v1',0,zoom=1.82*z,dx=dx,dy=dy,angle=angle,blur=.2)

def world_plate(p,n):
 angle,z,dx,dy=world(n);h,w=p.shape[:2]
 m=cv2.getRotationMatrix2D((w/2,h/2),angle,z)
 m[:,2]+=np.array([W/2+dx,AH/2+dy])-np.array([w/2,h/2])
 return warp(p.astype(np.float32),m)

def pattern(n,who='pair'):
 p=opening.plates((n//14)%6)
 if who=='pair':p=np.repeat(p.mean(axis=2,keepdims=True),3,axis=2)
 elif who=='sakiko':
  ink=np.clip((245-p.mean(axis=2))/80,0,1)[:,:,None];p=242*(1-ink)+np.array([198,166,78])*ink
 return world_plate(p,n)

@functools.lru_cache(None)
def pale_plate():
 p=np.full((1100,1600,3),(226,224,228),np.float32);p[::3,::3]*=.965
 rng=np.random.default_rng(9)
 for x,y,r in zip(rng.integers(0,1600,65),rng.integers(0,1100,65),rng.integers(5,15,65)):
  for j in range(5):
   a=j*math.tau/5;cv2.circle(p,(int(x+r*.6*math.cos(a)),int(y+r*.6*math.sin(a))),int(r*.55),(249,248,245),-1,cv2.LINE_AA)
 return p
def pale(n):return world_plate(pale_plate(),n)
@functools.lru_cache(maxsize=20)
def glitch_tex(t):
 rng=np.random.default_rng(673+t)
 f=cv2.resize(rng.random((15,27)).astype(np.float32),(W,AH),interpolation=cv2.INTER_CUBIC)
 g=cv2.resize(rng.random((36,75)).astype(np.float32),(W,AH),interpolation=cv2.INTER_NEAREST)
 fine=rng.random((AH,W)).astype(np.float32)
 score=f*.75+g*.2+fine*.05
 palette=np.array([[63,18,108],[112,19,204],[148,18,195],[234,22,111],[225,37,57],[203,25,86],[17,63,244],[37,221,112],[33,205,215],[216,192,211]],np.float32)
 colour_field=cv2.resize(rng.random((8,15)).astype(np.float32),(W,AH),interpolation=cv2.INTER_CUBIC)
 idx=np.clip((colour_field*10).astype(int),0,9)
 tex=palette[idx]*(.74+.26*fine[:,:,None])
 tex[::3]*=.7
 alpha=np.clip((f*.65+g*.35-.46)*5,0,.93)*(fine>.19)
 return tex,alpha
def glitch(bg,n,strength=1):
 tex,a=glitch_tex((n-2200)//2)
 return mix(bg,tex,a*strength)
def pack(bg):
 out=np.zeros((H,W,3),np.uint8);out[53:487]=np.clip(bg,0,255)
 return cv2.cvtColor(out,cv2.COLOR_RGB2BGR)

def swap_tail(n,outgoing,oldfg):
 p=opening.rgba(HERE.parent/'opening_00_05/assets/swapped_pair_v3.png');h,w=p.shape[:2]
 scale=.49*(1-.0012*max(0,n-1950))
 fg=place(p,(w*.5,h*.38),(478,205),scale,15)
 progress=n-1941;spine=650-progress*24-.24*(Y-217)
 mask=smooth((X-spine-18*np.sin(Y*.020)-9*np.sin(Y*.047+.5))/3)
 for y0,delay,amp in [(80,0,25),(195,1.2,34),(322,2,29)]:
  t=max(0,progress-delay);center=665-t*27-.20*(Y-217)+amp*np.sin(Y*.018+y0*.01)
  mask=np.maximum(mask,smooth((min(30,t*5)-np.abs(X-center))/2)*smooth((100-np.abs(Y-y0))/25))
 left=smooth(((n-1951)*63-np.abs(X-392)+.12*(Y-217)+8*np.sin(Y*.035))/3)
 mask=np.maximum(mask,left)
 if n>=1958:mask=np.ones_like(mask)
 bg=np.full((AH,W,3),(218,215,220),np.float32)
 for cy in range(-40,AH+100,100):
  for cx in range(-40,W+100,100):
   for dx,dy in [(0,-18),(18,0),(0,18),(-18,0)]:cv2.circle(bg,(cx+dx,cy+dy),27,(230,227,232),1,cv2.LINE_AA)
 result=mix(outgoing,over(bg,fg),mask)
 if 1943<=n<=1953:
  r=np.interp(n,[1943,1946,1949,1953],[15,48,15,0]);cx=397-(n-1943)*7
  for j in range(5):
   a=j*math.tau/5+n*.02;cv2.circle(result,(int(cx+r*.5*math.cos(a)),int(235+r*.5*math.sin(a))),max(1,int(r*.53)),(230,66,116),-1,cv2.LINE_AA)
 return result
opening.swapping=swap_tail
old_backdrop=opening.backdrop
opening.backdrop=lambda n:old_backdrop(min(n,1949))

def locked(name):
 # Screenshot-edited full 16:9 canvas. Preserve source crop and body angle.
 p=art('local:'+name);p=cv2.resize(p,(W,H),interpolation=cv2.INTER_AREA)
 return p[53:487]
def turn(n):
 name='turn_locked_b.png' if ((n-2200)//4)%2 else 'turn_locked_a.png'
 p=cv2.resize(art('local:'+name),(W,H),interpolation=cv2.INTER_AREA)
 dx=np.interp(n,[2200,2205,2210,2214,2220,2226],[188,88,27,0,-13,-17])
 dy=np.interp(n,[2200,2210,2214,2226],[-55,-13,0,9])
 p=cv2.warpAffine(p,np.float32([[1,0,dx],[0,1,dy]]),(W,H),flags=cv2.INTER_LINEAR)[53:487]
 bg=sea(n);desat=smooth((n-2199.36)/20)
 graybg=cv2.cvtColor(bg.astype(np.uint8),cv2.COLOR_RGB2GRAY)[:,:,None].repeat(3,axis=2).astype(np.float32)
 bg=mix(bg,graybg,desat)
 bg=mix(bg,np.full_like(bg,239),desat*.87)
 p=mix(p,gray(p),desat)
 result=over(bg,p)
 if n>=2223:result=glitch(result,n,min(1,(n-2223)/2))
 return result

def dance(n):
 ns=[2227,2236,2245,2260,2270,2280,2295,2310,2335]
 scales=[.83,.67,.56,.48,.445,.425,.397,.38,.355]
 angles=[15.8,10.25,5.37,0,-2.6,-4.7,-7.4,-9.2,-11.5]
 s=np.interp(n,ns,scales)*1.25;angle=np.interp(n,ns,angles)+3
 anchor=(850,420);dest=(506,215)
 pf=lambda p:place(p,anchor,dest,s,angle)
 p0=pf(art('local:dance_pair_surprised_v5.png'));p1=pf(styled('local:dance_pair_surprised_v5.png','outline','pair'));p2=pf(styled('local:dance_pair_surprised_v5.png','solid','pair'))
 # Glitch layer is beneath all subsequent appearance masks.
 scene0=over(sea(n),p0)
 if n<2239:scene0=glitch(scene0,n,1-smooth((n-2230)/9))
 mb=reveal(n,2228,2241,13);mc=reveal(n,2231,2243,13)
 bg=mix(sea(n),pattern(n),mb)
 fg=mix(p0,p1,mc)
 scene1=over(over(bg,edge(fg)),fg)
 scene=mix(scene0,scene1,np.maximum(mb,mc))
 mb2=reveal(n,2253,2269,31,'down');mc2=reveal(n,2255,2271,31,'down')
 bg2=mix(bg,pale(n),mb2);fg2=mix(fg,p2,mc2)
 scene=mix(scene,over(bg2,fg2),np.maximum(mb2,mc2))
 mb3=reveal(n,2279,2294,44);mc3=reveal(n,2280,2295,44)
 bg3=mix(bg2,sea(n),mb3);fg3=mix(fg2,p0,mc3)
 scene=mix(scene,over(bg3,fg3),np.maximum(mb3,mc3))
 if n>=2311:
  # Hallucinatory vivid patches live behind the actors, not a floral/ink plate.
  scene=over(glitch(sea(n),n,min(1,(n-2310)/4)),p0)
 return scene

def closeup(n,who):
 anon=who=='anon';start=2335 if anon else 2389;k=n-start
 c=camera(n);ca=c['angle'];cs=c['scale'];cx=c['dx'];cy=c['dy']
 if anon:
  name='anon_front_v1';p=art(name);anchor=(630,450);dest=(604+cx,148+cy);scale=.69*cs
  get=lambda p:place(p,anchor,dest,scale,ca)
 else:
  name='local:sakiko_close_locked.png'
  # Exact screenshot-based body/face inclination; camera settles after the cut.
  p=cv2.resize(art(name),(W,H),interpolation=cv2.INTER_AREA)
  m=cv2.getRotationMatrix2D((480,270),ca,cs);m[:,2]+=np.array([cx,cy])
  get=lambda p:cv2.warpAffine(p,m,(W,H),flags=cv2.INTER_LINEAR)[53:487]
 p0=get(p)
 if anon:
  p1=get(styled(name,'outline',who));p2=get(styled(name,'solid',who))
 else:
  # Render styles in full screenshot coordinates before the identical crop.
  a=styled(name,'outline',who);b=styled(name,'solid',who)
  p1=get(cv2.resize(a,(W,H),interpolation=cv2.INTER_AREA));p2=get(cv2.resize(b,(W,H),interpolation=cv2.INTER_AREA))
 mb=reveal(k,0,12,54,'left');mc=reveal(k,2 if anon else 6,12 if anon else 18,54,'left')
 bg=mix(sea(n),pattern(n,who),mb);fg=mix(p0,p1,mc)
 msb=reveal(k,14,23 if anon else 27,64,'down');msc=reveal(k,14 if anon else 17,23 if anon else 30,64,'down')
 bg=mix(bg,pale(n),msb);fg=mix(fg,p2,msc)
 mrb=reveal(k,29 if anon else 32,42 if anon else 47,74);mrc=reveal(k,30 if anon else 34,41 if anon else 49,74)
 bg=mix(bg,sea(n),mrb);fg=mix(fg,p0,mrc)
 return over(over(bg,edge(fg)),fg)

def rebuilt(n):
 if 1941<=n<1973:return opening.frame(n)[0]
 if 2200<=n<2227:bg=turn(n);shot='turn'
 elif 2227<=n<2335:bg=dance(n);shot='dance'
 elif n<2389:
  bg=closeup(n,'anon');shot='anon_close'
  if n<2343:
   t=smooth((n-2335)/7);off=round(W*t)
   old=dance(2334);whole=np.hstack([old,bg]);bg=whole[:,off:off+W]
 elif n<2443:
  bg=closeup(n,'sakiko');shot='sakiko_close'
  if n<2397:
   t=smooth((n-2389)/7);off=round(W*t)
   old=closeup(2388,'anon');whole=np.hstack([old,bg]);bg=whole[:,off:off+W]
 else:raise ValueError(n)
 return pack(base.caption(bg,shot,white=(2200<=n<2212 or 2295<=n<2311 or n>=2433)))

def main(preview=False):
 selected={1940,1941,1949,1950,1953,1958,1972,1973,2199,2200,2204,2208,2213,2218,2223,2225,2227,2232,2237,2245,2257,2263,2271,2284,2294,2303,2311,2317,2326,2334,2335,2338,2342,2346,2353,2360,2368,2377,2388,2389,2395,2403,2413,2421,2430,2442}
 source=cv2.VideoCapture(str(base.SRC));source.set(cv2.CAP_PROP_POS_FRAMES,1800)
 previous=cv2.VideoCapture(str(HERE.parent/'output/rebuild_60_90_v1.mp4'))
 approved_open=cv2.VideoCapture(str(HERE.parent/'opening_00_05/output/opening_00_05_v3.mp4'))
 proc=None;records=[];pairs=[]
 if not preview:proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','960x540','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(OUT/'silent_v5.mp4')],stdin=subprocess.PIPE)
 for n in range(1800,2443):
  ok,src=source.read();assert ok
  ok,old=previous.read();assert ok
  if n<1950:ok,op=approved_open.read();assert ok
  if preview and n not in selected:continue
  if n<1941:im=op;mode='opening_v3_reuse'
  elif 1973<=n<2200:im=old;mode='user_accepted_v1_reuse'
  else:im=rebuilt(n);mode='rebuilt'
  assert im.shape==(540,960,3) and np.isfinite(im).all()
  records.append(dict(frame=n,mode=mode))
  if proc:proc.stdin.write(im.tobytes())
  if n in selected:
   cv2.imencode('.jpg',im)[1].tofile(str(QA/f'f{n}.jpg'))
   pair=np.hstack([cv2.resize(src,(480,270)),cv2.resize(im,(480,270))]);pairs.append((n,pair))
 if proc:
  proc.stdin.close();assert proc.wait()==0
  base.ff(['-i',OUT/'silent_v5.mp4','-ss','60','-i',base.SRC,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-t',str(643/30),'-movflags','+faststart',OUT/'review_00_21_v5.mp4'])
  base.ff(['-ss','60','-t',str(643/30),'-i',base.SRC,'-i',OUT/'review_00_21_v5.mp4','-filter_complex','[0:v]scale=960:540,setsar=1[l];[l][1:v]hstack=inputs=2[v]','-map','[v]','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','18','-c:a','copy','-t',str(643/30),'-movflags','+faststart',OUT/'source_vs_review_00_21_v5.mp4'])
  base.ff(['-ss','5','-i',OUT/'review_00_21_v5.mp4','-c:v','libx264','-crf','17','-c:a','aac','-t',str(493/30),'-movflags','+faststart',OUT/'segment_05_21_v5.mp4'])
 for page in range(math.ceil(len(pairs)/6)):
  items=pairs[page*6:page*6+6];sheet=np.full((292*len(items),960,3),25,np.uint8)
  for i,(n,pair) in enumerate(items):
   sheet[i*292+22:(i+1)*292]=pair;cv2.putText(sheet,f'{n} | {(n-1800)/30:.3f}s | SOURCE / REBUILD',(5,i*292+16),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1)
  cv2.imencode('.jpg',sheet)[1].tofile(str(QA/f'comparison_{page}.jpg'))
 (QA/'camera_tracks.json').write_text(json.dumps([dict(frame=n,actor=camera(n),background=dict(angle=world(n)[0],scale=world(n)[1],dx=world(n)[2],dy=world(n)[3])) for n in range(2200,2443)],indent=2),encoding='utf-8')
 (QA/'coverage.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
 print(json.dumps(dict(preview=preview,frames=len(records))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');main(p.parse_args().preview)
