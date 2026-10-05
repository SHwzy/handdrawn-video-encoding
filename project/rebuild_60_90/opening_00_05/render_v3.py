"""60–65s only: independently drawn actors, measured camera, procedural reveals.

Source pixels are read only for comparison output, never as a composite layer.
"""
from pathlib import Path
import sys,json,math,subprocess,functools,argparse,hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import render as base
W,H,AH=960,540,434
OUT=HERE/'output'; QA=HERE/'qa_v3'
for p in (OUT,QA):p.mkdir(exist_ok=True)
Y,X=np.mgrid[:AH,:W].astype(np.float32)

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
def over(bg,p):return bg*(1-p[:,:,3:4])+p[:,:,:3]*255
def warp(p,m):return cv2.warpAffine(p,m,(W,AH),flags=cv2.INTER_LINEAR)
def rgba(path):
 p=np.array(Image.open(path).convert('RGBA')).astype(np.float32)/255
 p[:,:,:3]*=p[:,:,3:4];return p

@functools.lru_cache(None)
def actor():
 # Registered outpainting preserves the approved upper-body artwork exactly.
 # All inputs are independent generated artwork, never frames of the original PV.
 old=base.asset('opening_continuity')
 p=rgba(HERE/'assets/partner_extended_v2.png')
 m=np.array([[1.57133741,-.00276927317,-24.5262689],[.00276927317,1.57133741,-8.53969414]])*(900/1536)
 m[1,2]+=300
 ext=cv2.warpAffine(p,m,(900,1800))
 a=np.zeros((1800,900,1),np.float32);a[300:840]=1
 a[840:900,0,0]=1-smooth(np.arange(60)/59);a[840:900]=a[840:900,0:1]
 oldpad=np.zeros_like(ext);oldpad[300:900]=old
 result=oldpad*a+ext*(1-a)
 cap=rgba(HERE/'assets/partner_complete_v3.png')
 cm=np.array(json.loads((HERE/'assets/cap_registration.json').read_text())['matrix'])*(900/1536)
 cm[1,2]+=300
 top=cv2.warpAffine(cap,cm,(900,1800))
 # Outpainted crown enters from off-screen as the camera pulls back.
 mix=np.clip((340-np.arange(1800))/40,0,1)[:,None,None]
 return (result*(1-mix)+top*mix).astype(np.float32)

RAW=json.loads((HERE/'source_motion_raw.json').read_text())
TRACK=[r for r in RAW if 1799<=r['frame']<=1949 and .5<r['scale']<1.1 and r['inliers']>=16]
TRACK.sort(key=lambda r:r['frame'])
def motion(n):
 fs=[r['frame'] for r in TRACK]
 s=np.interp(n,fs,[r['scale'] for r in TRACK])/1.0177748216
 a=np.interp(n,fs,[r['angle_cv_deg'] for r in TRACK])+8.68+.48438075
 cx=np.interp(n,fs,[r['center'][0] for r in TRACK])+.23488
 cy=np.interp(n,fs,[r['center'][1] for r in TRACK])-1.15289
 m=cv2.getRotationMatrix2D((450,562),float(a),float(s))
 m[:,2]+=np.array([cx,cy])-np.array([450,562])
 return m,dict(frame=n,scale=float(s),angle=float(a),center=[float(cx),float(cy)])

@functools.lru_cache(None)
def plates(kind):
 p=np.full((1100,1600,3),(242,237,231),np.uint8);col=(193,93,120)
 step=160
 for y in range(-160,1260,step):
  for x in range(-160,1760,step):
   if kind==0:
    for dx,dy,ang in [(0,0,0),(160,0,90),(160,160,180),(0,160,270)]:
     cv2.ellipse(p,(x+dx,y+dy),(112,112),0,ang,ang+90,col,1,cv2.LINE_AA)
   elif kind==1:
    cv2.rectangle(p,(x+22,y+22),(x+138,y+138),col,1,cv2.LINE_AA)
    pts=np.array([[x+80,y],[x+160,y+80],[x+80,y+160],[x,y+80]],np.int32)
    cv2.polylines(p,[pts],True,col,1,cv2.LINE_AA)
    for j in [45,115]:cv2.line(p,(x+j,y),(x+j,y+160),col,1,cv2.LINE_AA);cv2.line(p,(x,y+j),(x+160,y+j),col,1,cv2.LINE_AA)
   elif kind==2:
    for dx,dy in [(55,55),(105,55),(55,105),(105,105)]:cv2.circle(p,(x+dx,y+dy),34,col,1,cv2.LINE_AA)
    cv2.polylines(p,[np.array([[x,y],[x+160,y+160],[x+160,y],[x,y+160]],np.int32)],True,col,1,cv2.LINE_AA)
   elif kind==3:
    for z in [0,6,13]:cv2.rectangle(p,(x+z,y+z),(x+160-z,y+160-z),col,1,cv2.LINE_AA)
   elif kind==4:
    for z in [0,18,30]:
     cv2.polylines(p,[np.array([[x+80,y+z],[x+160-z,y+160-z],[x+z,y+160-z]],np.int32)],True,col,1,cv2.LINE_AA)
   else:
    a=np.arange(16)*math.tau/16;r=np.where(np.arange(16)%2,53,78)
    pts=np.rint(np.c_[x+80+np.cos(a)*r,y+80+np.sin(a)*r]).astype(np.int32)
    cv2.polylines(p,[pts],True,col,1,cv2.LINE_AA)
 p[::3,::3]=(p[::3,::3]*.94).astype(np.uint8)
 return p.astype(np.float32)

def backdrop(n):
 key=next(k for end,k in [(1805,0),(1819,1),(1832,2),(1850,3),(1860,0),(1875,4),(1887,5),(1950,2)] if n<end)
 p=plates(key);m=cv2.getRotationMatrix2D((800,550),-(n-1800)*.105,1.05)
 m[:,2]+=np.array([480,217])-np.array([800,550])
 bg=warp(p,m)
 # Two-frame continuation of approved endpoint into the changing ornament track.
 t=smooth((n-1800)/5)
 return base.continuity_grid(n-1800)*(1-t)+bg*t

@functools.lru_cache(None)
def pattern_texture():
 p=np.full((700,1300,3),(78,70,123),np.uint8)
 rng=np.random.default_rng(4)
 for i in range(520):
  x,y=rng.integers(-50,1300),rng.integers(-50,700);r=rng.integers(14,36);a=rng.uniform(0,math.tau)
  pts=np.array([[x+r*math.cos(a+j*math.tau/3),y+r*math.sin(a+j*math.tau/3)] for j in range(3)],np.int32)
  cv2.polylines(p,[pts],True,[(105,96,149),(128,116,165),(66,61,110)][i%3],int(rng.integers(7,14)),cv2.LINE_AA)
 return p.astype(np.float32)

def texture(n):
 m=cv2.getRotationMatrix2D((650,350),-10+(n-1885)*.12,1)
 m[:,2]+=np.array([480,217])-np.array([650,350])
 return warp(pattern_texture(),m)

def ribbons(n,delay=0):
 t=n-delay
 if t<1884:return np.zeros_like(X)
 if t>=1905:return np.ones_like(X)
 tip=np.interp(t,[1884,1888,1894,1905],[0,750,1150,1500])
 centre=221-30*np.sin(X*.007)+8*np.sin(X*.023+t*.12)
 width=np.interp(t,[1884,1889,1895,1900,1905],[0,49,132,267,490])
 taper=np.clip((tip-X)/280,0,1)
 wav=9*np.sin(X*.011+t*.04)+5*np.sin(X*.027)
 d=width*taper+wav-np.abs(Y-centre)
 return smooth((d+1)/2)*smooth((tip-X)/6)

def return_reveal(n,delay=0):
 t=n-delay
 if t<=1913:return np.zeros_like(X)
 if t>=1930:return np.ones_like(X)
 cx=482+23*np.sin(Y*.008)+(217-Y)*.15
 width=np.interp(t,[1913,1917,1922,1926,1930],[0,8,39,90,650])
 width=width*np.clip((Y-40+(t-1913)*12)/220,.08,1.3)
 return smooth((width-np.abs(X-cx)+1.5)/3)

def style_actor(p,n,mode):
 a=p[:,:,3:4];rgb=p[:,:,:3]/np.maximum(a,.0001)
 h,w=a.shape[:2];yy,xx=np.mgrid[:h,:w];yy=yy-300
 # Authored separation follows arm/hand overlap in the independent artwork.
 boundary=np.interp(yy[:,0],[0,210,290,370,440,560,1500],[450,444,475,435,365,330,320])[:,None]
 left=(xx<boundary)[:,:,None]
 ink=np.clip((.65-rgb.mean(axis=2,keepdims=True))*3,0,1)
 teal=np.array([54,128,127])/255;pink=np.array([230,64,124])/255
 colour=np.where(left,teal,pink)
 if mode=='pattern':
  pat=(np.sin(xx*.11+yy*.04)>0)&(np.sin(yy*.13-xx*.08)>.3)
  rgb=colour*(.82+.18*pat[:,:,None])
 else:
  gray=np.full_like(rgb,.70)
  darkcloth=((rgb[:,:,1]<.75)&(rgb[:,:,1]>rgb[:,:,0]*1.35)&(yy>230))[:,:,None]
  rgb=gray*(1-ink)+colour*ink
  rgb=np.where(darkcloth&left,np.array([.06,.09,.10]),rgb)
 return np.concatenate([rgb*a,a],axis=2).astype(np.float32)

def outline(p,colour,r=1):
 a=p[:,:,3];outer=cv2.dilate(a,np.ones((r*2+1,r*2+1),np.uint8))-a
 out=np.dstack([outer[:,:,None]*np.array(colour)[None,None,:]/255,outer])
 return out.astype(np.float32)

def flowers(bg,n):
 return base.petals(bg,n-1930,seed=21,count=26,colour=(232,70,119),speed=.5,flowers=True)

def swapping(n,outlined_bg,oldfg):
 if n<1941:return outlined_bg
 p=rgba(HERE/'assets/swapped_pair_v3.png');h,w=p.shape[:2]
 # Both complete new characters enter in staggered masks; no source heads/bodies.
 m=cv2.getRotationMatrix2D((w*.50,h*.38),0,.49)
 m[:,2]+=np.array([478,205])-np.array([w*.50,h*.38])
 new=warp(p,m)
 # Several staggered curved strips share fixed spines; no animated random noise.
 # Their edges accelerate around the hat, face and sleeves separately.
 progress=n-1941
 spine=650-progress*24-.24*(Y-217)
 bend=18*np.sin(Y*.020)+9*np.sin(Y*.047+.5)
 primary=smooth((X-spine-bend)/3)
 ribbon=np.zeros_like(X)
 for y0,delay,amp in [(80,0,25),(195,1.2,34),(322,2.0,29)]:
  phase=max(0,progress-delay)
  center=665-phase*27-.20*(Y-217)+amp*np.sin(Y*.018+y0*.01)
  width=min(30,phase*5)
  local=smooth((width-np.abs(X-center))/2)
  envelope=smooth((100-np.abs(Y-y0))/25)
  ribbon=np.maximum(ribbon,local*envelope)
 right=np.maximum(primary,ribbon)
 left=smooth(((n-1951)*63-np.abs(X-392)+.12*(Y-217)+8*np.sin(Y*.035))/3)
 mask=np.maximum(right,left)[:,:,None]
 # Reveal a separately patterned background as well as both entire figures.
 bg=np.full((AH,W,3),(218,215,220),np.float32)
 # Fine pale rosettes behind the arriving characters, separately revealed.
 for cy in range(-40,AH+100,100):
  for cx in range(-40,W+100,100):
   for dx,dy in [(0,-18),(18,0),(0,18),(-18,0)]:
    cv2.circle(bg,(cx+dx,cy+dy),27,(230,227,232),1,cv2.LINE_AA)
 bg[::3,::3]*=.96
 newscene=over(bg,new)
 out=outlined_bg*(1-mask)+newscene*mask
 # Foreground flower expands and then travels independently of the reveal.
 if n>=1943:
  radius=np.interp(n,[1943,1946,1949],[15,48,15]);cx=397-(n-1943)*7;cy=235
  for j in range(5):
   theta=j*math.tau/5+n*.02
   cv2.circle(out,(int(cx+radius*.5*math.cos(theta)),int(cy+radius*.5*math.sin(theta))),max(1,int(radius*.53)),(230,66,116),-1,cv2.LINE_AA)
 return out

def frame(n):
 m,track=motion(n);p=actor()
 fg=warp(p,m)
 f1=warp(style_actor(p,n,'pattern'),m)
 f2=warp(style_actor(p,n,'outline'),m)
 rb=ribbons(n);rc=np.where(X<470,ribbons(n,0),ribbons(n,1));re=return_reveal(n);rechar=return_reveal(n,1)
 bg=backdrop(n);tex=texture(n)
 bg=bg*(1-rb[:,:,None])+tex*rb[:,:,None]
 white=np.full_like(bg,(239,239,238))
 if n>=1914:
  for r in [188,195,202]:cv2.ellipse(white,(475,206),(r,int(r*.9)),0,0,360,(22,25,29),1,cv2.LINE_AA)
  for j in range(40):
   angle=j*math.tau/40+(n-1930)*.0007;r=218+12*(j%3)
   cv2.line(white,(round(475+193*math.cos(angle)),round(206+174*math.sin(angle))),(round(475+r*math.cos(angle)),round(206+r*.9*math.sin(angle))),(22,25,29),2,cv2.LINE_AA)
  white=flowers(white,n)
 bg=bg*(1-re[:,:,None])+white*re[:,:,None]
 # Independent background, left actor, right actor, contour and foreground tracks.
 weights=np.stack([(1-rc)*(1-rechar),rc*(1-rechar),rechar],axis=2)
 assert np.max(np.abs(weights.sum(axis=2)-1))<1e-5
 outfg=fg*weights[:,:,0:1]+f1*weights[:,:,1:2]+f2*weights[:,:,2:3]
 out=over(bg,outline(outfg,(61,85,97),1));out=over(out,outfg)
 out=swapping(n,out,outfg)
 out=base.caption(out,'partner',white=1903<=n<1927)
 result=np.zeros((H,W,3),np.uint8);result[53:487]=np.clip(out,0,255)
 debug=np.zeros((H,W,3),np.uint8);debug[53:487]=np.uint8(weights*255)
 assert np.isfinite(out).all()
 return cv2.cvtColor(result,cv2.COLOR_RGB2BGR),cv2.cvtColor(debug,cv2.COLOR_RGB2BGR),track

def ff(args):base.ff(args)
def main(preview=False):
 selected=[1800,1805,1815,1830,1860,1885,1890,1895,1900,1905,1915,1920,1925,1930,1938,1943,1946,1949]
 cap=cv2.VideoCapture(str(base.SRC));cap.set(cv2.CAP_PROP_POS_FRAMES,1800)
 procs=[]
 if not preview:
  for name in ['silent_v3','masks_v3']:
   procs.append(subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','960x540','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(OUT/f'{name}.mp4')],stdin=subprocess.PIPE))
 pairs=[];tracks=[]
 for n in range(1800,1950):
  ok,src=cap.read();assert ok
  if preview and n not in selected:continue
  im,dbg,tr=frame(n);tracks.append(tr)
  if not preview:procs[0].stdin.write(im.tobytes());procs[1].stdin.write(dbg.tobytes())
  if n in selected:
   cv2.imencode('.jpg',im)[1].tofile(str(QA/f'f{n}.jpg'))
   pair=np.hstack([cv2.resize(src,(480,270)),cv2.resize(im,(480,270))]);pairs.append((n,pair))
 for proc in procs:proc.stdin.close();assert proc.wait()==0
 for page in range(math.ceil(len(pairs)/6)):
  items=pairs[page*6:(page+1)*6];sheet=np.full((len(items)*292,960,3),25,np.uint8)
  for j,(n,pair) in enumerate(items):
   sheet[j*292+22:(j+1)*292]=pair;cv2.putText(sheet,f'{n} / {(n-1800)/30:.3f}s   SOURCE | NEW LAYERS',(8,j*292+16),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
  cv2.imencode('.jpg',sheet)[1].tofile(str(QA/f'comparison_{page}.jpg'))
 if not preview:
  ff(['-i',OUT/'silent_v3.mp4','-ss','60','-i',base.SRC,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-t','5','-movflags','+faststart',OUT/'opening_00_05_v3.mp4'])
  ff(['-ss','60','-t','5','-i',base.SRC,'-i',OUT/'opening_00_05_v3.mp4','-filter_complex','[0:v]scale=960:540,setsar=1[l];[l][1:v]hstack=inputs=2[v]','-map','[v]','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','18','-c:a','copy','-t','5','-movflags','+faststart',OUT/'source_vs_opening_00_05_v3.mp4'])
 (QA/'motion_applied.json').write_text(json.dumps(tracks,indent=2),encoding='utf-8')
 print(json.dumps(dict(preview=preview,frames=len(tracks))))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');main(parser.parse_args().preview)
