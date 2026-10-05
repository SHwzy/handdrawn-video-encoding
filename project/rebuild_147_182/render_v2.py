"""v2: two authored actor states + separate backgrounds, shared scene matte."""
from pathlib import Path
import cv2,numpy as np,json,functools,subprocess,argparse
from PIL import Image,ImageDraw
import render as b
P=b.P;R=b.R;V=P/'v2';Q=V/'qa';O=b.O;X,Y=b.X,b.Y;W,H=b.W,b.H
SOURCE_REUSE=[(4469,4525),(5380,5460)]
@functools.lru_cache(None)
def pair(name):
 normal_path='assets/stand.png' if name=='stand' else 'v2/assets/'+name+'_normal.png'
 alt_path='v2/assets/'+name+('_dark.png' if name=='panic' else '_alt.png')
 if name=='offer':normal_path='v2/assets/offer_overhead_normal.png';alt_path='v2/assets/offer_overhead_alt.png'
 a=b.art(normal_path).copy();d=b.art(alt_path).copy()
 if a.shape!=d.shape:d=cv2.resize(d,(a.shape[1],a.shape[0]))
 return a,d

def relative(name,n,ref):return b.mul(b.camera(name,n),cv2.invertAffineTransform(b.camera(name,ref)))
def actors(name,n):
 a,d=pair(name)
 if name=='stand':m=b.mul(b.camera(name,n),b.fitm('assets/stand.png',430,-20,height=950))
 elif name=='offer':m=b.mat(.548+(n-4525)*.00015,-25,-5)
 elif name=='book':m=b.mul(relative(name,n,4624),b.mat(.54,85,-15))
 else:m=b.mul(relative(name,n,4790),b.mat(.63,-35,0))
 return b.warp(a,m),b.warp(d,m)

def bg(name,n):
 if name=='stand':return b.plate('backgrounds/hall_v1.png',dy=(n-4434)*.6,zoom=1.18,light=.10)
 if name=='offer':return b.plate('v2/backgrounds/offer_bg.png',zoom=1.13+(n-4525)*.0003)
 if name=='book':return b.warp(b.plate('v2/backgrounds/book_bg.png',zoom=1.22),relative(name,n,4624))
 return b.warp(b.plate('v2/backgrounds/panic_bg.png',zoom=1.55),relative(name,n,4790))

def scene_mask(n,start,end,direction=1,kind=0):
 # Folded advancing world boundary, with genuine enclosed islands and curved branches.
 if n<=start:return np.zeros((H,W),np.float32)
 if n>=end:return np.ones((H,W),np.float32)
 t=(n-start)/(end-start);u=(X/W if direction==1 else 1-X/W);v=(Y-53)/434
 field=u+.25*np.sin(v*5.8+u*3.4+kind)+.13*np.sin(v*11-u*6+kind*.7)
 field+=.11*np.sin(u*14+v*5)+.035*(b.noise()-.5)
 m=b.smooth((t*1.85-.40-field)/.008+.5)
 # Secondary closed openings travel with the boundary, not with actors.
 cx=1150-1450*t if direction==-1 else -180+1350*t
 hole=((X-cx-90*np.sin(v*5)) / (120+45*np.sin(t*3.14)))**2+((Y-235)/(160))**2
 aperture=b.smooth((1-hole)*12);island=b.smooth((.18-((X-cx-200)/150)**2-((Y-400)/90)**2)*15)
 m=np.clip(m*(1-aperture*.96)+island,0,1)
 return m

def states(name,n):
 a,d=actors(name,n);real=bg(name,n)
 if name in ['stand','offer']:
  alt=np.full_like(real,[247,246,246]);alt[::3,::3]*=.97
 elif name=='book':alt=b.pattern(1)
 else:alt=b.pattern(0)
 return b.over(real,a),b.over(alt,d)

def early(n,diagnostic=False):
 if n<4469:
  name='stand';m=scene_mask(n,4420,4471,1,0)
 elif n<4574:
  name='offer';m=scene_mask(n,4540,4578,-1,1)
 elif n<4627:
  name='book';m=1-scene_mask(n,4582,4632,1,2)
 else:
  name='panic';m=scene_mask(n,4615,4700,1,2)*(1-scene_mask(n,4737,4791,-1,0))
 a,d=states(name,n)
 result=b.mix(a,d,m) # BOTH complete worlds use precisely the same screen-space matte.
 if diagnostic:return a,d,m,result
 sigma=float(np.interp(n,[4410,4425],[2.5,0])) if n<4469 else (float(np.interp(n,[4627,4642],[2,0])) if name=='panic' else 0)
 if sigma>.05:result=cv2.GaussianBlur(result,(0,0),sigma)
 return result

@functools.lru_cache(None)
def book_layer():
 a=np.zeros((H,W,4),np.float32)
 def poly(pts,col):cv2.fillPoly(a,[np.array(pts,np.int32)],tuple(col)+(1,),cv2.LINE_AA)
 # Book is a separate invariant object, including thickness, pages and cover drawing.
 poly([[663,452],[707,455],[722,473],[666,479]],(241,224,241))
 poly([[666,470],[722,466],[722,477],[669,483]],(249,239,245))
 poly([[656,454],[708,450],[722,466],[667,473]],(251,244,250))
 for pts in [[[656,454],[708,450],[722,466],[667,473],[656,454]],[[666,473],[722,468]],[[668,477],[720,473]]]:cv2.polylines(a,[np.array(pts,np.int32)],False,(179,18,81,1),1,cv2.LINE_AA)
 cv2.ellipse(a,(683,460),(9,3),15,0,360,(66,55,77,1),1,cv2.LINE_AA)
 poly([[687,457],[676,433],[658,427],[655,413],[615,404],[648,413],[650,431],[667,441],[664,450]],(201,26,105))
 return a

@functools.lru_cache(None)
def door_actors():
 a=b.art('../rebuild_118_147/assets/door_pair.png').copy();a[:,a.shape[1]//2:]=0
 d=b.art('v2/assets/door_alt.png').copy();d=cv2.resize(d,(a.shape[1],a.shape[0]));d[:,d.shape[1]//2:]=0
 return b.warp(a,b.mat(.48,110,-5)),b.warp(d,b.mat(.48,110,-5))

def door(n):
 a,d=door_actors();real=b.plate('../rebuild_118_147/backgrounds/ballroom_clean_v1.png',zoom=1.02)
 crowd=b.cel('../rebuild_118_147/backgrounds/door_onlookers_v2.png',-35,105,height=510);real=b.over(b.over(real,crowd),a)
 plain=np.full_like(real,250)
 radius=8.5*np.clip((365-Y)/260,0,1);dots=(((X%24-12)**2+(Y%24-12)**2)<radius**2).astype(np.float32)
 columns=np.maximum(b.smooth((70-np.abs(X-212))/35),b.smooth((80-np.abs(X-651))/40))
 dotted=b.over(b.mix(plain,np.full_like(real,[52,61,70]),dots*columns*.9),a)
 alt_bg=b.plate('../rebuild_118_147/backgrounds/ballroom_clean_v1.png',zoom=1.02,light=.92)
 alt_bg=b.mix(alt_bg,np.full_like(alt_bg,[230,182,224]),dots*.24)
 alt=b.over(alt_bg,d)
 if n<4804 or n>=4832:im=real
 elif n<=4807:im=b.mix(plain,dotted,b.smooth((Y/H+(n-4804)*.32-.45)*6))
 elif n<=4816:im=dotted
 elif n<=4820:im=b.mix(dotted,alt,scene_mask(n,4816,4821,-1,1))
 elif n<=4829:im=alt
 else:im=b.mix(alt,real,scene_mask(n,4829,4832,-1,0))
 return b.over(im,book_layer()) # Book survives every world state and transition by user requirement.


def columns(n):
 a=np.zeros((H,W,4),np.float32);extent=float(np.interp(n,[5292,5310,5340,5379],[0,20,50,88]));alpha=1.
 def shape(points,color,flip):
  pts=np.array(points,np.float32)
  if flip:pts[:,0]=W-pts[:,0]
  cv2.fillPoly(a,[np.round(pts).astype(np.int32)],tuple(color)+(alpha,),cv2.LINE_AA)
 def rim(points,color,flip):
  pts=np.array(points,np.float32)
  if flip:pts[:,0]=W-pts[:,0]
  cv2.polylines(a,[np.round(pts).astype(np.int32)],True,tuple(color)+(alpha,),1,cv2.LINE_AA)
 for flip in [False,True]:
  # Projected square column shaft and three stepped plinths; opaque faces, not floating rectangles.
  e=extent;y=276;h=54-6*(n-5292)/87
  shape([[-160,y-30],[e-27,y-22],[e-27,y+25],[-160,y+45]],(222,221,228),flip)
  for j in range(3):
   x=e-22+j*7;hh=h-j*5;pts=[[x-14,y-hh/2+4],[x+5,y-hh/2],[x+5,y+hh/2],[x-14,y+hh/2+5]];shape(pts,(249,249,250),flip);rim(pts,(237,235,241),flip)
  x=e-8;pts=[[x-11,y-19],[x+2,y-21],[x+2,y+18],[x-11,y+21]];shape(pts,(230,228,235),flip);rim(pts,(251,251,252),flip)
  for j in range(4):rim([[x-9+j,y-17+j],[x+j,y-18+j],[x+j,y+17-j],[x-9+j,y+19-j]],(244,243,248),flip)
  # Far pair enters the top corners as camera pulls back, same architecture.
  far=e-43;shape([[-110,20],[far,45],[far,61],[-110,34]],(247,247,249),flip)
  rim([[-110,20],[far,45],[far,61],[-110,34]],(230,229,236),flip)
 return a


def prone(n):
 # v1 preserved same cel/ground and board zoom, remove its old floating rectangles by rendering below.
 if n<5281:return b.prone(n)
 z=1-.00215*(n-5281);cell=224*z;board=((np.floor((X-480)/cell+.5)+np.floor((Y-280)/cell+.5))%2)[:,:,None]
 ground=np.zeros((H,W,3),np.float32)+np.where(board,[249,250,250],[148,146,157]);ground+=b.noise()[:,:,None]*3;ground[::2,::2]*=.96
 c=b.mat(z,480*(1-z),280*(1-z));p=b.cel('assets/prone.png',416,120,height=321,c=c);al=p[:,:,3:4];col=p[:,:,:3]/np.maximum(al,.001);p[:,:,:3]=(col.mean(2,keepdims=True)*.84+np.array([18,14,13]))*al
 im=b.over(b.over(ground,p),columns(n))
 if n<5289:
  size=int(np.interp(n,[5281,5284,5288],[38,22,1]));im=cv2.resize(cv2.resize(im,(max(1,W//size),max(1,H//size)),interpolation=cv2.INTER_AREA),(W,H),interpolation=cv2.INTER_NEAREST)
 return im


def frame(n,source=None):
 if any(lo<=n<hi for lo,hi in SOURCE_REUSE):
  source.set(cv2.CAP_PROP_POS_FRAMES,n);ok,a=source.read();assert ok;return cv2.cvtColor(cv2.resize(a,(W,H)),cv2.COLOR_BGR2RGB)
 if n<4791:im=early(n)
 elif n<4846:im=door(n)
 elif n<4854:
  # Reuse approved v1 forest assets; new outgoing world includes invariant book.
  f=b.smooth((n-4846)/8);im=b.mix(door(4842),b.forest(4854),b.smooth((f-(b.noise()*.8+.2)+.3)*3))
 elif n<5069:im=b.forest(n)
 else:im=prone(n)
 if 4854<=n<5069:text='君と泣く　君と笑う　君と怒る'
 elif 5069<=n<5281:text='君と歌う　君と踊る　君と話す'
 elif 5281<=n<5380:text='何時まで続くだろうと同じ様に同じ様に呟く'
 else:text=None
 if text:im=b.mix(im,np.full_like(im,25 if n>=5281 else 245),b.lyrics(text))
 return b.bars(im)


def preview():
 ns=[4415,4440,4455,4474,4530,4551,4572,4580,4602,4624,4641,4677,4710,4755,4785,4797,4806,4818,4824,4831,5310,5340,5379,5400];source=cv2.VideoCapture(str(next(R.glob('*.mp4'))));sheet=Image.new('RGB',(1280,1212));dr=ImageDraw.Draw(sheet)
 for j,n in enumerate(ns):
  im=Image.fromarray(frame(n,source));im.save(Q/f'f{n}.png');im=im.resize((320,180));x=j%4*320;y=j//4*202;sheet.paste(im,(x,y));dr.text((x,y+183),f'{(n-4410)/30:.2f}s / f{n}',fill='white')
 sheet.save(Q/'contact_v2.jpg')
 for n in [4455,4551,4602,4677,4755]:
  a,d,m,out=early(n,True);sheet=Image.new('RGB',(960,540));
  for j,img in enumerate([a,d,np.repeat((m*255)[:,:,None],3,axis=2),out]):sheet.paste(Image.fromarray(b.bars(img)).resize((480,270)),(j%2*480,j//2*270))
  sheet.save(Q/f'states_and_shared_mask_{n}.jpg')

def render():
 out=O/'silent_v2.mp4';proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE);src=cv2.VideoCapture(str(next(R.glob('*.mp4'))))
 for n in range(4410,5460):
  proc.stdin.write(frame(n,src).tobytes())
  if n%150==0:print(n,flush=True)
 proc.stdin.close();assert proc.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(out),'-ss','147','-i',str(next(R.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t','35','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(O/'segment_147_182_v2.mp4')],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args();preview() if args.preview else render()
