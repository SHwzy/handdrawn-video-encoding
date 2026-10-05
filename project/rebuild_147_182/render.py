"""New artwork and code compositing. Source [4410,5460), approved inputs read-only."""
from pathlib import Path
import cv2,numpy as np,json,functools,subprocess,argparse
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;R=P.parent;O=P/'output';Q=P/'qa';Q.mkdir(exist_ok=True);W,H=960,540
Y,X=np.mgrid[:H,:W].astype(np.float32)
def smooth(t):
 t=np.clip(t,0,1);return t*t*(3-2*t)
def mat(s=1,x=0,y=0):return np.float32([[s,0,x],[0,s,y]])
def mul(a,b):return (np.vstack([a,[0,0,1]])@np.vstack([b,[0,0,1]]))[:2].astype(np.float32)
def warp(a,m):return cv2.warpAffine(a,m,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
def over(b,a):return b*(1-a[:,:,3:4])+a[:,:,:3]
def mix(a,b,m):return a*(1-m[:,:,None])+b*m[:,:,None]
def bars(a):
 a=a.copy();a[:53]=0;a[487:]=0;return np.clip(a,0,255).astype(np.uint8)
def save(path,a):Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(path)
@functools.lru_cache(None)
def art(name):
 path=(R/name[3:]) if name.startswith('../') else P/name
 a=np.array(Image.open(path).convert('RGBA')).astype(np.float32);a[:,:,3]=np.clip((a[:,:,3]-1)/252,0,1);a[:,:,:3]*=a[:,:,3:4];return a
@functools.lru_cache(None)
def bounds(name):
 a=art(name);yy,xx=np.where(a[:,:,3]>.3);return np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
def fitm(name,x,y,width=None,height=None):
 x0,y0,x1,y1=bounds(name);s=width/(x1-x0) if width is not None else height/(y1-y0);return mat(s,x-x0*s,y-y0*s)
def cel(name,x,y,width=None,height=None,c=None,mode=0):
 return warp(style(name,mode),mul(c if c is not None else mat(),fitm(name,x,y,width,height)))
@functools.lru_cache(None)
def style(name,mode):
 a=art(name).copy()
 if not mode:return a
 al=a[:,:,3:4];rgb=a[:,:,:3]/np.maximum(al,.001);yy,xx=np.mgrid[:a.shape[0],:a.shape[1]];ink=np.clip((205-rgb.mean(2))/160,0,1)
 if mode==1:
  color=np.array([173,213,207])*(1-ink[:,:,None])+np.array([91,40,99])*ink[:,:,None]
  bands=((xx+yy*.2)//140)%3==0;color=np.where(bands[:,:,None],color*np.array([1.2,.52,.8]),color)
 elif mode==2:
  color=np.where((xx<a.shape[1]*.55)[:,:,None],[194,42,99],[38,36,60])+np.zeros_like(rgb)
  color=np.where((((xx%24-12)**2+(yy%24-12)**2)<20)[:,:,None],color*.65,color)
 else:
  color=np.zeros_like(rgb)+[18,19,24];gold=(rgb[:,:,0]>110)&(rgb[:,:,1]>85)&(rgb[:,:,2]<rgb[:,:,1]*.88);pink=(rgb[:,:,0]>rgb[:,:,1]*1.4)&(rgb[:,:,0]>90)
  eyes=(((xx-684)/75)**2+((yy-321)/65)**2<1)|(((xx-903)/39)**2+((yy-234)/52)**2<1) if name.endswith('panic.png') else np.zeros_like(gold)
  color=np.where((gold|pink|eyes)[:,:,None],rgb,color)
 a[:,:,:3]=color*al;return a
@functools.lru_cache(None)
def rgb(name):return np.array(Image.open(P/name).convert('RGB')).astype(np.float32)
def plate(name,dx=0,dy=0,zoom=1,angle=0,light=0):
 a=rgb(name);s=max(W/a.shape[1],434/a.shape[0])*zoom;m=cv2.getRotationMatrix2D((a.shape[1]/2,a.shape[0]/2),angle,s);m[:,2]+=np.array([480+dx,270+dy])-np.array([a.shape[1]/2,a.shape[0]/2]);im=warp(a,m)
 return im*(1-light)+245*light
RAW=json.loads((P/'motion_used.json').read_text())
def camera(name,n):
 v=RAW[name];d=n-v['reference'];values={k:np.polyval(c,d) for k,c in v['coefficients'].items()};x,y,x2,y2=v['roi'];anchor=np.array([(x+x2)/2,(y+y2)/2]);m=cv2.getRotationMatrix2D(tuple(anchor),values['angle'],values['scale']);m[:,2]+=np.array([values['x'],values['y']])-anchor;return m
@functools.lru_cache(None)
def pattern(kind):
 a=np.full((H,W,3),[250,247,248],np.float32)
 if kind==0:
  for y in range(-30,H+90,85):
   for x in range(-30,W+90,85):
    for j in range(6):
     ag=j*np.pi/3;cx=int(x+12*np.cos(ag));cy=int(y+12*np.sin(ag));cv2.ellipse(a,(cx,cy),(4,8),j*60,0,360,(223,140,172),1,cv2.LINE_AA)
    cv2.circle(a,(x,y),3,(230,115,153),1,cv2.LINE_AA)
 else:
  a[:]=[244,241,195]
  for y in range(-100,H+180,130):
   for x in range(-100,W+180,130):
    cv2.polylines(a,[np.array([[x,y+65],[x+65,y],[x+130,y+65],[x+65,y+130]])],True,(214,178,226),2,cv2.LINE_AA)
    for off in [24,48,72]:cv2.line(a,(x+off,y+45),(x+off,y+68),(208,155,219),2,cv2.LINE_AA)
 a[::3,::3]*=.93;return a
@functools.lru_cache(None)
def noise():return cv2.resize(np.random.default_rng(182).random((12,22)).astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)
def ribbon(n,start,end,reverse=False):
 t=np.clip((n-start)/(end-start),0,1)
 if n<=start:return np.zeros((H,W),np.float32)
 if n>=end:return np.ones((H,W),np.float32)
 u=1-X/W if reverse else X/W
 f=u+.18*np.sin(Y*.013+u*4)+.12*np.sin(Y*.023-u*7)+noise()*.045
 return smooth((t*1.65-.3-f)/.017+.5)
def loops(n,origin):
 phase=(n-origin)*.026;f=X/W+.22*np.sin(Y*.015+phase)+.10*np.sin(Y*.033-phase*.7)
 d=np.abs(np.sin((f-phase*.27)*np.pi*1.8));return smooth((.29-d)*70)
def floor(n):
 a=np.full((H,W,3),[40,48,66],np.float32);a+=(Y[:,:,None]/H)*80
 for j in range(-7,10):cv2.line(a,(480+j*50,80),(480+j*250,540),(153,156,167),1,cv2.LINE_AA)
 for y in [130,170,230,310,410,535]:cv2.line(a,(0,y),(960,y),(148,151,160),1,cv2.LINE_AA)
 return a

def early(n):
 if n<4469:
  name='stand';c=camera(name,n);bg=plate('backgrounds/hall_v1.png',dy=(n-4434)*.6,zoom=1.18,light=.14);args=(430,-20,None,950);normal=cel('assets/stand.png',*args,c=c);alt=cel('assets/stand.png',*args,c=c,mode=3);mp=ribbon(n,4436,4475);bmask=loops(n,4420)*smooth((n-4420)/20);im=over(mix(bg,np.full_like(bg,246),bmask),normal*(1-mp[:,:,None])+alt*mp[:,:,None]);sigma=np.interp(n,[4410,4425,4468],[2.5,0,0])
 elif n<4525:
  name='whisper';c=camera(name,n);bg=plate('backgrounds/hall_v1.png',zoom=1.3,light=.08);args=(30,-25,680,None);normal=cel('assets/whisper.png',*args,c=c);alt=cel('assets/whisper.png',*args,c=c,mode=1);m=ribbon(n,4474,4512);bmask=loops(n,4469)*.97;im=over(mix(bg,np.full_like(bg,[239,226,241]),bmask),alt*(1-m[:,:,None])+normal*m[:,:,None]);sigma=0
 elif n<4574:
  name='offer';c=camera(name,n);bg=plate('backgrounds/hall_v1.png',zoom=1.9,dy=-80,light=.18);args=(-35,-75,1040,None);normal=cel('assets/offer.png',*args,c=c);alt=cel('assets/offer.png',*args,c=c,mode=2);m=ribbon(n,4550,4580,True);im=over(mix(bg,np.full_like(bg,245),loops(n,4538)*smooth((n-4538)/20)),normal*(1-m[:,:,None])+alt*m[:,:,None]);sigma=0
 elif n<4627:
  name='book';c=camera(name,n);bg=plate('backgrounds/hall_v1.png',zoom=1.7,angle=30,light=.04);args=(-40,-28,1050,None);normal=cel('assets/book.png',*args,c=c);m=ribbon(n,4590,4636);b=mix(pattern(1),bg,m);im=over(mix(b,np.full_like(bg,248),loops(n,4570)*.80),normal);sigma=0
 else:
  name='panic';c=camera(name,n);bg=floor(n);args=(70,65,880,None);normal=cel('assets/panic.png',*args,c=c);alt=cel('assets/panic.png',*args,c=c,mode=3)
  mb=ribbon(n,4663,4694);mb*=1-ribbon(n,4770,4791,True);mp=ribbon(n,4645,4690);mp*=1-ribbon(n,4737,4787,True);im=over(mix(bg,pattern(0),mb),normal*(1-mp[:,:,None])+alt*mp[:,:,None]);sigma=np.interp(n,[4627,4635,4645],[2,1,0])
 if sigma>.05:im=cv2.GaussianBlur(im,(0,0),sigma)
 return im

def door_layers(n):
 b=plate('../rebuild_118_147/backgrounds/ballroom_clean_v1.png',zoom=1.02)
 crowd=cel('../rebuild_118_147/backgrounds/door_onlookers_v2.png',-35,105,height=510)
 b=over(b,crowd)
 # Crop shared approved independent cel into Anon only. Alpha zero outside left half.
 a=art('../rebuild_118_147/assets/door_pair.png').copy();a[:,a.shape[1]//2:]=0
 character=warp(a,mat(.48,110,-5))
 # Code-native fallen book and angular pink trail.
 cv2.fillPoly(b,[np.array([[654,464],[698,462],[718,476],[669,481]])],(252,226,244));cv2.polylines(b,[np.array([[654,464],[698,462],[718,476],[669,481],[654,464]])],False,(193,28,94),1,cv2.LINE_AA)
 cv2.fillPoly(b,[np.array([[684,464],[673,439],[655,433],[646,416],[605,413],[647,427],[650,443]])],(201,14,91))
 return b,character

def door(n):
 b,p=door_layers(n);real=over(b,p)
 if n<4804 or n>=4832:return real
 alt=plate('../rebuild_118_147/backgrounds/ballroom_clean_v1.png',zoom=1.02,light=.86);alt=mix(alt,pattern(1),.25*np.ones((H,W),np.float32));p2=p.copy();p2[:,:,:3]=p[:,:,3:4]*np.array([198,54,121]);styl=over(alt,p2)
 # One halftone screen sweep followed by world switch and a second light wipe.
 if n<=4816:
  t=(n-4804)/12;radius=8.5*np.clip((365-Y)/260,0,1);dots=(((X%24-12)**2+(Y%24-12)**2)<radius**2).astype(np.float32)
  columns=np.maximum(smooth((70-np.abs(X-212))/35),smooth((80-np.abs(X-651))/40));screen=np.full_like(real,250);screen=mix(screen,np.full_like(real,[52,61,70]),dots*columns*.9)
  isolated=over(screen,p)
  cv2.fillPoly(isolated,[np.array([[654,464],[698,462],[718,476],[669,481]])],(252,226,244));cv2.polylines(isolated,[np.array([[654,464],[698,462],[718,476],[669,481],[654,464]])],False,(193,28,94),1,cv2.LINE_AA)
  cv2.fillPoly(isolated,[np.array([[684,464],[673,439],[655,433],[646,416],[605,413],[647,427],[650,443]])],(201,14,91))
  if n<4808:return mix(np.full_like(real,250),isolated,smooth((Y/H+(n-4804)*.32-.45)*6))
  return isolated
 if n<=4820:
  t=(n-4817)/3;return mix(np.full_like(real,250),styl,smooth((t*1.5-X/W)*4))
 if n<=4829:return styl
 return mix(styl,np.full_like(real,250),ribbon(n,4828,4832,True))

def forest(n):
 d=n-4950;c=camera('forest_char',n)
 b=plate('backgrounds/forest_back_v1.png',dx=-.31*d-20,dy=np.sin(d*.025)*.7)
 person=cel('assets/forest.png',498,150,width=444,c=c)
 b=over(b,person)
 front=art('backgrounds/forest_front_v1.png');s=434/front.shape[0];m=mat(s,-30-.80*d,53+np.sin(d*.028)*1.0);b=over(b,warp(front,m))
 if n<4854:
  f=smooth((n-4846)/8);old=door(4842);veil=noise()*.8+.2;return mix(old,b,smooth((f-veil+.3)*3))
 return b

def prone(n):
 # Reuse identical pose, scale and alpha before/after scene replacement.
 dy=.758*(n-5220);dx=np.sin(n*.065)*.8
 if n<5281:
  b=plate('backgrounds/ground_code_v1.png',dy=dy,zoom=1.5)
 else:
  z=1-.00215*(n-5281);cell=224*z;board=((np.floor((X-480)/cell+.5)+np.floor((Y-280)/cell+.5))%2)[:,:,None];b=np.zeros((H,W,3),np.float32)+np.where(board,[249,250,250],[148,146,157]);b+=noise()[:,:,None]*3;b[::2,::2]*=.96
 if n<5281:p=cel('assets/prone.png',416+dx,74+dy,height=321)
 else:
  z=1-.00215*(n-5281);c=mat(z,480*(1-z),280*(1-z));p=cel('assets/prone.png',416,120,height=321,c=c)
 if n>=5281:
  al=p[:,:,3:4];col=p[:,:,:3]/np.maximum(al,.001);gray=col.mean(2,keepdims=True);p[:,:,:3]=(gray*.84+np.array([18,14,13]))*al
 im=over(b,p)
 if 5278<=n<5289:
  cell=int(np.interp(n,[5278,5281,5284,5288],[3,38,22,1]));im=cv2.resize(cv2.resize(im,(max(1,W//cell),max(1,H//cell)),interpolation=cv2.INTER_AREA),(W,H),interpolation=cv2.INTER_NEAREST)
 if n>=5300:
  t=smooth((n-5300)/65)
  for x in [-20,930]:
   for j in range(4):cv2.rectangle(im,(int(x+j*6),int(245-j*3)),(int(x+22+j*6),int(270-j*3)),(245,245,247),1,cv2.LINE_AA)
 return im
@functools.lru_cache(None)
def lyrics(text):
 im=Image.new('L',(W,H));dr=ImageDraw.Draw(im);font=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc'),12);y=179
 for ch in text:dr.text((908,y),ch,font=font,fill=215);y+=13
 return np.array(im,dtype=np.float32)/255

def frame(n,source=None):
 if n>=5380:
  source.set(cv2.CAP_PROP_POS_FRAMES,n);ok,a=source.read();assert ok;return cv2.cvtColor(cv2.resize(a,(W,H)),cv2.COLOR_BGR2RGB)
 if n<4791:a=early(n)
 elif n<4846:a=door(n)
 elif n<5069:a=forest(n)
 else:a=prone(n)
 if 4854<=n<5069:text='君と泣く　君と笑う　君と怒る'
 elif 5069<=n<5281:text='君と歌う　君と踊る　君と話す'
 elif 5281<=n<5380:text='何時まで続くだろうと同じ様に同じ様に呟く'
 else:text=None
 if text:
  m=lyrics(text);a=mix(a,np.full_like(a,25 if n>=5281 else 245),m)
 return bars(a)

def previews():
 ns=[4410,4440,4474,4515,4530,4560,4580,4620,4640,4695,4760,4795,4808,4824,4848,4860,4950,5060,5070,5220,5281,5287,5379,5400];source=cv2.VideoCapture(str(next(R.glob('*.mp4'))));out=Image.new('RGB',(1280,202*6));dr=ImageDraw.Draw(out)
 for j,n in enumerate(ns):
  a=frame(n,source);save(Q/f'f{n}.jpg',a);im=Image.fromarray(a).resize((320,180));x=j%4*320;y=j//4*202;out.paste(im,(x,y));dr.text((x+3,y+183),f'{n/30:.2f}s f{n}',fill='white')
 out.save(Q/'contact_v1.jpg')

def render():
 silent=O/'silent_v1.mp4';p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE);source=cv2.VideoCapture(str(next(R.glob('*.mp4'))))
 for n in range(4410,5460):
  p.stdin.write(frame(n,source).tobytes())
  if n%150==0:print(n,flush=True)
 p.stdin.close();assert p.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(silent),'-ss','147','-i',str(next(R.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t','35','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(O/'segment_147_182_v1.mp4')],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args();previews() if args.preview else render()
