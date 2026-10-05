"""Source-observed loop ribbons, separated state timing, corrected acting."""
import render_v2 as v
import cv2,numpy as np,functools,subprocess,argparse,json
from PIL import Image,ImageDraw
b=v.b;P=b.P;Q=P/'v3/qa';X,Y=b.X,b.Y;W,H=b.W,b.H
old_pair=v.pair

def ink_state(p,color=(225,66,124),faceless=False):
 a=p[:,:,3:4];rgb=p[:,:,:3]/np.maximum(a,.001)
 gray=cv2.cvtColor(np.uint8(np.clip(rgb,0,255)),cv2.COLOR_RGB2GRAY)
 lines=cv2.Canny(cv2.GaussianBlur(gray,(0,0),1.6),35,90).astype(np.float32)/255
 lines=cv2.dilate(lines,np.ones((2,2),np.uint8))
 if faceless:
  # Suppress facial internals, retaining original outer silhouette and hair lines.
  cv2.fillPoly(lines,[np.array([[356,191],[402,190],[425,230],[402,264],[365,251]],np.int32)],0)
 edge=cv2.dilate(a[:,:,0],np.ones((3,3),np.uint8))-cv2.erode(a[:,:,0],np.ones((3,3),np.uint8))
 lines=np.maximum(lines,edge)[:,:,None]
 col=np.array([170,168,176])*(1-lines)+np.array(color)*lines
 return np.concatenate([col*a,a],axis=2).astype(np.float32)

@functools.lru_cache(None)
def pair(name):
 if name=='offer':
  a=b.art('v3/assets/offer_normal.png');return a,ink_state(a)
 if name=='book':
  a=b.art('v3/assets/book_normal.png');d=b.art('v3/assets/book_alt.png')
  return a,cv2.resize(d,(a.shape[1],a.shape[0]))
 return old_pair(name)
v.pair=pair

def ring(cx,cy,rx,ry,width,angle=0):
 a=np.deg2rad(angle);sx=X+24*np.sin(Y*.009)+13*np.sin((X+Y)*.005);sy=Y+21*np.sin(X*.007)
 xx=(sx-cx)*np.cos(a)+(sy-cy)*np.sin(a);yy=-(sx-cx)*np.sin(a)+(sy-cy)*np.cos(a)
 # Elliptical distances produce continuous enclosed apertures, not a generic wipe.
 radius=np.sqrt((xx/rx)**2+(yy/ry)**2);theta=np.arctan2(yy/ry,xx/rx)
 distance=(radius-1)*min(rx,ry)+13*np.sin(theta*3+.5)+6*np.sin(theta*5-1)
 taper=.82+.30*np.sin(theta*2+.8)+.15*np.cos(theta*3)
 return b.smooth((width*taper-np.abs(distance)+1.0)/2.0)

def loops(n,start,end,kind=0):
 t=float(np.clip((n-start)/(end-start),0,1))
 if t==0:return np.zeros((H,W),np.float32)
 layouts=[[(105,257,155,235,-19),(795,230,220,300,-30),(470,-150,560,320,14)],[(38,233,154,260,-18),(721,307,237,198,26),(1090,125,380,270,-26)],[(134,329,262,180,-31),(767,68,295,210,-27),(598,710,510,350,-22)],[(30,200,235,215,-15),(836,298,315,194,-32),(400,-203,470,340,15)]]
 out=np.zeros((H,W),np.float32)
 for j,(cx,cy,rx,ry,ang) in enumerate(layouts[kind]):
  u=max(0,(t-j*.085)/(1-j*.085));width=(u**1.65)* (112 if kind!=3 else 162)
  if width<=0:continue
  drift=(t-.5)*36
  out=np.maximum(out,ring(cx+drift*(1 if j%2 else -1),cy+drift*.3,rx,ry,width,ang))
 return out

def early(n,diagnostic=False):
 if n<4469:
  name='stand';m=loops(n,4425,4468,0)
 elif n<4574:
  name='offer';m=loops(n,4532,4573,1)
 elif n<4627:
  name='book';m=1-loops(n,4558,4626,2)
 else:name='panic';m=loops(n,4628,4668,3)
 a,d=v.actors(name,n);real=v.bg(name,n)
 if name in ['stand','offer']:alt=np.full_like(real,[249,248,248])
 elif name=='book':alt=b.pattern(1)
 else:alt=np.full_like(real,[25,25,31])
 A=b.over(real,a);B=b.over(alt,d)
 if name!='panic':
  out=b.mix(A,B,m)
  if name=='stand' and n>=4451:
   # Late warm ribbon state remains subordinate to the same scene geometry.
   warm=d.copy();al=warm[:,:,3:4];col=warm[:,:,:3]/np.maximum(al,.001)
   warm[:,:,:3]=(col*.48+np.array([241,119,66])*.52)*al
   wm=loops(n,4451,4478,1)*m
   out=b.mix(out,b.over(alt,warm),wm)
 else:
  # Source: dark-image strips first; floral world arrives ~4680; light face returns ~4740.
  dark_m=loops(n,4628,4665,3)
  dark_m=np.maximum(dark_m,b.smooth((n-4658)/21))
  floral=loops(n,4677,4713,1)
  floral=np.maximum(floral,b.smooth((n-4700)/28)*.87)
  floral*=1-b.smooth((n-4776)/14)
  back=b.mix(real,b.pattern(0),floral)
  restore=loops(n,4736,4784,2)
  restore=np.maximum(restore,b.smooth((n-4774)/15))
  actor_m=dark_m*(1-restore)
  out=b.mix(b.over(back,a),b.over(back,d),actor_m)
  A=b.over(back,a);B=b.over(back,d);m=actor_m
 sigma=float(np.interp(n,[4410,4425],[2.5,0])) if name=='stand' else (float(np.interp(n,[4627,4642],[2,0])) if name=='panic' else 0)
 if sigma>.05:out=cv2.GaussianBlur(out,(0,0),sigma)
 return (A,B,m,out) if diagnostic else out
v.early=early

@functools.lru_cache(None)
def door_actors():
 a=b.art('../rebuild_118_147/assets/door_pair.png').copy();a[:,a.shape[1]//2:]=0
 d=ink_state(a,faceless=True)
 return b.warp(a,b.mat(.48,110,-5)),b.warp(d,b.mat(.48,110,-5))

def door(n):
 a,d=door_actors();real=b.plate('../rebuild_118_147/backgrounds/ballroom_clean_v1.png',zoom=1.02)
 crowd=b.cel('../rebuild_118_147/backgrounds/door_onlookers_v2.png',-35,105,height=510)
 real=b.over(b.over(real,crowd),a)
 plain=np.full_like(real,250);radius=8.5*np.clip((365-Y)/260,0,1)
 dots=(((X%24-12)**2+(Y%24-12)**2)<radius**2).astype(np.float32)
 columns=np.maximum(b.smooth((70-np.abs(X-212))/35),b.smooth((80-np.abs(X-651))/40))
 dotted=b.over(b.mix(plain,np.full_like(real,[52,61,70]),dots*columns*.9),d)
 # Same robe/head geometry, faceless drawn effect; never substitute another costume.
 alternate=b.over(b.plate('../rebuild_118_147/backgrounds/ballroom_clean_v1.png',zoom=1.02,light=.91),d)
 if n<4804 or n>=4832:out=real
 elif n<4809:out=b.mix(plain,dotted,b.smooth((Y/H+(n-4804)*.32-.45)*6))
 elif n<4817:out=dotted
 elif n<4821:out=b.mix(dotted,alternate,b.smooth((n-4817)/4)*np.ones((H,W)))
 elif n<4830:out=alternate
 else:out=b.mix(alternate,plain,b.smooth((n-4830)/2)*np.ones((H,W)))
 return b.over(out,v.book_layer())
v.door=door

def preview():
 ns=[4410,4435,4455,4530,4550,4570,4580,4590,4605,4624,4635,4655,4685,4710,4748,4786,4809,4824]
 src=cv2.VideoCapture(str(next(b.R.glob('*.mp4'))));sheet=Image.new('RGB',(1280,202*((len(ns)+3)//4)));draw=ImageDraw.Draw(sheet)
 for i,n in enumerate(ns):
  im=Image.fromarray(v.frame(n,src));im.save(Q/f'f{n}.png');x=i%4*320;y=i//4*202;sheet.paste(im.resize((320,180)),(x,y));draw.text((x,y+182),str(n),fill='white')
 sheet.save(Q/'contact.jpg')
 for n in [4435,4455,4550,4570,4590,4605,4635,4685,4748]:
  A,B,m,out=early(n,True);s=Image.new('RGB',(960,540))
  for i,im in enumerate([A,B,m[:,:,None]*np.ones((1,1,3))*255,out]):s.paste(Image.fromarray(b.bars(im)).resize((480,270)),(i%2*480,i//2*270))
  s.save(Q/f'mask_{n}.jpg')

def render():
 out=b.O/'silent_v3.mp4';proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE);src=cv2.VideoCapture(str(next(b.R.glob('*.mp4'))))
 for n in range(4410,5460):
  proc.stdin.write(v.frame(n,src).tobytes())
  if n%150==0:print(n,flush=True)
 proc.stdin.close();assert proc.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(out),'-ss','147','-i',str(next(b.R.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t','35','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(b.O/'segment_147_182_v3.mp4')],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args();preview() if args.preview else render()
