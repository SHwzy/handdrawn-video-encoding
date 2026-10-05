"""Independent 60–90 second reconstruction. No scene masks or old production imports.

Only generated sprites' intrinsic alpha is used for ordinary layer composition.
Background state changes are full-frame cuts, never spatial reveals.
"""
from pathlib import Path
import argparse,functools,json,math,subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
SRC=next(ROOT.glob('*.mp4')); W,H,AH=960,540,434; FPS=30
OUT=HERE/'output'; QA=HERE/'qa'; OUT.mkdir(exist_ok=True); QA.mkdir(exist_ok=True)
JP=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc'),12)
SHOTS=[
 ('partner',1800,1950),('duo',1950,1973),('feet',1973,2034),
 ('standing',2034,2102),('hand',2102,2130),('reach',2130,2172),
 ('hands',2172,2192),('contact_montage',2192,2200),('turn',2200,2227),
 ('dance',2227,2335),('anon_close',2335,2389),('sakiko_close',2389,2443),
 ('embrace',2443,2547),('back',2547,2601),('eye',2601,2633),('outro',2633,2700)]
REUSE={'feet','hand','hands','outro'}
AUDIT=[]

def ease(t,p=3): return 1-(1-np.clip(t,0,1))**p
def lerp(a,b,t): return a+(b-a)*t

@functools.lru_cache(None)
def asset(name):
 p=HERE/'assets'/f'{name}.png'; im=np.array(Image.open(p).convert('RGBA')).astype(np.float32)/255
 if name not in ('sea_v1','sea_wide_v1','floral_v1'):
  assert im[:,:,3].min()==0 and im[:,:,3].max()>.95,(name,'missing alpha')
 im[:,:,:3]*=im[:,:,3:4]
 if name=='opening_continuity':im=cv2.resize(im,(900,round(im.shape[0]*900/im.shape[1])),interpolation=cv2.INTER_AREA)
 return im

def layer(bg,name,anchor,dest,scale,angle=0,tint=None):
 p=asset(name)
 if tint is not None:
  p=p.copy();p[:,:,:3]=p[:,:,:3]*.78+np.array(tint,dtype=np.float32)[None,None,:]*p[:,:,3:4]*.22
 ax,ay=anchor
 m=cv2.getRotationMatrix2D((ax,ay),float(angle),float(scale))
 m[:,2]+=np.array(dest)-np.array(anchor)
 fg=cv2.warpAffine(p,m,(W,AH),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
 return bg*(1-fg[:,:,3:4])+fg[:,:,:3]*255

def background(name,q,zoom=1.05,dx=0,dy=0,angle=0,blur=0):
 p=asset(name)[:,:,:3]*255; h,w=p.shape[:2]
 s=max(W/w,AH/h)*zoom
 m=cv2.getRotationMatrix2D((w/2,h/2),angle,s);m[:,2]+=np.array([W/2+dx,AH/2+dy])-np.array([w/2,h/2])
 inv=cv2.invertAffineTransform(m);corners=np.array([[0,0,1],[W-1,0,1],[0,AH-1,1],[W-1,AH-1,1]])@inv.T
 assert corners[:,0].min()>=-1 and corners[:,1].min()>=-1 and corners[:,0].max()<w+1 and corners[:,1].max()<h+1,(name,corners)
 bg=cv2.warpAffine(p,m,(W,AH),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 if blur>.05:bg=cv2.GaussianBlur(bg,(0,0),blur)
 return bg

@functools.lru_cache(None)
def pattern_plate(kind):
 pw,ph=1600,1100
 if kind=='purple':
  p=np.full((ph,pw,3),(75,69,112),np.uint8)
  for y in range(-90,ph+90,54):
   for x in range(-90,pw+90,54):
    c=(99,91,143) if ((x+y)//54)%2 else (119,108,157)
    pts=np.array([[x,y],[x+38,y+20],[x+18,y+54],[x-19,y+33]],np.int32)
    cv2.fillPoly(p,[pts],c);cv2.line(p,(x,y+10),(x+21,y+20),(80,73,122),8)
 else:
  dark=kind=='slate';p=np.full((ph,pw,3),(100,102,118) if dark else (239,236,232),np.uint8)
  col=(197,194,206) if dark else ((190,111,133) if kind=='rose' else (205,180,110))
  for y in range(-120,ph+120,132):
   for x in range(-120,pw+120,132):
    cv2.circle(p,(x,y),65,col,1,cv2.LINE_AA)
    cv2.rectangle(p,(x-46,y-46),(x+46,y+46),col,1,cv2.LINE_AA)
    pts=np.array([[x,y-66],[x+66,y],[x,y+66],[x-66,y]],np.int32)
    cv2.polylines(p,[pts],True,col,1,cv2.LINE_AA)
    if not dark:
     cv2.circle(p,(x,y),32,col,1,cv2.LINE_AA)
     cv2.line(p,(x-66,y),(x+66,y),col,1,cv2.LINE_AA)
  p[::3,::3]=(p[::3,::3].astype(float)*.93).astype(np.uint8)
 return p.astype(np.float32)

def pattern(kind,n,angle=0,zoom=1):
 p=pattern_plate(kind);h,w=p.shape[:2]
 m=cv2.getRotationMatrix2D((w/2,h/2),angle,zoom)
 m[:,2]+=np.array([W/2+18*math.sin(n*.004),AH/2+8*math.cos(n*.006)])-np.array([w/2,h/2])
 return cv2.warpAffine(p,m,(W,AH),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def continuity_grid(k):
 # The independent line artwork continues the approved 59.967s endpoint.
 p=np.full((AH,W,3),(242,237,231),np.uint8)
 bend=7*math.exp(-k/5);z=np.linspace(-80,AH+80,180)
 for x in range(-AH,W+AH,64):
  for side in [-1,1]:
   px=x+side*z+bend*np.sin(z/75+x/120)
   py=z+bend*.45*np.sin(px/130+side)
   cv2.polylines(p,[np.rint(np.column_stack([px,py])).astype(np.int32)],False,(192,79,103),1,cv2.LINE_AA)
 p[::3,::3]=np.uint8(p[::3,::3].astype(float)*.94)
 return p.astype(np.float32)

def petals(bg,n,seed=11,count=18,colour=(242,234,237),speed=1,flowers=False):
 # Ordinary drawn foreground objects. No reveal mattes and no image compositing masks.
 p=bg.copy();rng=np.random.default_rng(seed)
 for i in range(count):
  x0,y0=rng.uniform(-80,W+80),rng.uniform(-70,AH+70);size=rng.uniform(3,12)
  x=(x0+n*speed*(.3+i%4*.1)+100)%(W+200)-100
  y=(y0+n*speed*.32+80)%(AH+160)-80
  a=rng.uniform(0,6.28)+n*.013;cx,cy=int(x),int(y)
  if flowers:
   for k in range(5):
    theta=a+k*math.tau/5
    cv2.circle(p,(int(x+size*.55*math.cos(theta)),int(y+size*.55*math.sin(theta))),max(1,int(size*.43)),colour,-1,cv2.LINE_AA)
  else:
   pts=np.array([[x+size*math.cos(a),y+size*math.sin(a)], [x+size*.35*math.cos(a+1.3),y+size*.35*math.sin(a+1.3)], [x-size*math.cos(a),y-size*math.sin(a)], [x+size*.3*math.cos(a-1.2),y+size*.3*math.sin(a-1.2)]],np.int32)
   cv2.fillPoly(p,[pts],colour,cv2.LINE_AA)
 return p

def rays(bg,centre,radius,n,colour):
 p=bg.copy();cx,cy=centre
 for k in range(24):
  a=k*math.tau/24+n*.002
  r=radius+(k%3)*7
  cv2.line(p,(round(cx+math.cos(a)*r),round(cy+math.sin(a)*r)),(round(cx+math.cos(a)*(r+30)),round(cy+math.sin(a)*(r+30))),colour,1,cv2.LINE_AA)
 return p

LYRICS={
 'partner':['感じてたものが遠く放たれていた','同じ様で違うなんだか違う'],
 'duo':['感じてたものが遠く放たれていた','同じ様で違うなんだか違う'],
 'standing':['何時まで行こうか','何処まで行けるのか'],
 'reach':['定かじゃないなら何を想うの'],
 'contact_montage':['定かじゃないなら何を想うの'],
 'turn':['定かじゃないなら何を想うの'],
 'dance':['僕らが離れるなら','僕らが迷うなら'],
 'anon_close':['その度に何回も繋がれる様に'],
 'sakiko_close':['その度に何回も繋がれる様に'],
 'embrace':['ここに居てくれるなら','離さずいられたら'],
 'back':['まだ誰も知らない感覚で救われていく'],
 'eye':['まだ誰も知らない感覚で救われていく']}

def caption(bg,shot,white=False):
 im=Image.fromarray(np.clip(bg,0,255).astype(np.uint8));d=ImageDraw.Draw(im)
 for j,line in enumerate(LYRICS.get(shot,[])):
  x=908-j*24;y=max(22,(AH-len(line)*12)//2)
  for k,ch in enumerate(line): d.text((x,y+k*12),ch,font=JP,fill=(245,241,239) if white else (35,33,41),stroke_width=0)
 return np.asarray(im)

def frame(n,source):
 shot,start,end=next(s for s in SHOTS if s[1]<=n<s[2]);k=n-start;q=k/max(1,end-start-1);e=ease(q)
 if shot in REUSE:
  out=cv2.resize(source,(W,H),interpolation=cv2.INTER_AREA)
  return out,shot
 white=False
 if shot=='partner':
  a=(n-1790)/180; z=lerp(.90,.50,ease(min(a,1),2))
  kind='rose' if n<1900 else ('purple' if n<1930 else 'rose')
  bg=pattern(kind,n,angle=lerp(8.68,16,e),zoom=1.05)
  if n>=1930:bg=rays(bg,(530,150),130,k,(35,37,45))
  if n<1900:
   bg=continuity_grid(k)
   bg=layer(bg,'opening_continuity',(450,262),(480,217),.85+.15*math.exp(-k/3),angle=8.68)
  else:
   bg=layer(bg,'opening_pair_v1',(490,200),(475,100+22*e),z,angle=0,tint=(.7,.3,.6) if 1900<=n<1930 else None)
  if n>=1930:bg=petals(bg,k,count=14,flowers=True,colour=(208,53,104))
  white=kind=='purple'
 elif shot=='duo':
  bg=pattern('gold',n,angle=12+q*2)
  bg=layer(bg,'opening_duo_v1',(775,385),(465,180),.65-.03*e,angle=-6)
  bg=petals(bg,k,count=12,flowers=True,colour=(225,90,139))
 elif shot=='standing':
  lift=ease(q,2)
  bg=background('sea_v1',q,zoom=1.30-.16*lift,dx=8-16*q,dy=-49*(1-lift))
  bg=layer(bg,'sakiko_stand_v1',(580,1500),(713-17*q,397+145*lift),.355-.012*q)
  bg=layer(bg,'anon_stand_v1',(470,1510),(280+9*q,395+145*lift),.365-.012*q)
  white=True
 elif shot=='reach':
  bg=background('sea_wide_v1',q,zoom=1.15,dy=20-40*q,dx=4-8*q)
  bg=layer(bg,'sakiko_stand_v1',(580,1500),(599-8*q,508-58*q),.165+.006*q)
  bg=layer(bg,'anon_reach_v1',(450,1510),(364-8*q,508-58*q),.18+.006*q)
  white=True
 elif shot=='contact_montage':
  bg=background('sea_v1',q,zoom=1.15,blur=1)
  anchors=[(865,420),(755,340),(600,300),(950,460)]
  bg=layer(bg,'dance_pair_v1',anchors[k//2],(470,235),2.25,angle=0)
  white=True
 elif shot=='turn':
  bg=background('sea_v1',q,zoom=1.14,dx=14-28*e,blur=2.0*(1-e))
  bg=layer(bg,'sakiko_turn_v2',(700,400),(735-50*e,170+5*e),lerp(1.36,.72,e),angle=0)
  if k<9:bg=cv2.GaussianBlur(bg,(0,0),max(.1,2.5*(1-k/9)))
  white=True
 elif shot=='dance':
  state=0 if k<28 else (1 if k<57 else (2 if k<82 else 3))
  bg=pattern('rose',n,angle=-8+12*q) if state==0 else (pattern('gold',n,angle=14+8*q) if state==1 else background('sea_v1',q,zoom=1.12,dx=-20+40*q,angle=1.2,blur=.25))
  if state==3:bg=background('floral_v1',q,zoom=1.06,dx=-10+20*q)
  bg=layer(bg,'dance_pair_v1',(850,420),(508-27*q,205+13*q),lerp(.71,.59,ease(q,2)),angle=lerp(-8,3,q),tint=(.95,.55,.7) if state==1 else None)
  bg=petals(bg,k,seed=27,count=26,colour=(249,243,241),speed=2)
 elif shot in ('anon_close','sakiko_close'):
  anon=shot=='anon_close';kind='rose' if anon else 'gold'
  bg=pattern(kind,n,angle=8*q) if k<30 else background('sea_v1',q,zoom=1.13,dx=15-20*q,blur=.8)
  if anon:
   entry=ease(k/15)
   bg=layer(bg,'anon_front_v1',(630,450),(lerp(920,592,entry),145),lerp(.77,.66,ease(k/22)),angle=0)
  else:
   entry=ease(k/15)
   bg=layer(bg,'sakiko_front_v1',(600,440),(lerp(-45,340,entry),175),lerp(.70,.61,ease(k/22)),angle=0)
  bg=petals(bg,k,seed=42 if anon else 44,count=17,colour=(250,245,244),speed=1.2,flowers=True)
 elif shot=='embrace':
  state=0 if k<22 else (1 if k<51 else (2 if k<82 else 3))
  bg=background('floral_v1',q,zoom=1.07,dx=12-24*q) if state in (0,3) else (pattern('slate',n,angle=3+q*8) if state==1 else background('sea_v1',q,zoom=1.10,dx=12-24*q))
  if state==1:bg=rays(bg,(520,95),125,k,(236,234,239))
  bg=layer(bg,'hug_side_v1',(530,560),(485-15*q,lerp(150,248,ease(q,2))),lerp(.70,.40,ease(q,2)),angle=0)
  if state in (0,3):bg=petals(bg,k,seed=61,count=14,colour=(221,190,208),speed=.7,flowers=True)
  white=state==1
 elif shot=='back':
  bg=pattern('slate',n,angle=0,zoom=1.02)
  bg=layer(bg,'hug_back_v1',(610,780),(535-12*e,lerp(500,260,e)),lerp(.82,.65,e),angle=0)
  if k<13:bg=cv2.GaussianBlur(bg,(0,0),max(.1,4.2*(1-k/13)))
  white=True
 elif shot=='eye':
  bg=pattern('slate',n,angle=0,zoom=1.07)
  bg=layer(bg,'hug_eye_v1',(960,430),(498-35*e,lerp(-70,165,ease(k/10))),lerp(1.04,.88,e),angle=0)
  if k<8:bg=cv2.GaussianBlur(bg,(0,0),max(.1,2.5*(1-k/8)))
  white=True
 else:raise ValueError(shot)
 bg=caption(bg,shot,white)
 out=np.zeros((H,W,3),np.uint8);out[53:487]=bg
 return cv2.cvtColor(out,cv2.COLOR_RGB2BGR),shot

def ff(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*map(str,args)],check=True)

def main(preview=False):
 cap=cv2.VideoCapture(str(SRC));cap.set(cv2.CAP_PROP_POS_FRAMES,1800)
 selected=set(range(1800,2700,30))
 for _,a,b in SHOTS:selected.update([a,(a+b)//2,b-1])
 contact=[];coverage=[]
 if not preview:
  proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','960x540','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(OUT/'silent_v1.mp4')],stdin=subprocess.PIPE)
 for n in range(1800,2700):
  ok,src=cap.read();assert ok,n
  if preview and n not in selected:continue
  im,shot=frame(n,src)
  assert im.shape==(H,W,3) and im.dtype==np.uint8
  if not preview:proc.stdin.write(im.tobytes())
  coverage.append({'frame':n,'shot':shot,'mode':'source_reuse' if shot in REUSE else 'new_layers'})
  if n in selected:
   rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));rgb.save(QA/f'f{n}.jpg',quality=92)
   orig=Image.fromarray(cv2.cvtColor(cv2.resize(src,(W,H)),cv2.COLOR_BGR2RGB))
   pair=Image.new('RGB',(640,180));pair.paste(orig.resize((320,180)),(0,0));pair.paste(rgb.resize((320,180)),(320,0));contact.append((n,pair))
 if not preview:
  proc.stdin.close();assert proc.wait()==0
  ff(['-i',OUT/'silent_v1.mp4','-ss','60','-i',SRC,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-t','30','-movflags','+faststart',OUT/'rebuild_60_90_v1.mp4'])
  ff(['-ss','60','-t','30','-i',SRC,'-i',OUT/'rebuild_60_90_v1.mp4','-filter_complex','[0:v]fps=30,scale=960:540,setsar=1[l];[l][1:v]hstack=inputs=2[v]','-map','[v]','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','18','-c:a','copy','-t','30','-movflags','+faststart',OUT/'source_vs_rebuild_60_90_v1.mp4'])
  (QA/'coverage.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
 for page in range(math.ceil(len(contact)/12)):
  items=contact[page*12:(page+1)*12];sheet=Image.new('RGB',(1280,6*203),'#222');d=ImageDraw.Draw(sheet)
  for i,(n,im) in enumerate(items):
   x=i%2*640;y=i//2*203;sheet.paste(im,(x,y+22));d.text((x+5,y+4),f'{n} / {n/30:.3f}s | SOURCE / REBUILD',fill='white')
  sheet.save(QA/f'comparison_{page}.jpg',quality=93)
 (HERE/'timeline.json').write_text(json.dumps([{'shot':s,'start':a,'end':b,'mode':'source_reuse' if s in REUSE else 'new_layers'} for s,a,b in SHOTS],indent=2),encoding='utf-8')
 print(json.dumps({'preview':preview,'checked_frames':len(coverage),'contact_pages':math.ceil(len(contact)/12)}))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args();main(args.preview)
