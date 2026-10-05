"""Independent artwork + H3 actions + deterministic compositing, source [5460,6077)."""
from pathlib import Path
import cv2,numpy as np,json,functools,subprocess,argparse
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;R=P.parent;SRC=R/json.loads((P/'scope.json').read_text(encoding='utf-8'))['source'];W,H=960,540
Y,X=np.mgrid[:H,:W].astype(np.float32);A=P/'assets';O=P/'output';Q=P/'qa_v4';Q.mkdir(exist_ok=True)
def smooth(t):
 t=np.clip(t,0,1);return t*t*(3-2*t)
def mix(a,b,m):return a*(1-m[:,:,None])+b*m[:,:,None] if np.ndim(m) else a*(1-m)+b*m
def mat(s=1,x=0,y=0):return np.float32([[s,0,x],[0,s,y]])
def mul(a,b):return (np.vstack([a,[0,0,1]])@np.vstack([b,[0,0,1]]))[:2].astype(np.float32)
def warp(a,m):
 # Area prefilter before affine minification; avoid beating halftone against output pixels.
 scale=float(np.linalg.norm(m[:,0]))
 if scale<.8 and max(a.shape[:2])>960:
  ratio=min(1.,max(.15,scale*1.15));hh,ww=a.shape[:2];nw,nh=max(1,round(ww*ratio)),max(1,round(hh*ratio))
  a=cv2.resize(a,(nw,nh),interpolation=cv2.INTER_AREA);m=m.copy();m[:,0]*=ww/nw;m[:,1]*=hh/nh
 return cv2.warpAffine(a,m,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
def over(b,a):return b*(1-a[:,:,3:4])+a[:,:,:3]
def save(path,a):Image.fromarray(np.clip(a,0,255).astype('uint8')).save(path)
@functools.lru_cache(None)
def art(name):
 path=R/name[3:] if name.startswith('../') else A/name
 replacements={'hood_full_raw.png':'hood_full.png','dip_bone.png':'dip_bone.png','dip_solo.png':'dip_solo.png','dip_normal.png':'dip_normal.png','hand_s_sad.png':'hand_s_sad.png'}
 if name in replacements:path=P/'assets_v2'/replacements[name]
 if name=='hood_full_raw.png':path=P/'assets_v4/hood_full_user.png'
 if name=='dance_solo.png':path=P/'assets_v4/dance_solo.png'
 im=np.array(Image.open(path).convert('RGBA'),np.float32);im[:,:,3]=np.clip((im[:,:,3]-1)/252,0,1);im[:,:,:3]*=im[:,:,3:4];return im
@functools.lru_cache(None)
def bounds(name):
 a=art(name);yy,xx=np.where(a[:,:,3]>.5);return xx.min(),yy.min(),xx.max()+1,yy.max()+1
def cel(name,x,y,width=None,height=None,c=None,canvas=False):
 a=art(name)
 x0,y0,x1,y1=(0,0,a.shape[1],a.shape[0]) if canvas else bounds(name)
 s=width/(x1-x0) if width is not None else height/(y1-y0)
 m=mat(s,x-x0*s,y-y0*s)
 return warp(a,m if c is None else mul(c,m))
@functools.lru_cache(None)
def raster(name):return np.array(Image.open(P/'assets_v3'/name if name=='prone_hood.png' else A/name).convert('RGB'),np.float32)
def plate(name,dx=0,dy=0,zoom=1,angle=0):
 im=raster(name);s=max(W/im.shape[1],434/im.shape[0])*zoom
 m=cv2.getRotationMatrix2D((im.shape[1]/2,im.shape[0]/2),angle,s);m[:,2]+=np.array([480+dx,270+dy])-np.array([im.shape[1]/2,im.shape[0]/2]);return warp(im,m)
def fit_active(a):
 h,w=a.shape[:2];s=max(960/w,434/h);ww,hh=round(w*s),round(h*s)
 a=cv2.resize(a,(ww,hh),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
 return a[(hh-434)//2:(hh-434)//2+434,(ww-960)//2:(ww-960)//2+960].astype(np.float32)
@functools.lru_cache(None)
def video(name):
 c=cv2.VideoCapture(str(P/'h3'/f'{name}_v1.mp4'));out=[]
 while True:
  ok,a=c.read()
  if not ok:break
  a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
  if name=='sakiko':a=a[74:694]
  out.append(fit_active(a))
 return out
def action(name,t):
 frames=video(name);im=np.full((H,W,3),246,np.float32);im[53:487]=frames[int(np.clip(t,0,1)*(len(frames)-1))];return im
def full(name):
 im=np.full((H,W,3),245,np.float32);im[53:487]=cv2.resize(raster(name),(960,434));return im
RAW=json.loads((P/'analysis/motion_raw.json').read_text())
def track(key,n):
 rs=RAW[key]['rows'];ns=np.array([r['frame'] for r in rs]);return np.array([np.interp(n,ns,[r['center'][0] for r in rs]),np.interp(n,ns,[r['center'][1] for r in rs])])
def camera(s=1,dx=0,dy=0,angle=0):
 m=cv2.getRotationMatrix2D((480,270),angle,s);m[:,2]+=[dx,dy];return m.astype(np.float32)
@functools.lru_cache(None)
def noise(seed=1):return cv2.resize(np.random.default_rng(seed).random((20,32)).astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)
def reveal(n,start,end,kind=0):
 if n<=start:return np.zeros((H,W),np.float32)
 if n>=end:return np.ones((H,W),np.float32)
 t=(n-start)/(end-start);u=X/W;v=Y/H
 if kind==1:u=1-u
 if kind==2:u=1-np.abs(u-.5)*2
 field=u+.16*np.sin(v*13+u*3)+.08*np.sin(v*31-u*9)+.1*(noise(14)-.5)
 # Torn, overlapping paper-shaped islands: coherent in screen coordinates.
 field+=.045*np.sin(u*63+v*27)*np.sin(v*19)
 return smooth((t*1.65-.32-field)/.022+.5)
def glitch(n):
 a=np.zeros((H,W,3),np.float32);rng=np.random.default_rng(108)
 cols=[[29,54,109],[230,24,107],[166,182,136],[236,233,244],[59,39,70],[50,159,166]]
 for j in range(200):
  x,y=rng.integers(-100,W),rng.integers(-100,H);w,h=rng.integers(25,200),rng.integers(8,90);col=cols[j%len(cols)];cv2.rectangle(a,(int(x),int(y)),(int(x+w),int(y+h)),col,-1)
 a=cv2.GaussianBlur(a,(0,0),5);a+=(noise(3)[:,:,None]-.5)*75;return np.clip(a,0,255)
@functools.lru_cache(None)
def pattern(kind):
 a=np.zeros((H,W,3),np.float32)
 dark=kind in ['pink_dark','gold_dark','flowers','halo']
 a[:]=[31,32,43] if dark else [242,241,244]
 col=(188,65,111) if 'pink' in kind else (171,151,65) if 'gold' in kind else (100,100,112)
 if kind in ['flowers','halo']:
  for x,y,r in [(55,101,7),(222,305,5),(445,420,6),(817,183,7),(735,345,5),(875,468,5),(376,114,4)]:
   for j in range(5):
    ag=j*np.pi*2/5;cv2.circle(a,(int(x+np.cos(ag)*r*.65),int(y+np.sin(ag)*r*.65)),int(r*.58),(235,234,239),-1,cv2.LINE_AA)
  if kind=='halo':
   for r in [115,122,128]:cv2.circle(a,(515,165),r,(246,244,247),1,cv2.LINE_AA)
   for j in range(28):
    ag=j*np.pi*2/28;r2=168 if j%3==0 else 147;cv2.line(a,(int(515+117*np.cos(ag)),int(165+117*np.sin(ag))),(int(515+r2*np.cos(ag)),int(165+r2*np.sin(ag))),(246,244,247),2,cv2.LINE_AA)
 else:
  for x in range(-100,1100,120):
   for y in range(-80,650,120):
    if 'pink' in kind:
     cv2.circle(a,(x,y),60,col,1,cv2.LINE_AA);cv2.line(a,(x-60,y),(x+60,y),col,1);cv2.line(a,(x,y-60),(x,y+60),col,1)
     cv2.polylines(a,[np.array([[x-60,y],[x,y-60],[x+60,y],[x,y+60]])],True,col,1,cv2.LINE_AA)
    else:
     for r in [25,60]:cv2.polylines(a,[np.array([[x-r,y-r],[x+r,y-r],[x+r,y+r],[x-r,y+r]])],True,col,1,cv2.LINE_AA)
     cv2.line(a,(x-60,y-60),(x+60,y+60),col,1);cv2.line(a,(x+60,y-60),(x-60,y+60),col,1)
 a[::3,::3]*=.92;return a
@functools.lru_cache(None)
def windows(bright=False):
 a=np.full((H,W,3),[160,157,163],np.float32)
 for x in range(-80,1100,185):
  col=(244,251,244) if bright else (37,37,47)
  cv2.rectangle(a,(x,170),(x+85,515),col,-1);cv2.ellipse(a,(x+42,170),(42,65),0,180,360,col,-1)
  cv2.rectangle(a,(x-4,170),(x+89,520),(242,239,243),2);cv2.ellipse(a,(x+42,170),(46,69),0,180,360,(242,239,243),2)
  for xx in [x+27,x+58]:cv2.line(a,(xx,110),(xx,516),(235,232,237),2)
  for yy in [177,252,386]:cv2.line(a,(x,yy),(x+85,yy),(235,232,237),2)
  cv2.rectangle(a,(x-24,-40),(x+112,68),(191,188,193),2)
 if bright:a=mix(a,np.full_like(a,[222,241,227]),.47)
 a[::3,::3]*=.92;return a
def crowd_bg(n):
 d=(n-5900)*.2;b=plate('facade.png',dx=d*.25,zoom=1.15)
 # Reuse independently drawn anonymous people as far silhouettes.
 layer=cel('crowd.png',-35+d,235,width=1060)
 al=layer[:,:,3:4];layer[:,:,:3]=np.array([41,57,87])*al
 b=over(b,layer);fog=np.clip((Y-414)/80,0,1)*.8;b=mix(b,np.full_like(b,248),fog);return b
def forest(n):
 dx=-15+35*float(smooth((n-5499)/95))
 b=plate('forest_back.png',dx=dx*.6,zoom=1.14)
 b=over(b,cel('hood_full_raw.png',662+dx*.12,151,height=334))
 b=over(b,cel('forest_front.png',-70+dx,45,height=460));return b
def crowd(n,second=False):
 # All people share one translation: Anon and Sakiko stay fixed relative to each other.
 t=np.clip((n-(5691 if second else 5595))/(26 if second else 67),0,1)
 d=(-22-12*t) if second else (54-80*(1-(1-t)**3))
 b=plate('facade.png',dx=d*.15,zoom=1.15)
 layer=cel('crowd.png',-24+d,88,height=355)
 layer[:,:,:3]=layer[:,:,:3]*.76+np.array([233,237,242])*layer[:,:,3:4]*.24
 b=over(b,layer)
 # Complete the rightmost background spectator with an existing independent silhouette.
 people=art('crowd.png');right=people[175:817,1240:1420];scale=352/right.shape[0]
 far=warp(right,mat(scale,865+d,93));far[:,:,:3]=far[:,:,:3]*.65+np.array([233,237,242])*far[:,:,3:4]*.35;b=over(b,far)
 b=over(b,cel('anon_full.png',342+d,98,height=374))
 b=over(b,cel('hood_full_raw.png',760+d,90,height=381))
 if second:b=warp(b,camera(1+t*.055))
 return b
@functools.lru_cache(None)
def matched_end(name):
 a=np.array(Image.open(P/'assets_v2'/f'{name}_alt.png').convert('RGB'),np.float32)
 # Two eye centers and chin measured on actual H3 last frame and replacement.
 if name=='anon':m=np.float32([[137/153,0,600-749*137/153],[0,137/153,294-349*137/153]])
 else:m=np.float32([[120/143,0,610-760*120/143],[0,120/143,266-309*120/143]])
 a=cv2.warpAffine(a,m,(1344,768),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
 if name=='sakiko':a=a[74:694]
 im=np.zeros((H,W,3),np.float32);im[53:487]=fit_active(a);return im

def alt_anon(n):return matched_end('anon').copy()
def alt_sakiko(n):return matched_end('sakiko').copy()
def dance(n):
 ns=[5773,5781,5795,5809,5820,5844,5868,5873,5877]
 dx=np.interp(n,ns,[-200,-142,-94,-64,-40,-20,0,42,220]);dy=np.interp(n,ns,[-345,-280,-174,-121,-92,-44,0,1,-15]);z=np.interp(n,ns,[.985,.985,.984,.989,.993,.997,1,1.06,1.22]);c=camera(z,dx,dy)
 # Identical registration of all three states, mapped by their shared canvas.
 cn=cel('dance_normal.png',135,-28,width=745,c=c,canvas=True)
 cs=cel('dance_solo.png',135,-28,width=745,c=c,canvas=True)
 cb=cel('dance_bone.png',135,-28,width=745,c=c,canvas=True)
 bn=crowd_bg(n);bs=mix(windows(),pattern('pink_light'),.19);bb=windows(True)
 bn=warp(bn,camera(1.08,dx*.25,dy*.20));bs=warp(bs,camera(1.15,dx*.2,dy*.15));bb=warp(bb,camera(1.15,dx*.2,dy*.15))
 normal=over(bn,cn);sad=over(bs,cs);bone=over(bb,cb)
 im=mix(normal,sad,reveal(n,5798,5820,1));im=mix(im,bone,reveal(n,5825,5841,0));im=mix(im,normal,reveal(n,5851,5872,1))
 if n<5784:
  m=reveal(n,5772,5784,0);im=mix(glitch(n),im,m)
 return im
def hand_s(n):
 # Measured drift of rigid face/hand; extend last clean measurements through whip.
 pos=track('hand_s_s',min(n,5932));base=track('hand_s_s',5925);dx,dy=pos-base
 if n<5884:dx-=max(0,5884-n)*30
 if n>5931:dx-=((n-5931)/5)**2*400
 c=camera(1,dx,dy)
 ims=[]
 for state,bg in [('normal',crowd_bg(n)),('sad',pattern('gold_light')),('bone',pattern('flowers'))]:
  ims.append(over(bg.copy(),cel('hand_s_'+state+'.png',0,53,width=960,c=c,canvas=True)))
 im=mix(ims[0],ims[1],reveal(n,5880,5892,1));im=mix(im,ims[2],reveal(n,5896,5912,1));return mix(im,ims[0],reveal(n,5912,5925,0))
def hand_a(n):
 pos=track('hand_a_s',min(n,5977));base=track('hand_a_s',5977);dx,dy=pos-base
 # Reference target cel corresponds to stable late frontal closeup.
 dx-=25;dy-=4
 if n>=5977:dx-=(n-5977)*2.5;dy-=(n-5977)*.4
 if n>5987:dx-=((n-5987)/5)**2*175
 c=camera(1,dx,dy);ims=[]
 for state,bg in [('normal',crowd_bg(n)),('sad',pattern('pink_light')),('bone',pattern('flowers'))]:ims.append(over(bg.copy(),cel('hand_a_'+state+'.png',0,53,width=960,c=c,canvas=True)))
 im=mix(ims[0],ims[1],reveal(n,5940,5953,1));im=mix(im,ims[2],reveal(n,5951,5960,1));return mix(im,ims[0],reveal(n,5968,5983,1))
def dip(n):
 z=np.interp(n,[5992,5995,6000,6012,6025,6040,6066,6073,6076],[2.1,1.72,1.48,1.22,1.10,1.055,1,.99,.79]);c=camera(z)
 normal=over(crowd_bg(n),cel('dip_normal.png',-55,24,width=940,c=c,canvas=True))
 sad=over(pattern('gray_light').copy(),cel('dip_solo.png',-55,24,width=940,c=c,canvas=True))
 bone=over(pattern('halo').copy(),cel('dip_bone.png',-55,24,width=940,c=c,canvas=True))
 im=mix(normal,sad,reveal(n,5994,6007,1));im=mix(im,bone,reveal(n,6016,6029,1));im=mix(im,normal,reveal(n,6047,6063,1))
 if n<6006:im=mix(glitch(n),im,reveal(n,5991,6006,1))
 if n>=6073:im=mix(im,np.full_like(im,246),.30*smooth((n-6072)/5))
 return im
@functools.lru_cache(None)
def lyrics(text):
 im=Image.new('L',(W,H));d=ImageDraw.Draw(im);f=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc'),12);y=175
 for ch in text:
  if ch==' ':y+=13;continue
  d.text((907,y),ch,font=f,fill=230);y+=13
 return np.array(im,np.float32)/255
def frame(n):
 if n<5463:
  # Three-frame boundary hold from prior frozen accepted clip, never rewritten.
  return frozen_tail().copy()
 if 5744<=n<5773:return source(n)
 if n<5490:a=action('prone',np.interp(n,[5463,5477,5484,5489],[0,.32,.87,1]))
 elif n<5499:a=mix(action('prone',1),full('prone_hood.png'),reveal(n,5489,5493,2))
 elif n<5595:a=forest(n)
 elif n<5663:a=crowd(n)
 elif n<5678:a=action('anon',(n-5663)/14)
 elif n<5691:a=mix(action('anon',1),alt_anon(n),reveal(n,5677,5682,2))
 elif n<5718:a=crowd(n,True)
 elif n<5733:a=action('sakiko',(n-5718)/14)
 elif n<5744:a=mix(action('sakiko',1),alt_sakiko(n),reveal(n,5732,5737,2))
 elif n<5878:a=dance(n)
 elif n<5936:a=hand_s(n)
 elif n<5992:a=hand_a(n)
 else:a=dip(n)
 # Fast spatial handoffs preserve outgoing/incoming movement at the cut.
 if 5875<=n<5882:
  t=smooth((n-5875)/7);a=spatial_handoff(dance(min(n,5877)),hand_s(max(n,5878)),t,'up')
 if 5933<=n<5940:
  t=smooth((n-5933)/7);a=spatial_handoff(hand_s(min(n,5935)),hand_a(max(n,5936)),t,'left')
 if 5989<=n<5995:
  t=smooth((n-5989)/6);a=spatial_handoff(hand_a(min(n,5991)),dip(max(n,5992)),t,'left')
 text=None
 if n<5595:text='いま忘れないよう刻まれた空気を'
 elif n<5663 or 5691<=n<5718:text='これから何度思い出すのだろう'
 elif 5773<=n<5878:text='僕らが離れるなら 僕らが迷うなら'
 elif 5878<=n<5992:text='その度に何回も繋がれる様に'
 elif n>=5992:text='ここに居てくれるなら 離さずいられたら'
 if text:
  m=lyrics(text);color=235 if 5490<=n<5595 or 6027<=n<6050 or 5903<=n<5914 or 5957<=n<5970 else 30;a=mix(a,np.full_like(a,color),m)
 a[:53]=0;a[487:]=0;return np.clip(a,0,255).astype('uint8')
def spatial_handoff(a,b,t,direction):
 if direction=='left':m1=mat(1,-t*960,0);m2=mat(1,(1-t)*960,0);mask=X<(1-t)*960
 else:m1=mat(1,0,-t*434);m2=mat(1,0,(1-t)*434);mask=Y<53+(1-t)*434
 aa=warp(a,m1);bb=warp(b,m2);return np.where(mask[:,:,None],aa,bb)
@functools.lru_cache(None)
def frozen_tail():
 c=cv2.VideoCapture(str(R/'rebuild_147_182/output/approved_147_182_v5.mp4'));c.set(cv2.CAP_PROP_POS_FRAMES,1049);ok,a=c.read();assert ok;return cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
@functools.lru_cache(None)
def source(n):
 c=cv2.VideoCapture(str(SRC));c.set(cv2.CAP_PROP_POS_FRAMES,n);ok,a=c.read();c.release();assert ok;return cv2.resize(cv2.cvtColor(a,cv2.COLOR_BGR2RGB),(960,540))
def preview():
 ns=[5463,5481,5495,5505,5580,5595,5630,5663,5675,5686,5700,5726,5740,5755,5784,5800,5817,5834,5847,5866,5883,5893,5907,5926,5942,5952,5964,5985,5996,6012,6034,6054,6068,6076]
 sh=Image.new('RGB',(1280,204*((len(ns)+3)//4)),'#222222');d=ImageDraw.Draw(sh)
 for i,n in enumerate(ns):
  a=frame(n);save(Q/f'f{n}.jpg',a);im=Image.fromarray(a).resize((320,180));x=i%4*320;y=i//4*204;sh.paste(im,(x,y));d.text((x+5,y+183),f'{n/30:.3f}s f{n}',fill='white')
 sh.save(Q/'overview_v4.jpg');print('preview saved')
def render():
 tmp=O/'silent_v4.mp4';cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-an','-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p',str(tmp)]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 for n in range(5460,6077):
  p.stdin.write(frame(n).tobytes())
  if n%60==0:print(n,flush=True)
 p.stdin.close();assert p.wait()==0
 src=SRC;subprocess.run(['ffmpeg','-v','error','-y','-i',str(tmp),'-ss','182','-i',str(src),'-map','0:v','-map','1:a','-t',str(617/30),'-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(O/'segment_182_202p567_v4.mp4')],check=True)
 print('render complete')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');args=ap.parse_args();render() if args.render else preview()
