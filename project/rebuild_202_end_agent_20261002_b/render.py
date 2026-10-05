from pathlib import Path
import cv2,numpy as np,json,subprocess,sys
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;R=P.parent;W,H=960,540;START,END=6077,6900;SRC=next(R.glob('*.mp4'))
def load(name,alpha=False):
 im=np.array(Image.open(P/'assets'/f'{name}.png').convert('RGBA' if alpha else 'RGB'))
 return im if alpha else cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA)
assets={n:load(n) for n in ['choke','fire_base','back_background','bow']}
chars={n:load(n,True) for n in ['choke_full','back_character']}
# Resample only after premultiplication.
def over(bg,im,scale,x,y):
 a=im[:,:,3:4].astype(np.float32)/255;pre=np.concatenate([im[:,:,:3].astype(np.float32)*a,a],2)
 m=np.array([[scale,0,x],[0,scale,y]],np.float32);v=cv2.warpAffine(pre,m,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 return np.clip(bg.astype(float)*(1-v[:,:,3:4])+v[:,:,:3],0,255).astype('uint8')
def zoom(im,s,cx=480,cy=270):
 return cv2.warpAffine(im,np.float32([[s,0,cx*(1-s)],[0,s,cy*(1-s)]]),(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(240,240,240))
vids={}
for tag,file in [('spin','spin_v1'),('pair','pair_v1'),('lean','lean_v1'),('hands','hands_v2_v1'),('close','close_v2_v1')]:
 path=P/'h3'/f'{file}.mp4'
 if not path.exists():continue
 c=cv2.VideoCapture(str(path));v=[]
 while True:
  ok,b=c.read()
  if not ok:break
  v.append(cv2.cvtColor(cv2.resize(b,(W,H)),cv2.COLOR_BGR2RGB))
 vids[tag]=v
back={r[0]:r[1:] for r in json.loads((P/'back_track.json').read_text())};motion=json.loads((P/'motion.json').read_text());choke_m={r['f']:r['m'] for r in motion['choke']['rows']}
rng=np.random.default_rng(20261002);yy,xx=np.mgrid[:H,:W];dark=np.zeros((H,W,3),np.float32);dark[:]=[34,35,44];dark+=rng.normal(0,1.5,(H,W,1));dark[::2,::2]*=.83;dark=np.uint8(np.clip(dark,0,255))
def anim(tag,f,a,b):
 v=vids[tag];t=(f-a)/max(b-a-1,1);return v[int(round(t*(len(v)-1)))].copy()
def captions(dst,src,darkbg=False):
 # Original captions only; no source character or scene plate.
 x0,x1=902,934;y0,y1=145,435;roi=src[y0:y1,x0:x1].astype(float);mx=roi.max(2);mn=roi.min(2)
 mask=np.clip(((mn-120)/90 if darkbg else (140-mx)/80),0,1)*(mx-mn<60)
 dst[y0:y1,x0:x1]=np.uint8(dst[y0:y1,x0:x1]*(1-mask[:,:,None])+roi*mask[:,:,None]);return dst

def render(f,src):
 changed=True
 if f<6094:
  # Whole bodies drawn with overscan. Motion measured on original rigid character pair.
  m=np.array(choke_m[f]);dy=m[1,2];im=chars['choke_full'];s=.47
  out=over(dark,im,s,278,-11+dy)
  if f<6081:
   flash=(6081-f)/4*.14;out=np.uint8(out*(1-flash)+220*flash)
  out=captions(out,src,True)
 elif f<6120:out=src.copy();changed=False
 elif f<6149:
  # Source has two repeated revolutions; H3 made one effective revolution + hold.
  j=f-6120;phase=(j%15)/14;idx=int(round(phase*35));out=vids['spin'][idx].copy();out=captions(out,src)
 elif f<6205:out=src.copy();changed=False
 elif f<6227:out=captions(anim('pair',f,6205,6227),src)
 elif f<6257:out=captions(anim('lean',f,6227,6257),src)
 elif f<6291:out=captions(anim('hands',f,6257,6291),src)
 elif f<6332:
  out=anim('close',min(f,6330),6291,6331)
  if f>=6325:
   cell=int(np.interp(f,[6325,6331],[3,24]));out=cv2.resize(cv2.resize(out,(W//cell,H//cell),interpolation=cv2.INTER_AREA),(W,H),interpolation=cv2.INTER_NEAREST)
  out=captions(out,src)
 elif f<6535:out=src.copy();changed=False
 elif f<6644:
  t=f-6535;s=1+.58*np.exp(-t/22)-.005*t/108;out=zoom(assets['fire_base'],s)
  # Saturation/chroma extracts the original red/pink flame effect, excluding gray scene.
  z=src.astype(np.float32);r,g,b=z[:,:,0],z[:,:,1],z[:,:,2];chroma=r-np.maximum(g,b*.68);alpha=np.clip((chroma-9)/45,0,1)*(r>g+12)*(r>b+8)
  out=np.uint8(out*(1-alpha[:,:,None])+z*alpha[:,:,None])
 elif f<6698:
  k=f-6644;bg=zoom(assets['back_background'],1+.09*np.exp(-k/12))
  # Track measured head bounds; first blurred/merged components replaced by bounded extrapolation.
  x,y,w,h,area=back[f]
  if f<6649:
   w=float(np.interp(f,[6644,6649],[720,476]));y=float(np.interp(f,[6644,6649],[-90,43]));center=530
  else:center=x+w/2
  im=chars['back_character'];s=w/(im.shape[1]*.64);out=over(bg,im,s,center-im.shape[1]*.5*s,y-im.shape[0]*.01*s)
  if f<6653:out=cv2.GaussianBlur(out,(0,0),max(.2,(6653-f)*.22))
 elif f<6713:
  # Explicit user exception: recolor only original dark hair under the witch hat.
  out=src.copy();z=out.astype(np.float32);r,g,b=z[:,:,0],z[:,:,1],z[:,:,2];mask=(b>r*1.18)&(b>g*1.06)&(b<130)&(xx>260)&(xx<740)&(yy>325)&(yy<486)
  a=mask.astype(np.float32)[:,:,None];lum=cv2.cvtColor(out,cv2.COLOR_RGB2GRAY).astype(float);col=np.stack([lum*.45+120,lum*.45+161,lum*.45+196],2);out=np.uint8(z*(1-a)+np.clip(col,0,255)*a)
 elif f<6742:out=src.copy();changed=False
 elif f<6752:out=assets['bow'].copy()
 else:out=src.copy();changed=False
 if changed:out[:53]=0;out[487:]=0
 return out

def main():
 c=cv2.VideoCapture(str(SRC));(P/'qa').mkdir(exist_ok=True)
 if '--samples' in sys.argv:
  fs=[6077,6081,6093,6120,6127,6134,6147,6205,6215,6226,6235,6256,6257,6274,6290,6291,6310,6330,6535,6555,6590,6643,6644,6652,6670,6697,6698,6704,6712,6742,6751,6752]
  sheet=Image.new('RGB',(960,155*((len(fs)+3)//4)));d=ImageDraw.Draw(sheet)
  for i,f in enumerate(fs):
   c.set(cv2.CAP_PROP_POS_FRAMES,f);src=cv2.cvtColor(cv2.resize(c.read()[1],(W,H)),cv2.COLOR_BGR2RGB);out=render(f,src);Image.fromarray(out).save(P/'qa'/f'f{f}.png');im=Image.fromarray(out);im.thumbnail((240,135));x=i%4*240;y=i//4*155;sheet.paste(im,(x,y));d.text((x,y+135),str(f),fill='white')
  sheet.save(P/'qa/candidate_sheet.jpg');return
 c.set(cv2.CAP_PROP_POS_FRAMES,START)
 cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-ss',str(START/30),'-i',str(SRC),'-map','0:v','-map','1:a','-t',str((END-START)/30),'-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(P/'output/segment_202p567_230_v1.mp4')]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 for f in range(START,END):
  ok,b=c.read();assert ok;src=cv2.cvtColor(cv2.resize(b,(W,H)),cv2.COLOR_BGR2RGB);out=render(f,src);proc.stdin.write(out.tobytes())
 proc.stdin.close();assert proc.wait()==0;print('Rendered',END-START,'frames')
if __name__=='__main__':main()
