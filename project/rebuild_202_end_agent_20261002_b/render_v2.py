from pathlib import Path
import json,cv2,numpy as np,subprocess,sys
import render as old
P=Path(__file__).resolve().parent;V=P/'v2';W,H=960,540
# Monotone cubic Hermite, smoothly varying velocity through sparse validated framing anchors.
knots=np.array([0,6,12,24,40,53],float)
values={'width':[700,455,393,342,310,273],'top':[-82,55,93,114,129,147],'center':[530,529,528,524,519,515]}
def monotone(x,ys):
 ys=np.array(ys,float);h=np.diff(knots);d=np.diff(ys)/h;m=np.empty(len(ys));m[0]=d[0];m[-1]=d[-1]
 for i in range(1,len(ys)-1):
  m[i]=0 if d[i-1]*d[i]<=0 else 3*(h[i-1]+h[i])/((2*h[i]+h[i-1])/d[i-1]+(h[i]+2*h[i-1])/d[i])
 i=min(len(h)-1,max(0,int(np.searchsorted(knots,x)-1)));t=(x-knots[i])/h[i]
 return (2*t**3-3*t*t+1)*ys[i]+(t**3-2*t*t+t)*h[i]*m[i]+(-2*t**3+3*t*t)*ys[i+1]+(t**3-t*t)*h[i]*m[i+1]
tracks=[{'f':6644+k,**{key:float(monotone(k,vals)) for key,vals in values.items()}} for k in range(54)]
(V/'back_motion.json').write_text(json.dumps({'anchors':{'frames':knots.tolist(),**values},'method':'monotone cubic Hermite; measured reliable framing anchors; rejected per-frame segmentation jitter and final clipped outlier','frames':tracks},indent=2))
im=old.chars['back_character'];a=im[:,:,3:4].astype(np.float32)/255;premul=np.concatenate([im[:,:,:3]*a,a],2).astype(np.float32)
def back(f):
 k=f-6644;t=tracks[k];s=t['width']/(im.shape[1]*.64)
 bg=old.zoom(old.assets['back_background'],1+.09*np.exp(-k/12))
 # Adaptive low-pass before shrinking suppresses halftone shimmer; transform remains subpixel.
 pre=cv2.GaussianBlur(premul,(0,0),max(.05,.5*(1/s-1)))
 mat=np.float32([[s,0,t['center']-im.shape[1]*.5*s],[0,s,t['top']-im.shape[0]*.01*s]])
 v=cv2.warpAffine(pre,mat,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 out=np.uint8(np.clip(bg*(1-v[:,:,3:4])+v[:,:,:3],0,255))
 blur=1.8*np.exp(-k/3.5)
 if blur>.08:out=cv2.GaussianBlur(out,(0,0),blur)
 return out
videos={}
for tag in ['whisper','nod']:
 path=V/'h3'/f'{tag}_result.mp4'
 if path.exists():
  cap=cv2.VideoCapture(str(path));frames=[]
  while True:
   ok,b=cap.read()
   if not ok:break
   frames.append(cv2.cvtColor(cv2.resize(b,(W,H)),cv2.COLOR_BGR2RGB))
  videos[tag]=frames

def anim(tag,f,a,b):
 v=videos[tag];return v[round((f-a)/(b-a-1)*(len(v)-1))].copy()
def render(f,src):
 if 6291<=f<6332:
  out=anim('whisper',min(f,6330),6291,6331)
  if f>=6325:
   cell=int(np.interp(f,[6325,6331],[3,24]));out=cv2.resize(cv2.resize(out,(W//cell,H//cell),interpolation=cv2.INTER_AREA),(W,H),interpolation=cv2.INTER_NEAREST)
  out=old.captions(out,src)
 elif 6644<=f<6698:out=back(f)
 elif 6698<=f<6713:out=anim('nod',f,6698,6713)
 else:return old.render(f,src)
 out[:53]=0;out[487:]=0;return out

def main():
 c=cv2.VideoCapture(str(old.SRC));start,end=old.START,old.END
 if '--back' in sys.argv:start,end=6644,6698
 if '--samples' in sys.argv:
  from PIL import Image,ImageDraw
  fs=[6291,6298,6305,6312,6319,6330,6644,6648,6652,6656,6662,6670,6680,6690,6697,6698,6701,6704,6707,6710,6712]
  sheet=Image.new('RGB',(960,155*((len(fs)+3)//4)));d=ImageDraw.Draw(sheet)
  for i,f in enumerate(fs):
   c.set(cv2.CAP_PROP_POS_FRAMES,f);src=cv2.cvtColor(cv2.resize(c.read()[1],(W,H)),cv2.COLOR_BGR2RGB);out=render(f,src);Image.fromarray(out).save(V/'qa'/f'f{f}.png');im2=Image.fromarray(out);im2.thumbnail((240,135));x=i%4*240;y=i//4*155;sheet.paste(im2,(x,y));d.text((x,y+135),str(f),fill='white')
  sheet.save(V/'qa/corrections.jpg');return
 dest=V/'qa/back_smooth.mp4' if '--back' in sys.argv else P/'output/segment_202p567_230_v2.mp4'
 cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-ss',str(start/30),'-i',str(old.SRC),'-map','0:v','-map','1:a','-t',str((end-start)/30),'-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(dest)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);c.set(cv2.CAP_PROP_POS_FRAMES,start)
 for f in range(start,end):
  ok,b=c.read();assert ok;src=cv2.cvtColor(cv2.resize(b,(W,H)),cv2.COLOR_BGR2RGB);proc.stdin.write(render(f,src).tobytes())
 proc.stdin.close();assert proc.wait()==0;print(dest)
if __name__=='__main__':main()
