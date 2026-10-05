"""Third eyes-only cel; each reveal mixes complete worlds above the character."""
import render_v4 as prior
import cv2,numpy as np,functools,subprocess,argparse
from PIL import Image,ImageDraw
c=prior.c;v=c.v;b=c.b;P=b.P;Q=P/'v5/qa';Q.mkdir(parents=True,exist_ok=True)
old_early=c.early

@functools.lru_cache(None)
def eyes_asset():
 a=b.art('v5/assets/panic_eyes_only.png');d=c.pair('panic')[1]
 return cv2.resize(a,(d.shape[1],d.shape[0]))

def early(n,diagnostic=False):
 if not 4627<=n<4791:return old_early(n,diagnostic)
 a,d=v.actors('panic',n)
 transform=b.mul(v.relative('panic',n,4790),b.mat(.63,-35,0))
 eyes=b.warp(eyes_asset(),transform)
 real=v.bg('panic',n)
 normal=b.over(real,a);dark=b.over(real,d);only_eyes=b.over(b.pattern(0),eyes)
 dark_m=np.maximum(c.loops(n,4628,4665,3),b.smooth((n-4658)/21))
 floral=np.maximum(c.loops(n,4677,4713,1),b.smooth((n-4700)/28))
 floral*=1-b.smooth((n-4776)/14)
 restore=np.maximum(c.loops(n,4736,4784,2),b.smooth((n-4774)/15))
 # All three masks operate on complete background+actor composites, never underneath an actor.
 under=b.mix(normal,dark,dark_m)
 out=b.mix(under,only_eyes,floral)
 out=b.mix(out,normal,restore)
 sigma=float(np.interp(n,[4627,4642],[2,0]))
 if sigma>.05:out=cv2.GaussianBlur(out,(0,0),sigma)
 if diagnostic:return under,only_eyes,floral,out
 return out
v.early=early

def preview():
 ns=[4668,4677,4680,4686,4695,4710,4745,4788];s=Image.new('RGB',(1280,404));dr=ImageDraw.Draw(s)
 for i,n in enumerate(ns):
  im=Image.fromarray(b.bars(early(n)));im.save(Q/f'f{n}.png');x=i%4*320;y=i//4*202;s.paste(im.resize((320,180)),(x,y));dr.text((x,y+182),f'{(n-4410)/30:.2f}s / {n}',fill='white')
 s.save(Q/'eyes_transition.jpg')
 for n in [4680,4686,4695,4710]:
  A,B,m,out=early(n,True);s=Image.new('RGB',(960,540))
  for i,im in enumerate([A,B,m[:,:,None]*np.ones((1,1,3))*255,out]):s.paste(Image.fromarray(b.bars(im)).resize((480,270)),(i%2*480,i//2*270))
  s.save(Q/f'mask_{n}.jpg')

def render():
 out=b.O/'silent_v5.mp4';proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE);src=cv2.VideoCapture(str(next(b.R.glob('*.mp4'))))
 for n in range(4410,5460):
  proc.stdin.write(v.frame(n,src).tobytes())
  if n%150==0:print(n,flush=True)
 proc.stdin.close();assert proc.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(out),'-ss','147','-i',str(next(b.R.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t','35','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(b.O/'segment_147_182_v5.mp4')],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args();preview() if args.preview else render()
