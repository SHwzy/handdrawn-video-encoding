"""Only fix the second door-effect state: patterned whole-body silhouette."""
import render_v3 as c
import cv2,numpy as np,functools,subprocess,argparse
from PIL import Image,ImageDraw
b=c.b;v=c.v;P=b.P;Q=P/'v4/qa';Q.mkdir(parents=True,exist_ok=True)
old_door=c.door

@functools.lru_cache(None)
def silhouette():
 # Use ONLY alpha: no face, hair, dress seams, hands or original RGB survive.
 a=b.art('../rebuild_118_147/assets/door_pair.png').copy();a[:,a.shape[1]//2:]=0
 al=a[:,:,3:4];h,w=al.shape[:2];yy,xx=np.mgrid[:h,:w]
 rgb=np.zeros((h,w,3),np.float32)+[216,49,112]
 pattern=np.zeros((h,w),np.uint8)
 for cy in range(50,h+350,345):
  cx=343
  cv2.line(pattern,(cx,cy-110),(cx,cy+110),255,9,cv2.LINE_AA)
  cv2.line(pattern,(cx-100,cy),(cx+100,cy),255,9,cv2.LINE_AA)
  for radius in [22,32,44]:
   cv2.polylines(pattern,[np.array([[cx,cy-radius],[cx+radius,cy],[cx,cy+radius],[cx-radius,cy]],np.int32)],True,255,3,cv2.LINE_AA)
  for dy,flip in [(-143,1),(143,-1)]:
   cv2.polylines(pattern,[np.array([[cx-27,cy+dy],[cx+27,cy+dy],[cx,cy+dy+flip*42]],np.int32)],True,255,4,cv2.LINE_AA)
 m=pattern[:,:,None]/255;rgb=rgb*(1-m)+np.array([91,44,79])*m
 dots=((xx%5-2)**2+(yy%5-2)**2<1.5);rgb[dots]*=.72
 inner=al[:,:,0]-cv2.erode(al[:,:,0],np.ones((3,3),np.uint8));rgb=rgb*(1-inner[:,:,None])+np.array([170,24,85])*inner[:,:,None]
 p=np.concatenate([rgb*al,al],axis=2).astype(np.float32)
 return b.warp(p,b.mat(.48,110,-5))

def door(n):
 if not 4817<=n<4832:return old_door(n)
 a,d=c.door_actors();plain=np.full((540,960,3),250,np.float32)
 radius=8.5*np.clip((365-b.Y)/260,0,1)
 dots=(((b.X%24-12)**2+(b.Y%24-12)**2)<radius**2).astype(np.float32)
 columns=np.maximum(b.smooth((70-np.abs(b.X-212))/35),b.smooth((80-np.abs(b.X-651))/40))
 dotted=b.over(b.mix(plain,np.full_like(plain,[52,61,70]),dots*columns*.9),d)
 alternate=b.over(b.plate('../rebuild_118_147/backgrounds/ballroom_clean_v1.png',zoom=1.02,light=.91),silhouette())
 if n<4821:out=b.mix(dotted,alternate,b.smooth((n-4817)/4)*np.ones((540,960)))
 elif n<4830:out=alternate
 else:out=b.mix(alternate,plain,b.smooth((n-4830)/2)*np.ones((540,960)))
 return b.over(out,v.book_layer())
v.door=door

def preview():
 ns=[4809,4817,4819,4821,4824,4829,4831,4833];s=Image.new('RGB',(1280,404));dr=ImageDraw.Draw(s)
 for i,n in enumerate(ns):
  im=Image.fromarray(b.bars(door(n)));im.save(Q/f'f{n}.png');x=i%4*320;y=i//4*202;s.paste(im.resize((320,180)),(x,y));dr.text((x,y+182),str(n),fill='white')
 s.save(Q/'door_states.jpg')
 # Exact compositor equality around corrected interval and at forest transition.
 for n in list(range(4791,4817))+list(range(4832,4847)):
  assert np.array_equal(door(n),old_door(n)),n

def render():
 out=b.O/'silent_v4.mp4';proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE);src=cv2.VideoCapture(str(next(b.R.glob('*.mp4'))))
 for n in range(4410,5460):
  proc.stdin.write(v.frame(n,src).tobytes())
  if n%150==0:print(n,flush=True)
 proc.stdin.close();assert proc.wait()==0
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(out),'-ss','147','-i',str(next(b.R.glob('*.mp4'))),'-map','0:v:0','-map','1:a:0','-t','35','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(b.O/'segment_147_182_v4.mp4')],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args();preview() if args.preview else render()
