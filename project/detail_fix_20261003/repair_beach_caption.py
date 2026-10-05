from pathlib import Path
import subprocess,cv2,numpy as np,json
from PIL import Image
from render_fixes import rt,P
from fixes import caption
def repair(a,n):
 lines=None
 if 1973<=n<2034:lines=['何時まで行こうか','何処まで行けるのか']
 elif 2102<=n<2130 or 2172<=n<2192:lines=['定かじゃないなら何を想うの']
 if lines:
  # This measured strip is sand behind the legacy caption; it contains no body.
  # Remove the whole old column, including colored antialiased pixels and connected words.
  # Rebuild sand from adjacent measured texture, with the row lighting gradient.
  # A large inpaint mask produced a visible smear beside the shoe shadow.
  tile=np.tile(a[280:880,1876:1916].astype(float),(1,2,1))
  texture=tile-tile.mean(axis=1,keepdims=True)
  left=np.median(a[280:880,1764:1796],axis=1)[:,None,:]
  right=np.median(a[280:880,1876:1916],axis=1)[:,None,:]
  weight=np.linspace(0,1,80)[None,:,None]
  a[280:880,1796:1876]=np.clip(left*(1-weight)+right*weight+texture,0,255).astype('uint8')
  a=rt.raw(caption(rt.tagged(a,2),lines,(29,28,35))).astype('uint8')
 return a
def main():
 rt.SCALE=2
 src=P/'output/montage_v5_parts/03.mp4';tmp=src.with_name('03.caption_repaired.mp4')
 original=src.with_name('03.before_caption_repair.mp4')
 c=cv2.VideoCapture(str(original if original.exists() else src));proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],stdin=subprocess.PIPE)
 i=0
 while True:
  ok,b=c.read()
  if not ok:break
  a=repair(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),1790+i)
  if 1790+i==1973:Image.fromarray(a).save(P/'qa/beach_caption_repaired_1973.png')
  proc.stdin.write(a.tobytes());i+=1
 c.release();proc.stdin.close();assert proc.wait()==0 and i==402
 archive=src.with_name('03.inpaint_caption_trial.mp4') if original.exists() else original
 assert not archive.exists();src.rename(archive);tmp.rename(src)
 (P/'beach_caption_repair.json').write_text(json.dumps({'source_frames':[[1973,2034],[2102,2130],[2172,2192]],'strip_pixels':[1796,280,1876,880],'method':'Remove old text in measured sand-only strip and redraw unified caption; preserve pre-repair chapter.'},indent=2))
 print('Repaired chapter 3',i)
if __name__=='__main__':main()
