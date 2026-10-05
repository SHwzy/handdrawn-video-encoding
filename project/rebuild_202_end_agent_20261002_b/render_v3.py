from pathlib import Path
import cv2,numpy as np,subprocess,json
import render_v2 as prior
old=prior.old;P=Path(__file__).resolve().parent;V=P/'v3';W,H=960,540
c=cv2.VideoCapture(str(V/'h3/whisper_result.mp4'));whisper=[]
while True:
 ok,b=c.read()
 if not ok:break
 whisper.append(cv2.cvtColor(cv2.resize(b,(W,H)),cv2.COLOR_BGR2RGB))
assert len(whisper)==56,len(whisper)
# Keep the original edit point but cut the generated approach before it settles.
hands_indices=[round(k*36/33) for k in range(34)]
(V/'retiming.json').write_text(json.dumps({'source_range':[6257,6291],'h3_file':'h3/hands_v2_v1.mp4','old_indices':[round(k*55/33) for k in range(34)],'new_indices':hands_indices,'reason':'cut into whisper while approach is still moving; discard settled generated tail; source cut remains 6291'},indent=2))
def render(f,src):
 if 6257<=f<6291:out=old.captions(old.vids['hands'][hands_indices[f-6257]].copy(),src)
 elif 6291<=f<6332:
  out=whisper[round((min(f,6330)-6291)/39*55)].copy()
  if f>=6325:
   cell=int(np.interp(f,[6325,6331],[3,24]));out=cv2.resize(cv2.resize(out,(W//cell,H//cell),interpolation=cv2.INTER_AREA),(W,H),interpolation=cv2.INTER_NEAREST)
  out=old.captions(out,src)
 else:return prior.render(f,src)
 out[:53]=0;out[487:]=0;return out
def main():
 c=cv2.VideoCapture(str(old.SRC));c.set(cv2.CAP_PROP_POS_FRAMES,old.START)
 dest=P/'output/segment_202p567_230_v3.mp4'
 cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','30','-i','-','-ss',str(old.START/30),'-i',str(old.SRC),'-map','0:v','-map','1:a','-t',str((old.END-old.START)/30),'-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(dest)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 for f in range(old.START,old.END):
  ok,b=c.read();assert ok;src=cv2.cvtColor(cv2.resize(b,(W,H)),cv2.COLOR_BGR2RGB);out=render(f,src);proc.stdin.write(out.tobytes())
 proc.stdin.close();assert proc.wait()==0;print(dest)
if __name__=='__main__':main()
