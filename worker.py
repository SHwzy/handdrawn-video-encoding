"""One isolated renderer recipe per process; prevents install-hook cross contamination."""
from pathlib import Path
import argparse,sys,os,importlib.util,json,subprocess
from portable import ROOT,PROJECT,install
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--recipe',required=True);ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);ap.add_argument('--scale',type=int,default=2);ap.add_argument('--samples',nargs='*',type=int);ap.add_argument('--output',type=Path);a=ap.parse_args()
 work=ROOT/'work'/a.recipe
 for d in ['logs','migrated','state','qa','output']:(work/d).mkdir(parents=True,exist_ok=True)
 os.environ['HANDDRAWN_READ_LOG']=str(work/'logs'/f'reads_{a.start}_{a.end}.json')
 install()
 sys.path.insert(0,str(PROJECT/'render_1080p'));sys.path.insert(0,str(PROJECT/'detail_fix_20261003'))
 os.environ['HANDDRAWN_H3_SR_MANIFEST']=str(PROJECT/'render_1080p/h3_sr/replacements.json')
 import cv2,numpy as np
 from PIL import Image
 cv2.setNumThreads(2)
 names={'v12':('detail_fix_20261003/subtitles_v12/render.py','Film'),'base':('render_1080p/render.py','Film'),'v5':('detail_fix_20261003/render_fixes.py','FixedFilm'),'v7':('detail_fix_20261003/revision_v7/revision.py','RevisedFilm'),'v9':('detail_fix_20261003/step2_v9/render.py','Film'),'v10':('detail_fix_20261003/step2_v10/render.py','Film'),'v11':('detail_fix_20261003/grounding_v11/render.py','Film')}
 path,cls=names[a.recipe];sp=importlib.util.spec_from_file_location('selected_recipe',PROJECT/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 rt=m.rt;rt.SCALE=a.scale;rt.WORK=work;rt.JOB=f'{a.recipe}_{a.start}_{a.end}'
 if a.recipe in ['v9','v10','v11','v12']:
  # The original patch classes open an old complete film in __init__. Editable mode must not need it.
  m.Film.__init__=lambda self:m.prior.RevisedFilm.__init__(self)
 f=getattr(m,cls)()
 walk_frames=None
 action_caps={}
 # Only the nod retiming recipe intentionally reads existing action footage. It is an explicit replaceable asset.
 # Never silently fetch unchanged frames from an old complete movie in editable mode.
 def frozen(n):
  assert 3479<=n<3506,('Unspecified frozen-frame branch in editable mode',a.recipe,n)
  c=cv2.VideoCapture(str(ROOT/'assets/nod_retime_input.mp4'));c.set(1,n-3479);ok,b=c.read();c.release();assert ok
  b=cv2.resize(b,(960*a.scale,540*a.scale));return rt.tagged(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),a.scale)
 if hasattr(f,'frozen'):f.frozen=frozen
 def frame(n):
  nonlocal walk_frames
  # Current approved composited action cels replace historical face-reference takes.
  if (2601<=n<2708) or (3427<=n<3479):
   name,start=('tear_scene',2601) if n<2708 else ('mouth_scene',3427)
   if name not in action_caps:action_caps[name]=cv2.VideoCapture(str(ROOT/'assets/current_actions'/(name+'.mp4')))
   cap=action_caps[name];cap.set(cv2.CAP_PROP_POS_FRAMES,n-start);ok,b=cap.read();assert ok,n
   return cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
  im=rt.raw(f.subtitle_frame(n) if a.recipe=='v12' else f.frame(n)).astype('uint8')
  if a.recipe=='v5':
   from repair_beach_caption import repair
   if a.scale!=2:im=cv2.resize(im,(1920,1080))
   im=repair(im,n)
   if 6149<=n<6205:
    from repair_tail_caption import remove_residue
    from fixes import caption
    im=rt.raw(caption(rt.tagged(remove_residue(im),2),['まだ誰も知らない感覚で'],(242,242,242))).astype('uint8')
   if a.scale!=2:im=cv2.resize(im,(960*a.scale,540*a.scale))
  # Generated nod input is clean. Its screen-space lyric is applied after all blur/retiming.
  from portable import OVERRIDES
  if 207<=n<235 and 'assets/walk_retime_input.mp4' in OVERRIDES:
   # Explicit 28-frame H3 walking repair: original scene with only obsolete long hair removed.
   if walk_frames is None:
    c=cv2.VideoCapture(str(ROOT/'assets/walk_retime_input.mp4'));walk_frames=[]
    while True:
     ok,b=c.read()
     if not ok:break
     walk_frames.append(cv2.cvtColor(cv2.resize(b,(960*a.scale,540*a.scale)),cv2.COLOR_BGR2RGB))
    c.release();assert len(walk_frames)==28,len(walk_frames)
   im=walk_frames[n-207].copy()
  if 3479<=n<3506 and 'assets/nod_retime_input.mp4' in OVERRIDES:
   from fixes import caption
   im=rt.raw(caption(rt.tagged(im,a.scale),['綻ぶ前にここを出ていこうかと'],(29,28,35))).astype('uint8')
  assert im.shape==(540*a.scale,960*a.scale,3),(n,im.shape)
  im[:53*a.scale]=0;im[487*a.scale:]=0
  return im
 if a.samples:
  for n in a.samples:
   Image.fromarray(frame(n)).save(work/'qa'/f'{n}.png');print('SAMPLE',a.recipe,n,flush=True)
 else:
  assert a.output and not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
  cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{960*a.scale}x{540*a.scale}','-r','30','-i','-','-an','-c:v','libx264','-qp','0','-preset','fast','-threads','2',str(a.output)]
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  for n in range(a.start,a.end):
   proc.stdin.write(frame(n).tobytes())
   if (n-a.start)%30==0:print('RENDER',a.recipe,n,'/',a.end,flush=True)
  proc.stdin.close();assert proc.wait()==0
 f.loader.save_log()
if __name__=='__main__':main()
