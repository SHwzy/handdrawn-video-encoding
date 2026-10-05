"""Public render entry: frozen replay, editable recipe render, or relocation smoke test."""
from pathlib import Path
import argparse,json,subprocess,sys,os
P=Path(__file__).resolve().parent
FF=str(P/'tools/ffmpeg.exe')
def run(cmd):subprocess.run(cmd,check=True,cwd=P)

def assemble(listing,out,start,end):
 source=P/'assets/soundtrack.m4a'
 audio=['-c:a','copy'] if (start,end)==(0,6900) else ['-c:a','aac','-b:a','192k']
 run([FF,'-v','error','-f','concat','-safe','0','-i',str(listing),'-ss',str(start/30),'-i',str(source),'-map','0:v:0','-map','1:a:0','-t',str((end-start)/30),'-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p','-threads','4',*audio,'-movflags','+faststart',str(out)])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['replay','edit','check'],required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=6900);ap.add_argument('--output',type=Path);ap.add_argument('--scale',type=int,default=2);a=ap.parse_args()
 assert 0<=a.start<a.end<=6900
 assert a.scale==2,'This project uses 1080p pixel masks; use --scale 2.'
 timeline=json.loads((P/'timeline.json').read_text(encoding='utf8'))
 assert timeline[0]['start']==0 and timeline[-1]['end']==6900
 assert all(r['start']<r['end'] for r in timeline)
 assert all(x['end']==y['start'] for x,y in zip(timeline,timeline[1:])), 'Timeline has a gap or overlap'
 if a.mode=='check':
  groups={}
  for row in timeline:
   ns=groups.setdefault(row['recipe'],set());ns.update([row['start'],(row['start']+row['end'])//2,row['end']-1])
  # Extra shots within the long inherited ranges cover each distinct source module.
  for n in range(0,6900,30):
   r=next(r for r in timeline if r['start']<=n<r['end']);groups.setdefault(r['recipe'],set()).add(n)
  for recipe,ns in groups.items():run([sys.executable,str(P/'worker.py'),'--recipe',recipe,'--scale',str(a.scale),'--samples',*map(str,sorted(ns))])
  return
 assert a.output,'--output is required';out=a.output.resolve();assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 if a.mode=='replay':
  assert (a.start,a.end)==(0,6900),'replay is the complete frozen film; use edit for a frame range.'
  # Stream copy preserves the approved encoded video/audio. It intentionally ignores edited artwork.
  approved=json.loads((P/'baseline.json').read_text(encoding='utf8'))
  run([FF,'-v','error','-i',str(P/approved['master']),'-map','0','-c','copy','-movflags','+faststart',str(out)]);return
 parts=[]
 for row in timeline:
  start=max(a.start,row['start']);end=min(a.end,row['end'])
  if start>=end:continue
  dest=out.parent/(out.stem+'_parts')/f'{start}_{end}_{row["recipe"]}.mp4';parts.append(dest)
  run([sys.executable,str(P/'worker.py'),'--recipe',row['recipe'],'--start',str(start),'--end',str(end),'--scale',str(a.scale),'--output',str(dest)])
 listing=out.with_suffix('.concat.txt');listing.write_text('\n'.join("file '"+str(p).replace('\\','/').replace("'","'\\''")+"'" for p in parts),encoding='utf8')
 assemble(listing,out,a.start,a.end)
if __name__=='__main__':main()
