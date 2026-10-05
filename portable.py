"""Local path adapter and read audit. No network or generation is used."""
from pathlib import Path
import os,sys,builtins,io,json,atexit
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT/'project'
OLD='F:/手书'
READS=set()
_loads=json.loads
OVERRIDES=_loads((ROOT/'asset_overrides.json').read_text(encoding='utf8')) if (ROOT/'asset_overrides.json').exists() else {}
def portable_text(value):
 if not isinstance(value,str):return value
 if str(ROOT).replace('\\','/') not in value.replace('\\','/'):
  value=value.replace('F:\\手书',str(PROJECT)).replace('F:/手书',str(PROJECT))
 for prefix in ['C:/Windows/Fonts/','C:\\Windows\\Fonts\\']:
  value=value.replace(prefix,str(ROOT/'fonts')+os.sep)
 return value
def convert(v):
 if isinstance(v,dict):return {portable_text(k):convert(x) for k,x in v.items()}
 if isinstance(v,list):return [convert(x) for x in v]
 return portable_text(v)
def route(value):
 if not isinstance(value,(str,os.PathLike)):return value
 p=Path(portable_text(str(value))).absolute()
 try:key=p.relative_to(PROJECT).as_posix()
 except ValueError:
  try:key=p.relative_to(ROOT).as_posix()
  except ValueError:key=None
 if key in OVERRIDES:p=(ROOT/OVERRIDES[key]).resolve()
 return str(p)
def check(path):
 if not isinstance(path,(str,os.PathLike)):return
 p=Path(path).absolute();s=p.as_posix().lower()
 if s.startswith(OLD.lower()+'/') and not p.is_relative_to(ROOT):
  raise RuntimeError('External old-project dependency blocked: '+str(p))
 if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.mp4','.mov','.wav','.m4a','.ttf','.ttc','.otf','.npy','.npz'} and not p.is_relative_to(ROOT) and not p.is_relative_to(Path(sys.prefix)):
  raise RuntimeError('External media dependency blocked: '+str(p))
 if p.is_relative_to(ROOT):READS.add(p.relative_to(ROOT).as_posix())
def install():
 oldexists=Path.exists
 Path.exists=lambda self,*a,**kw:oldexists(Path(route(self)),*a,**kw)
 os.environ['PATH']=str(ROOT/'tools')+os.pathsep+os.environ.get('PATH','')
 builtins.portable_text=portable_text
 oldopen=builtins.open;oldio=io.open
 def opened(file,*a,**kw):
  f=route(file);check(f);return oldopen(f,*a,**kw)
 def ioopened(file,*a,**kw):
  f=route(file);check(f);return oldio(f,*a,**kw)
 builtins.open=opened;io.open=ioopened
 json.loads=lambda *a,**kw:convert(_loads(*a,**kw))
 def audit(event,args):
  if event=='open':check(args[0])
 sys.addaudithook(audit)
 import cv2,numpy as np
 capture=cv2.VideoCapture;imread=cv2.imread;fromfile=np.fromfile
 def cap(p,*a,**kw):
  p=route(p);check(p);return capture(p,*a,**kw)
 def read(p,*a,**kw):
  p=route(p);check(p);return imread(p,*a,**kw)
 def arr(p,*a,**kw):
  p=route(p);check(p);return fromfile(p,*a,**kw)
 cv2.VideoCapture=cap;cv2.imread=read;np.fromfile=arr
 @atexit.register
 def save():
  dest=os.environ.get('HANDDRAWN_READ_LOG')
  if dest:
   Path(dest).parent.mkdir(parents=True,exist_ok=True)
   with oldopen(dest,'w',encoding='utf8') as f:f.write(json.dumps(sorted(READS),ensure_ascii=False,indent=2))
