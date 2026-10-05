"""Isolated raster API: logical coordinates, dense storage, no global patches.

Native artwork stays in native coordinates. Only explicit screen allocations and
screen outputs are dense. The legacy formulas see logical shapes; OpenCV sees
physical arrays. Every transform explicitly composes source/destination grids.
"""
from pathlib import Path
import functools, json, re, hashlib, os
import numpy as N
import cv2 as C
from PIL import Image as PI, ImageDraw as PD, ImageFont as PF

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'render_1080p'
SOURCE=ROOT/"ロウワ― _ Flower -lower one's eyes P1 211111_ロウワー_youtube_440400082.mp4"
SCALE=2
JOB='samples'
READS={}
PROVIDERS={}
SR_MAP={}
if os.environ.get('HANDDRAWN_H3_SR_MANIFEST'):
    SR_MAP=json.loads(Path(os.environ['HANDDRAWN_H3_SR_MANIFEST']).read_text(encoding='utf-8'))['replacements']

def record(p):
    p=Path(p).resolve()
    if p.is_file(): READS[str(p)]=p.stat().st_size
    return p

def raw(a): return N.asarray(a)
def density(a):
    if isinstance(a,(tuple,list)) and not isinstance(a,LogicalShape):return max([density(x) for x in a]+[1])
    return getattr(a,'density',1)
def tagged(a,s):
    if not isinstance(a,N.ndarray) or a.ndim<2:return a
    if s>1 and a.dtype==N.float64:a=a.astype(N.float32)
    b=N.asarray(a).view(Raster);b.density=s;return b

class LogicalShape(tuple):
    def __new__(cls,values,s=1):
        v=super().__new__(cls,values);v.density=s;return v
    def __getitem__(self,k):
        v=super().__getitem__(k)
        return LogicalShape(v,self.density) if isinstance(k,slice) else v

class DomainInt(int):
    def __new__(cls,value,s):
        v=super().__new__(cls,value);v.density=s;return v

class Raster(N.ndarray):
    density=1
    def __array_finalize__(self,a):self.density=getattr(a,'density',1)
    @property
    def shape(self):
        v=N.ndarray.shape.__get__(self)
        return LogicalShape(tuple(DomainInt(round(x/self.density),self.density) if i<2 else x for i,x in enumerate(v)),self.density)
    def _key(self,key):
        if self.density==1:return key
        ks=list(key if isinstance(key,tuple) else (key,));axis=0
        for i,k in enumerate(ks):
            if k is None:continue
            if k is Ellipsis:axis=max(axis,self.ndim-(len(ks)-i-1));continue
            if axis<2:
                if isinstance(k,slice):
                    s=self.density
                    if k.step not in (None,1):
                        size=round(N.ndarray.shape.__get__(self)[axis]/s)
                        start,stop,step=k.indices(size)
                        # Logical stripe pixels retain their full area at dense resolution.
                        ks[i]=N.concatenate([N.arange(round(j*s),round((j+1)*s)) for j in range(start,stop,step)]) if len(range(start,stop,step)) else N.array([],dtype=int)
                    else:ks[i]=slice(None if k.start is None else round(k.start*s),None if k.stop is None else round(k.stop*s),k.step)
                elif isinstance(k,(int,N.integer)):ks[i]=round(int(k)*self.density)
            axis+=1
        # Two advanced spatial axes must describe a Cartesian pixel selection.
        if len(ks)>=2 and isinstance(ks[0],N.ndarray) and isinstance(ks[1],N.ndarray) and ks[0].ndim==ks[1].ndim==1:
            ks[0],ks[1]=N.ix_(ks[0],ks[1])
        return tuple(ks)
    def __getitem__(self,k):return tagged(raw(self)[self._key(k)],self.density)
    def __setitem__(self,k,v):
        key=self._key(k);value=raw(v) if isinstance(v,Raster) else v;target=raw(self)[key]
        if isinstance(value,N.ndarray) and value.ndim==target.ndim:
            for axis in range(min(2,value.ndim)):
                if value.shape[axis]==self.density and target.shape[axis]>self.density:
                    value=value.mean(axis,keepdims=True)
        raw(self)[key]=value

class GridSet(tuple):
    def astype(self,dtype):return GridSet(x.astype(dtype) for x in self)

def grid(slices,screen=True):
    if not isinstance(slices,tuple):return N.mgrid[slices]
    domain=next((sl.stop.density for sl in slices if isinstance(sl.stop,DomainInt)),None)
    if domain==1:screen=False
    s=domain if domain is not None else SCALE
    if (not screen and domain is None) or s==1:return N.mgrid[slices]
    vectors=[]
    for sl in slices:
        start=sl.start or 0;stop=sl.stop;step=sl.step or 1
        if isinstance(step,complex):return N.mgrid[slices]
        vectors.append(start+(N.arange(round((stop-start)/step*s))+.5)*step/s-.5*step)
    return GridSet(tagged(x,s) for x in N.meshgrid(*vectors,indexing='ij'))

def indices(shape,*a,**kw):
    return grid(tuple(slice(0,n) for n in shape)).astype(kw.get('dtype',N.int64))

def meshgrid(*vectors,**kw):
    if len(vectors)==2 and len(vectors[0])>=800 and len(vectors[1])>=400:
        v=[]
        for a in vectors:
            step=float(a[1]-a[0]);v.append(a[0]+(N.arange(len(a)*SCALE)+.5)*step/SCALE-.5*step)
        return GridSet(tagged(x,SCALE) for x in N.meshgrid(*v,**kw))
    return N.meshgrid(*vectors,**kw)

def screen_arange(stop):return (N.arange(stop*SCALE)+.5)/SCALE-.5
def grid_is_screen(values):return any(isinstance(a,Raster) and density(a)>1 for a in values.values())

class NP:
    def __getattr__(self,k):
        if k=='random':return Randoms()
        target=getattr(N,k)
        if k in {'float32','float64','uint8','int32','int64','bool_','ndarray'}:return target
        if not callable(target):return target
        @functools.wraps(target)
        def f(*args,**kwargs):
            s=max([density(a) for a in args]+[1])
            if k in {'array','asarray'} and args:
                a=args[0]
                if isinstance(a,PImage):return tagged(target(a.image,*args[1:],**kwargs),a.density)
            if k=='where' and len(args)==1:
                result=N.where(raw(args[0]))
                return tuple(x//s for x in result) if s>1 else result
            if k=='fromfile':
                p=record(args[0]);v=(N.zeros(1,N.uint8) if is_reference(p) else target(*args,**kwargs)).view(Raster);v.density=1;v.source_path=p;return v
            if k in {'zeros','ones','empty','full'} and args and isinstance(args[0],LogicalShape):
                shape=args[0];ss=shape.density
                dims=tuple(round(v*ss) if i<2 else v for i,v in enumerate(shape))
                return tagged(target(dims,*args[1:],**kwargs),ss)
            result=target(*[raw(a) if isinstance(a,Raster) else a for a in args],**kwargs)
            if isinstance(result,N.ndarray):return tagged(result,s)
            return result
        return f
    def screen(self,name,shape,*args,**kwargs):
        if len(shape)<2:return getattr(N,name)(shape,*args,**kwargs)
        s=next((x.density for x in shape[:2] if isinstance(x,DomainInt)),SCALE)
        dims=tuple(int(x*s) if i<2 else x for i,x in enumerate(shape))
        return tagged(getattr(N,name)(dims,*args,**kwargs),s)
np=NP()

class RNG:
    def __init__(self,seed):self.generator=N.random.default_rng(seed)
    def __getattr__(self,k):
        target=getattr(self.generator,k)
        def f(*a,**kw):
            result=target(*a,**kw)
            if isinstance(result,N.ndarray) and result.ndim>=2 and result.shape[0]>=400 and result.shape[1]>=800:
                # Preserve seed, sequence and original grain footprint; no SR.
                size=(result.shape[1]*SCALE,result.shape[0]*SCALE)
                out=C.resize(result.astype('float32'),size,interpolation=C.INTER_NEAREST)
                if result.ndim==3 and out.ndim==2:out=out[:,:,None]
                return tagged(out.astype(result.dtype),SCALE)
            return result
        return f
class Randoms:
    def default_rng(self,seed=None):return RNG(seed)
    def __getattr__(self,k):return getattr(N.random,k)

def pixel_matrix(s):return N.float64([[s,0,(s-1)/2],[0,s,(s-1)/2],[0,0,1]])
def transform(m,source_scale,target_scale):
    a=N.asarray(m,dtype=N.float64)
    if a.shape==(2,3):a=N.vstack([a,[0,0,1]])
    return pixel_matrix(target_scale)@a@N.linalg.inv(pixel_matrix(source_scale))

def target_density(size):
    if any(isinstance(x,DomainInt) for x in size):return next(x.density for x in size if isinstance(x,DomainInt))
    w,h=map(int,size)
    return SCALE if (w>=800 and h>=400) else 1

def prefilter(a,m):
    # Actual native/dense-source to output minification, not logical scale.
    ss=N.linalg.svd(m[:2,:2],compute_uv=False)
    small=float(min(ss))
    if small<.8:
        sigma=max(.01,.5*(1/small-1))
        return C.GaussianBlur(raw(a),(0,0),sigma)
    return raw(a)

class CV:
    def __getattr__(self,k):
        obj=getattr(C,k)
        if not callable(obj):return obj
        def call(*args,**kwargs):
            s=density(args[0]) if args else 1
            result=obj(*[raw(a) if isinstance(a,Raster) else a for a in args],**kwargs)
            if isinstance(result,N.ndarray) and k not in {'getRotationMatrix2D','invertAffineTransform','transform'}:return tagged(result,s)
            return result
        return call
    def resize(self,a,dsize,*args,output_space=None,**kwargs):
        s=target_density(dsize) if output_space!='native' else 1
        if isinstance(dsize,LogicalShape):s=dsize.density
        return tagged(C.resize(raw(a),tuple(round(x*s) for x in dsize),*args,**kwargs),s)
    def warpAffine(self,a,m,dsize,*args,output_space=None,**kwargs):
        s=1 if output_space=='native' else target_density(dsize);mm=transform(m,density(a),s)
        return tagged(C.warpAffine(prefilter(a,mm),mm[:2],tuple(round(x*s) for x in dsize),*args,**kwargs),s)
    def warpPerspective(self,a,m,dsize,*args,output_space=None,**kwargs):
        s=1 if output_space=='native' else target_density(dsize);mm=transform(m,density(a),s)
        return tagged(C.warpPerspective(raw(a),mm,tuple(round(x*s) for x in dsize),*args,**kwargs),s)
    def remap(self,a,x,y,*args,**kwargs):
        s=max(density(x),density(y));ss=density(a)
        mx=(raw(x)+.5)*ss-.5;my=(raw(y)+.5)*ss-.5
        return tagged(C.remap(raw(a),mx.astype('float32'),my.astype('float32'),*args,**kwargs),s)
    def GaussianBlur(self,a,ksize,sigmaX,*args,**kwargs):
        s=density(a);ks=tuple(0 if v==0 else max(1,round(v*s)//2*2+1) for v in ksize)
        if 'sigmaY' in kwargs:kwargs['sigmaY']*=s
        return tagged(C.GaussianBlur(raw(a),ks,sigmaX*s,*args,**kwargs),s)
    def _morph(self,k,a,kernel,*args,**kwargs):
        s=density(a)
        if s>1:kernel=C.resize(raw(kernel),(round(kernel.shape[1]*s),round(kernel.shape[0]*s)),interpolation=C.INTER_NEAREST)
        return tagged(getattr(C,k)(raw(a),kernel,*args,**kwargs),s)
    def dilate(self,*a,**kw):return self._morph('dilate',*a,**kw)
    def erode(self,*a,**kw):return self._morph('erode',*a,**kw)
    def morphologyEx(self,a,op,kernel,*args,**kwargs):
        s=density(a)
        if s>1:kernel=C.resize(raw(kernel),(round(kernel.shape[1]*s),round(kernel.shape[0]*s)),interpolation=C.INTER_NEAREST)
        return tagged(C.morphologyEx(raw(a),op,kernel,*args,**kwargs),s)
    def floodFill(self,a,mask,seedPoint,*args,**kwargs):
        s=density(a)
        return C.floodFill(raw(a),None if mask is None else raw(mask),tuple(round(x*s) for x in seedPoint),*args,**kwargs)
    def connectedComponentsWithStats(self,a,*args,**kwargs):
        count,labels,stats,centers=C.connectedComponentsWithStats(raw(a),*args,**kwargs)
        return count,tagged(labels,density(a)),stats,centers
    def _draw(self,k,a,*args,**kwargs):
        s=density(a);v=list(args)
        if k in {'line','rectangle'}:
            v[0]=tuple(round(x*s) for x in v[0]);v[1]=tuple(round(x*s) for x in v[1]);ti=3
        elif k=='circle':v[0]=tuple(round(x*s) for x in v[0]);v[1]=round(v[1]*s);ti=3
        elif k=='ellipse':v[0]=tuple(round(x*s) for x in v[0]);v[1]=tuple(round(x*s) for x in v[1]);ti=6
        elif k in {'fillPoly','polylines'}:
            v[0]=[N.rint(raw(x)*s).astype('int32') for x in v[0]];ti=3 if k=='polylines' else -1
        else:ti=-1
        if ti>=0 and len(v)>ti and v[ti]>0:v[ti]=max(1,round(v[ti]*s))
        if kwargs.get('thickness',-1)>0:kwargs['thickness']=max(1,round(kwargs['thickness']*s))
        getattr(C,k)(raw(a),*v,**kwargs);return a
    def line(self,*a,**kw):return self._draw('line',*a,**kw)
    def circle(self,*a,**kw):return self._draw('circle',*a,**kw)
    def ellipse(self,*a,**kw):return self._draw('ellipse',*a,**kw)
    def rectangle(self,*a,**kw):return self._draw('rectangle',*a,**kw)
    def polylines(self,*a,**kw):return self._draw('polylines',*a,**kw)
    def fillPoly(self,*a,**kw):return self._draw('fillPoly',*a,**kw)
    def imdecode(self,a,*args,**kwargs):
        p=getattr(a,'source_path',None)
        if p and is_reference(p):return reference(p,True)
        return C.imdecode(raw(a),*args,**kwargs)
    def VideoCapture(self,p):return Capture(p)
cv=CV()

def is_reference(p):return 'reference' in Path(p).parts and re.fullmatch(r'f\d+\.(jpg|png)',Path(p).name)
SOURCE_CAP=None
@functools.lru_cache(maxsize=3)
def source_frame(n):
    global SOURCE_CAP
    if SOURCE_CAP is None:SOURCE_CAP=C.VideoCapture(str(record(SOURCE)))
    if round(SOURCE_CAP.get(C.CAP_PROP_POS_FRAMES))!=int(n):SOURCE_CAP.set(C.CAP_PROP_POS_FRAMES,int(n))
    ok,a=SOURCE_CAP.read();assert ok,n
    if SCALE==1:return tagged(C.resize(a,(960,540),interpolation=C.INTER_AREA),1)
    return tagged(a,2)
def reference(p,bgr=False):
    n=int(re.search(r'f(\d+)',Path(p).name)[1]);record(p);a=source_frame(n).copy()
    return a if bgr else tagged(C.cvtColor(raw(a),C.COLOR_BGR2RGB),density(a))

class Capture:
    def __init__(self,p):
        self.path=Path(p).resolve();self.pos=0;self.cap=None
        self.provider=PROVIDERS.get(str(self.path))
        self.original=self.path==SOURCE or self.path.name=='original_first30.mp4'
        self.sr_density=1
        if not self.provider and not self.original:
            replacement=SR_MAP.get(str(self.path))
            if replacement:
                record(self.path);self.sr_density=replacement['spatial_density'];actual=Path(replacement['output'])
                self.cap=C.VideoCapture(str(record(actual)));assert self.cap.isOpened(),actual
            else:self.cap=C.VideoCapture(str(record(self.path)))
    def read(self):
        if self.provider:
            n=self.pos;self.pos+=1
            try:return True,self.provider(n)
            except (IndexError,ValueError):return False,None
        if self.original:
            if self.pos>=6900:return False,None
            a=source_frame(self.pos).copy();self.pos+=1;return True,a
        ok,a=self.cap.read();return ok,tagged(a,self.sr_density) if ok and self.sr_density!=1 else a
    def set(self,k,v):
        if k==C.CAP_PROP_POS_FRAMES:self.pos=int(v)
        if self.cap:return self.cap.set(k,v)
        return True
    def get(self,k):
        if self.cap:return self.cap.get(k)
        return 30 if k==C.CAP_PROP_FPS else 0
    def release(self):
        if self.cap:self.cap.release()

class PImage:
    def __init__(self,image,s=1):self.image=image;self.density=s
    @property
    def size(self):return tuple(round(x/self.density) for x in self.image.size)
    @property
    def width(self):return self.size[0]
    @property
    def height(self):return self.size[1]
    @property
    def mode(self):return self.image.mode
    def convert(self,*a,**kw):return PImage(self.image.convert(*a,**kw),self.density)
    def resize(self,size,*a,**kw):
        s=target_density(size)
        return PImage(self.image.resize(tuple(round(x*s) for x in size),*a,**kw),s)
    def save(self,p,*a,**kw):return self.image.save(safe_path(p),*a,**kw)
    def paste(self,im,box=None,*a,**kw):
        return self.image.paste(im.image if isinstance(im,PImage) else im,None if box is None else tuple(round(x*self.density) for x in box),*a,**kw)
    def __getattr__(self,k):return getattr(self.image,k)
class Images:
    Resampling=PI.Resampling
    def open(self,p):
        if is_reference(p):return PImage(PI.fromarray(raw(reference(p))),SCALE)
        return PImage(PI.open(record(p)),1)
    def new(self,mode,size,*a,**kw):
        s=target_density(size);return PImage(PI.new(mode,tuple(round(x*s) for x in size),*a,**kw),s)
    def fromarray(self,a,*args,**kw):return PImage(PI.fromarray(raw(a),*args,**kw),density(a))
Image=Images()
class Fonts:
    def truetype(self,p,size,*a,**kw):record(p);return (str(p),size,a,kw)
ImageFont=Fonts()
class Drawing:
    def __init__(self,im):self.draw=PD.Draw(im.image);self.s=im.density
    def text(self,xy,text,font=None,**kw):
        if isinstance(font,tuple):p,size,a,k=font;font=PF.truetype(p,round(size*self.s),*a,**k)
        return self.draw.text(tuple(x*self.s for x in xy),text,font=font,**kw)
    def __getattr__(self,k):return getattr(self.draw,k)
class Drawings:
    def Draw(self,im):return Drawing(im)
ImageDraw=Drawings()

def safe_path(p):
    p=Path(p).resolve()
    if WORK in p.parents or p==WORK:dest=p
    elif ROOT in p.parents:dest=WORK/'state'/JOB/p.relative_to(ROOT)
    else:raise RuntimeError('Refusing external output: '+str(p))
    dest.parent.mkdir(parents=True,exist_ok=True);return dest
def write_text(p,*a,**kw):return safe_path(p).write_text(*a,**kw)
def read_text(p,*a,**kw):return record(p).read_text(*a,**kw)
def mkdir(p,*a,**kw):return safe_path(Path(p)/'_directory_marker').parent.mkdir(parents=True,exist_ok=True)
