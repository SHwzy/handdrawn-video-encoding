"""Hybrid decoder. Original-media ranges are declared, never used to patch holes.
All animation is a rigid camera/layer transform, alpha compositing, sparse pose
selection, or an explicit effect. There is no optical flow or deforming mesh.
"""
from pathlib import Path
import importlib.util,json,subprocess,sys
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
cfg=json.loads((R/'encoding.json').read_text(encoding='utf-8'))
effect_tracks=json.loads((R/'effect_tracks.json').read_text(encoding='utf-8'))['tracks']
modules={}
for k,p in cfg['modules'].items():
    spec=importlib.util.spec_from_file_location('hybrid_'+k,R/p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);modules[k]=m
images={}
def read(path):
    if path not in images:
        im=cv2.imdecode(np.fromfile(R/path,dtype=np.uint8),cv2.IMREAD_COLOR)
        images[path]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)[53:487].copy()
    return images[path]

def camera(path,params):
    im=read(path);z,px,py=params
    ww,hh=960/z,434/z;cx=480+px;cy=217+py
    assert cx-ww/2>=-.01 and cx+ww/2<=960.01 and cy-hh/2>=-.01 and cy+hh/2<=434.01,(path,params)
    xs=np.linspace(cx-ww/2,cx+ww/2-1,960,dtype=np.float32)
    ys=np.linspace(cy-hh/2,cy+hh/2-1,434,dtype=np.float32)
    mx,my=np.meshgrid(xs,ys)
    active=cv2.remap(im,mx,my,cv2.INTER_LINEAR)
    out=np.zeros((540,960,3),np.uint8);out[53:487]=active;return out

text_cache={}
font=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc'),12)
def lyric(im,text,color):
    key=(text,color)
    if key not in text_cache:
        layer=Image.new('RGBA',(960,540));d=ImageDraw.Draw(layer)
        for i,c in enumerate(text):d.text((910,190+i*13),c,font=font,fill=color+(255,))
        text_cache[key]=np.asarray(layer)
    a=text_cache[key][:,:,3:4]/255.0;bgr=text_cache[key][:,:,:3][:,:,::-1]
    return np.uint8(im*(1-a)+bgr*a)

rng=np.random.default_rng(7)
petals=[(float(rng.uniform(0,960)),float(rng.uniform(70,440)),float(rng.uniform(-1,1)),float(rng.uniform(1.5,4)),float(rng.uniform(0,180))) for i in range(24)]
def render_frame(n,original):
    s=next(s for s in cfg['shots'] if s['start']<=n<s['end']);q=(n-s['start'])/max(1,s['end']-s['start']-1);kind=s['kind']
    if kind=='original':return original.copy()
    lyric_text=None
    if kind=='running':out=modules['running'].render(n-s['start'],s['variant']);lyric_text='そう簡単な祈りだった　端から'
    elif kind=='seedance_running':out=modules['seedance_running'].render(n-s['start'],s['variant']);lyric_text='そう簡単な祈りだった　端から'
    elif kind=='side':out=modules['side'].render(n-s['start']);lyric_text='段々と消える感嘆'
    elif kind=='crowd_panorama':out=modules['crowd'].render(n);lyric_text='今から緞帳が上がるから'
    elif kind=='stage':out=modules['stage'].render(n);lyric_text='静かな会場を後にさよなら'
    elif kind=='cathedral':out=modules['cathedral'].render(n)
    elif kind=='eye_motion':out=modules['cathedral'].render_eye(n)
    else:
        params=np.array(s['begin'])*(1-q)+np.array(s['finish'])*q
        if kind=='pose_keys':
            path=[k['asset'] for k in s['keys'] if n>=k['start']][-1]
        else:path=s['asset']
        out=camera(path,params)
        if kind=='pose_keys' and n>=s['particles']['start']:
            dt=n-s['particles']['start']
            for i,(x,y,vx,vy,angle) in enumerate(petals):
                if i>dt*3:continue
                cv2.ellipse(out,(round(x+vx*dt),round(y+vy*dt)),(2,5),angle+dt*5,0,360,(80,5,220),-1,cv2.LINE_AA)
        if 'focus' in s:
            sigma=s['focus'][0]*(1-q)+s['focus'][1]*q
            if sigma>.1:out[53:487]=cv2.GaussianBlur(out[53:487],(0,0),sigma)
        if n-s['start']<s.get('pixel_entry_frames',0):out[53:487]=cv2.resize(cv2.resize(out[53:487],(96,44)),(960,434),interpolation=cv2.INTER_NEAREST)
        if n==s.get('flash_frame'):
            out[53:487]=cv2.addWeighted(out[53:487],.2,np.full_like(out[53:487],255),.8,0)
    for effect in effect_tracks:
        if effect['start']<=n<effect['end']:
            dx,dy=effect['shake_xy'][n-effect['start']]
            active=cv2.warpAffine(out[53:487],np.float32([[1,0,dx],[0,1,dy]]),(960,434),
                                  flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            keys=effect['focus']
            sigma=float(np.interp(n,[k['frame'] for k in keys],[k['sigma_px'] for k in keys]))
            if sigma>.05:active=cv2.GaussianBlur(active,(0,0),sigma)
            out[53:487]=active
            break
    if lyric_text:out=lyric(out,lyric_text,(245,245,245))
    assert out.shape==(540,960,3) and out.dtype==np.uint8,(n,out.shape,out.dtype)
    out[:53]=0;out[487:]=0
    return out

def main():
    O=R/'output';O.mkdir(exist_ok=True);E=R/'evidence_baseline_v11';E.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(R/cfg['original_media']))
    outpath=O/'混合编码_前30秒_基线_v11.mp4'
    cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','960x540','-r','30','-i','pipe:0','-i',str(R/cfg['audio']),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t','30','-movflags','+faststart',str(outpath)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    ids={71,80,90,97,154,167,181,235,248,261,262,289,290,303,304,317,318,330,331,342,343,371,372,385,399,400,413,426,427,432,440,449,453,467,468,481,482,486,489,540,600,630,660,697,698,704,720,743,750,770,807,808,814,822,832,840,862,863,867,877,892,899}
    checks=[];prev=None
    for n in range(900):
        ok,original=cap.read()
        if not ok:raise RuntimeError('original-media frame missing '+str(n))
        out=render_frame(n,original)
        checks.append(dict(frame=n,mean_delta=0 if prev is None else float(np.abs(out.astype(float)-prev.astype(float)).mean()),direct_original=bool(np.array_equal(out,original))))
        proc.stdin.write(out.tobytes());prev=out
        if n in ids:cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(E/f'f{n:04d}.jpg')
        if n%90==0:print(n,flush=True)
    proc.stdin.close();cap.release()
    if proc.wait()!=0:raise RuntimeError('encoding failure')
    (E/'render_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(R/cfg['original_media']),'-i',str(outpath),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-map','1:a:0','-t','30','-c:v','libx264','-crf','18','-preset','fast','-c:a','copy','-movflags','+faststart',str(O/'原PV左_基线右_前30秒_v11.mp4')],check=True)
    from PIL import ImageDraw
    files=sorted(E.glob('f*.jpg'));w=1280;h=((len(files)+3)//4)*205
    contact=Image.new('RGB',(w,h),'#222222');d=ImageDraw.Draw(contact)
    for i,p in enumerate(files):
        x=i%4*320;y=i//4*205;contact.paste(Image.open(p).resize((320,180)),(x,y));d.text((x+4,y+183),p.stem,fill='white')
    contact.save(E/'contact.jpg')
    print('DONE',flush=True)

if __name__=='__main__':main()
