"""30–60s scene-plate composition, accepted C clip, parallax and portrait/dance tracks."""
from pathlib import Path
import argparse,json,subprocess,sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'analysis/segment_30_60/mask_lab'))
from seeded_masks import SeededMasks
CFG=json.loads((HERE/'encoding_v2.json').read_text(encoding='utf-8'))
W,H,AH=960,540,434
XX,YY=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(AH,dtype=np.float32))
FONT=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msyh.ttc'),17)
JP=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc'),12)

def load(path,rgba=False):
    return np.asarray(Image.open(HERE/path).convert('RGBA' if rgba else 'RGB')).copy()

def active_canvas(im):
    out=np.zeros((H,W,3),np.uint8);out[53:487]=np.clip(im+.5,0,255).astype(np.uint8);return out

def fit(im,width=W,height=AH):
    # Fit by uniform scaling, then central crop. No copied artwork borders.
    scale=max(width/im.shape[1],height/im.shape[0]);sw=int(np.ceil(im.shape[1]*scale));sh=int(np.ceil(im.shape[0]*scale))
    p=cv2.resize(im,(sw,sh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    x=(sw-width)//2;y=(sh-height)//2;return p[y:y+height,x:x+width].copy()

def glyph(lines):
    layer=Image.new('L',(W,AH));d=ImageDraw.Draw(layer)
    for j,line in enumerate(lines):
        x=908-24*j;y=max(22,(AH-len(line)*12)//2)
        for i,c in enumerate(line):d.text((x,y+i*12),c,font=JP,fill=255)
    return np.asarray(layer)[...,None].astype(np.float32)/255

def caption(im,g,colour):return im*(1-g)+np.asarray(colour)*g

class ScenePlates:
    def __init__(self,s):
        self.s=s;dx=s['shift_x'];self.pw=W+int(np.ceil(abs(dx)))+8
        self.plates=[fit(load(f"{s['directory']}/assets/{state}_v1.png"),self.pw,AH) for state in ['reality','white','pattern']]
        self.engine=SeededMasks(W,AH,**{k:s[k] for k in ['seed','rotation_deg','mirror_x','mirror_y','variation']})
        for g,times in zip(self.engine.groups,s['passes']):g['start'],g['end']=times
        self.engine.groups[1]['half_width']*=s['pattern_width']
        self.text=glyph(s['lyrics']);self.audit=[]
    def frame(self,n):
        q=n/(self.s['end']-self.s['start']-1);dx=self.s['shift_x']
        offset=4-dx*q if dx<0 else 4+dx*(1-q)
        assert offset>=0 and offset+W<=self.pw+1e-5
        crops=[cv2.remap(p,XX+offset,YY,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for p in self.plates]
        weights=self.engine.weights(n);assert np.max(np.abs(sum(weights)-1))<1e-5
        out=sum(p.astype(np.float32)*a[...,None] for p,a in zip(crops,weights))
        # Screen-space lyric colours travel through the same scene-state matte.
        colour=(weights[0]+weights[2])[...,None]*245
        if self.s['id']=='E':colour=30+(245-30)*weights[2][...,None]
        out=caption(out,self.text,colour)
        self.audit.append(dict(local_frame=n,camera_x=float(offset),coverage=[float(a.mean()) for a in weights]))
        return active_canvas(out)
    def metadata(self):return dict(settings=self.s,mask=self.engine.metadata(),audit=self.audit)

def premult_layer(path,width=None,height=None):
    p=load(path,True).astype(np.float32)/255
    assert np.min(p[:,:,3])==0 and np.max(p[:,:,3])>0,'Expected actual alpha'
    alpha=p[:,:,3:4];p=np.concatenate([p[:,:,:3]*alpha,alpha],2)
    if width is not None:p=cv2.resize(p,(width,round(p.shape[0]*width/p.shape[1])),interpolation=cv2.INTER_AREA)
    if height is not None:p=cv2.resize(p,(round(p.shape[1]*height/p.shape[0]),height),interpolation=cv2.INTER_AREA)
    return p

def over(bg,p,mat):
    fg=cv2.warpAffine(p,mat,(W,AH),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    return bg*(1-fg[:,:,3:4])+fg[:,:,:3]*255

def star(im,x,y,size,strength):
    layer=np.zeros_like(im,dtype=np.float32)
    pts=np.array([(x,y-size),(x+size*.12,y-size*.12),(x+size,y),(x+size*.12,y+size*.12),(x,y+size),(x-size*.12,y+size*.12),(x-size,y),(x-size*.12,y-size*.12)],np.int32)
    cv2.fillPoly(layer,[pts],(255,255,255),lineType=cv2.LINE_AA)
    return np.maximum(im,layer*strength)

def grid(n):
    # Independent screen-space fine crimson geometric line texture.
    im=np.full((AH,W,3),(242,237,231),np.uint8)
    bend=28*(1-n/12)
    z=np.linspace(-80,AH+80,180)
    for x in range(-AH,W+AH,64):
        for side in [-1,1]:
            px=x+side*z+ bend*np.sin(z/75+x/120)
            py=z+bend*.45*np.sin(px/130+side)
            cv2.polylines(im,[np.rint(np.column_stack([px,py])).astype(np.int32)],False,(192,79,103),1,cv2.LINE_AA)
    # Sparse halftone screen follows source's print texture, not a moving particle field.
    im[::3,::3]=np.uint8(im[::3,::3].astype(float)*.94)
    return im.astype(np.float32)

class Finale:
    def __init__(self,shots):
        self.shots={s['id']:s for s in shots}
        self.joined=fit(load(self.shots['F']['asset']),W+6,AH)
        self.room=fit(load(self.shots['G']['background']),W,AH+44)
        self.guests=premult_layer(self.shots['G']['foreground'],width=W)
        self.seated_pair=premult_layer(self.shots['G']['pair'],width=220)
        self.portrait=fit(load(self.shots['H']['asset']))
        self.pair=premult_layer(self.shots['I']['pair'],width=900)
        self.caption={k:glyph(s['lyrics']) for k,s in self.shots.items() if 'lyrics' in s}
    def frame(self,id,n):
        s=self.shots[id];q=n/max(1,s['end']-s['start']-1)
        if id=='F':
            out=cv2.remap(self.joined,XX+6*q,YY,cv2.INTER_LINEAR)
            out=caption(out,self.caption[id],(30,30,35))
        elif id=='G':
            bg=cv2.remap(self.room,XX,YY+s['background_up_px']*q,cv2.INTER_LINEAR)
            # This seated/kneeling pose differs from the upright pair in shot I.
            bg=over(bg,self.seated_pair,np.float32([[1,0,375],[0,1,270-s['pair_up_px']*q]]))
            bg=cv2.GaussianBlur(bg,(0,0),s['focus_sigma_px']*(1-q)+.35*q)
            # Guests stay sharp and rise faster than the scene behind them.
            guests=cv2.GaussianBlur(self.guests,(0,0),.15+2*q)
            assert guests.shape[0]+110-s['foreground_up_px']*q>=AH,'Foreground lower coverage'
            out=over(bg,guests,np.float32([[1,0,0],[0,1,110-s['foreground_up_px']*q]]))
        elif id=='H':
            z=s['zoom_start']*(1-q)+s['zoom_end']*q
            out=cv2.remap(self.portrait,(XX-480)/z+480,(YY-217)/z+217,cv2.INTER_LINEAR)
            sigma=s['focus_start_px']*(1-q)**2
            if sigma>.05:out=cv2.GaussianBlur(out,(0,0),sigma)
            if n<5:
                echo=cv2.warpAffine(out,np.float32([[1,0,4*(1-q)],[0,1,0]]),(W,AH))
                out=out*.82+echo*.18
            for i,(x,y) in enumerate([(120,230),(230,104),(730,104),(831,230),(250,350),(710,350)]):
                strength=.5+.5*np.sin(.28*n+i*.8)**2
                out=star(out,x,y,14 if i<4 else 11,strength)
        elif id=='I':
            z=s['zoom_start']*(1-q)+s['zoom_end']*q
            angle=s['rotation_start_deg']*(1-q)+s['rotation_end_deg']*q
            base=np.float32([[1,0,30],[0,1,-45]])
            turn=cv2.getRotationMatrix2D((480,217),angle,z)
            full=(np.vstack([turn,[0,0,1]])@np.vstack([base,[0,0,1]]))[:2]
            out=over(grid(n),self.pair,full)
            out=caption(out,self.caption[id],(30,25,30))
        return active_canvas(out)

def writer(path,width,source,start,duration):
    cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{H}','-r','30','-i','pipe:0','-ss',str(start),'-i',str(source),'-map','0:v:0','-map','1:a:0','-t',str(duration),'-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(path)]
    return subprocess.Popen(cmd,stdin=subprocess.PIPE)

def render(preview_only=False):
    source=next(ROOT.glob('*.mp4'));cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,900)
    accepted=cv2.VideoCapture(str(HERE/CFG['approved_shot_C']))
    scenes={s['id']:ScenePlates(s) for s in CFG['shots'] if s['mode']=='plates'}
    finale=Finale([s for s in CFG['shots'] if s['id'] in 'FGHI'])
    continuation=cv2.resize(load('../headswap_hybrid_30s/assets/two_people.png'),(W,H))[53:487]
    O=HERE/'output';E=HERE/'evidence_v2';O.mkdir(exist_ok=True);E.mkdir(exist_ok=True)
    writers={} if preview_only else {name:writer(O/(name+'.mp4'),width,source,30,30) for name,width in [('segment_30_60_v2',W),('source_vs_segment_30_60_v2',W*2)]}
    samples=set(range(900,1800,30))|{900,904,905,917,918,1135,1136,1353,1354,1571,1572,1735,1736,1750,1751,1760,1775,1776,1783,1789,1790,1795,1799}
    all_samples=[];audit=[];previous=None
    for frame in range(900,1800):
        ok,bgr=cap.read();assert ok
        original=cv2.cvtColor(cv2.resize(bgr,(W,H)),cv2.COLOR_BGR2RGB)
        s=next(s for s in CFG['shots'] if s['start']<=frame<s['end']);n=frame-s['start'];id=s['id']
        if id=='A':
            z=s['zoom_at_899']+(frame-899)*s['zoom_velocity_per_frame']
            out=active_canvas(cv2.remap(continuation,(XX-480)/z+480,(YY-217)/z+217,cv2.INTER_LINEAR))
        elif id=='T':out=original.copy()
        elif id=='C':
            ok,accepted_bgr=accepted.read();assert ok
            out=cv2.cvtColor(accepted_bgr,cv2.COLOR_BGR2RGB)
        elif id in scenes:out=scenes[id].frame(n)
        else:out=finale.frame(id,n)
        assert out.shape==(H,W,3) and out.dtype==np.uint8
        out[:53]=0;out[487:]=0
        delta=0 if previous is None else float(np.abs(out.astype(float)-previous.astype(float)).mean())
        audit.append(dict(source_frame=frame,shot=id,local_frame=n,delta=delta,source_reused=id=='T',approved_clip_reused=id=='C'))
        previous=out
        if not preview_only:
            writers['segment_30_60_v2'].stdin.write(out.tobytes())
            compare=Image.fromarray(np.concatenate([original,out],axis=1));d=ImageDraw.Draw(compare)
            d.text((18,15),'原片 · 30–60 秒',font=FONT,fill='white');d.text((978,15),f'重制 v2 · 镜头 {id}',font=FONT,fill='white')
            d.text((978,506),f'原片 {frame/30:.3f}s',font=FONT,fill='white')
            writers['source_vs_segment_30_60_v2'].stdin.write(np.asarray(compare).tobytes())
        if frame in samples:
            Image.fromarray(out).save(E/f'f{frame:04d}.jpg',quality=95);all_samples.append((frame,out.copy()))
        if frame%90==0:print(frame,flush=True)
    cap.release();accepted.release()
    for p in writers.values():p.stdin.close();assert p.wait()==0
    sheet=Image.new('RGB',(1280,int(np.ceil(len(all_samples)/4))*205),'#17171c');d=ImageDraw.Draw(sheet)
    for i,(frame,pixels) in enumerate(all_samples):
        x=i%4*320;y=i//4*205;sheet.paste(Image.fromarray(pixels).resize((320,180)),(x,y));d.text((x+5,y+183),f'f{frame} | {frame/30:.3f}s',fill='white')
    sheet.save(E/'contact_v2.jpg',quality=95)
    (E/'render_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (E/'scene_parameters.json').write_text(json.dumps({id:s.metadata() for id,s in scenes.items()},ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE 900 frames / 30 fps / 30 seconds',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preview-only',action='store_true');a=ap.parse_args();render(a.preview_only)
