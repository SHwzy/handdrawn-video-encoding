"""Independent moving boundary pairs, closed loops, three overlapping scene passes."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CFG=json.loads((HERE/'mask_config_v3.json').read_text(encoding='utf-8'))
W,H=CFG['width'],CFG['height'];AH=CFG['active_rect'][3]
START=CFG['source_start'];N=CFG['source_end_exclusive']-START

def ease(x):
    x=np.clip(x,0.,1.);return x*x*(3-2*x)

class CurvePairMasks:
    def __init__(self,w=W,h=AH):
        self.w,self.h=w,h
        self.z=np.linspace(-.9,1.9,240)
        self.circle=np.linspace(0,2*np.pi,140)
        yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
        x=xx/w;y=yy/h
        self.maps={}
        for g in CFG['groups']:
            u=x.copy();v=y.copy()
            for cx,cy,rx,ry,twist in [(.28,.29,.3,.62,2.2),(.74,.72,.27,.55,-2.6)]:
                dx=(u-cx)/rx;dy=(v-cy)/ry
                a=twist*np.exp(-(dx*dx+dy*dy)*1.4)
                co=np.cos(a);si=np.sin(a)
                u=cx+rx*(dx*co-dy*si);v=cy+ry*(dx*si+dy*co)
            edge=np.sin(x*np.pi)*np.sin(y*np.pi)
            self.maps[g['id']]=(np.float32((x+(u-x)*edge)*w),np.float32((y+(v-y)*edge)*h))

    def pair(self,n,g,i):
        # All time is forward. Each boundary gets its own rotation, translation and pivot.
        local=(n-g['start']-i*g['stagger'])/(g['end']-g['start']-i*g['stagger'])
        q=float(ease(local))
        t=(n-g['start'])/30
        p=g['phase']+i*1.57
        center=-.1+i*1.2/(g['count']-1)
        z=self.z
        cx=center+.115*np.sin(z*4.6+p+.14*t)+.035*np.sin(z*8.0-p+.11*t)
        cy=z+g['vertical_speed']*t*(.72+.12*(i%3))
        pts=np.column_stack([cx*self.w,cy*self.h])
        # Unequal widths and different pivots make wedge-shaped openings and pinches.
        taper=.58+.28*np.sin(z*2.1+p)+.14*np.sin(z*5.0-p)
        fill_gain=1+2.1*q**4 if g['id']!='pattern' else 1
        breadth=q*g['half_width']*self.w*taper*fill_gain
        curves=[]
        for side in [-1,1]:
            curve=pts.copy()
            curve[:,0]+=side*breadth*(1+.1*side*np.sin(p+t*.4))
            curve[:,1]+=side*q*self.h*.055*np.sin(z*2.8+p+t*.35)
            angle=(g['spin']*np.sin(t*.28+p*.35)+side*q*(5.5+2.5*np.sin(p+t*.2)))*np.pi/180
            pivot=np.array([center*self.w,(.5+side*.1*q)*self.h])
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            curve=(curve-pivot)@rotation.T+pivot
            curves.append(curve)
        return q,curves

    def group(self,n,g,padding=0):
        mask=np.zeros((self.h+2*padding,self.w+2*padding),np.uint8)
        if n<=g['start']:return mask.astype(np.float32)
        for i in range(g['count']):
            q,(left,right)=self.pair(n,g,i)
            if q<.0002:continue
            polygon=np.rint(np.concatenate([left,right[::-1]])+padding).astype(np.int32)
            strip=np.zeros_like(mask)
            cv2.fillPoly(strip,[polygon],255,lineType=cv2.LINE_AA)
            mask=np.maximum(mask,strip)
        # A closed outer/inner boundary pair creates a real ring, not a painted circle.
        for i,(x,y,rx,ry) in enumerate(g['rings'][:1]):
            q=float(ease((n-g['start']-10-i*7)/(g['end']-g['start']-17)))
            if q<=0:continue
            t=(n-g['start'])/30
            x=x+.025*np.sin(.6*t+g['phase']+i)
            y=y+g['vertical_speed']*t
            angle=(g['spin']*t+20*i)*np.pi/180
            rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            radius_factor=.78+.22*float(ease(q*2))
            deform=1+.18*np.cos(self.circle-.7*t)+.10*np.sin(2*self.circle+i)
            outer=np.column_stack([np.cos(self.circle)*rx*self.w*radius_factor*deform,np.sin(self.circle)*ry*self.h*radius_factor*deform])
            inner=outer*(1-.77*q)
            inner[:,0]+=.035*self.w*q*np.sin(t*.8+i)
            inner[:,1]+=.045*self.h*q*np.cos(t*.65+i)
            center=np.array([x*self.w+padding,y*self.h+padding])
            ring=np.zeros_like(mask)
            cv2.fillPoly(ring,[np.rint(outer@rot.T+center).astype(np.int32)],255,lineType=cv2.LINE_AA)
            inner_mask=np.zeros_like(mask)
            cv2.fillPoly(inner_mask,[np.rint(inner@rot.T+center).astype(np.int32)],255,lineType=cv2.LINE_AA)
            ring=np.minimum(ring,255-inner_mask)
            # Open a core while the closed band is visible; it closes as the pass completes.
            core_gate=float(ease(q*4))*float(1-ease((q-.7)/.3))
            if core_gate>0:
                core=inner*.72*core_gate
                cv2.fillPoly(mask,[np.rint(core@rot.T+center).astype(np.int32)],0,lineType=cv2.LINE_AA)
            mask=np.maximum(mask,ring)
        mx,my=self.maps[g['id']]
        if padding:
            ey,ex=np.mgrid[0:self.h+2*padding,0:self.w+2*padding].astype(np.float32)
            ex[padding:padding+self.h,padding:padding+self.w]=mx+padding
            ey[padding:padding+self.h,padding:padding+self.w]=my+padding
            mx,my=ex,ey
        # Extend only a mathematical matte's boundary value, never source-image pixels.
        mask=cv2.remap(mask,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        return mask.astype(np.float32)/255

    def weights(self,n):
        white,pattern,back=[self.group(n,g) for g in CFG['groups']]
        a2=white*pattern*(1-back)
        a1=white*(1-pattern)*(1-back)
        a0=1-a1-a2
        return a0,a1,a2

    def frame(self,n,bw=False):
        a0,a1,a2=self.weights(n)
        if bw:active=np.repeat(np.uint8(a1*255+.5)[...,None],3,axis=2)
        else:active=np.uint8(np.stack([a1+a2,a1,a1],axis=2)*255+.5)
        out=np.zeros((H,W,3),np.uint8);out[53:487]=active
        return out

def contact(e):
    ids=[54,64,76,90,102,114,125,136,148,160,172,184,195,205,217]
    im=Image.new('RGB',(1280,4*207),'#222');d=ImageDraw.Draw(im)
    for i,n in enumerate(ids):
        x=i%4*320;y=i//4*207
        im.paste(Image.fromarray(e.frame(n)).resize((320,180)),(x,y))
        d.text((x+6,y+184),f'{(START+n)/30:.3f}s | local {n/30:.3f}s',fill='white')
    im.save(HERE/'contact_v3.jpg',quality=95)

def render(e):
    src=next(ROOT.glob('*.mp4'));cap=cv2.VideoCapture(str(src));cap.set(cv2.CAP_PROP_POS_FRAMES,START)
    outputs={}
    for name,width in [('mask_three_colors_v3',W),('source_vs_mask_v3',W*2),('mask_white_weight_v3',W)]:
        cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{H}','-r','30','-i','pipe:0','-ss',str(START/30),'-i',str(src),'-map','0:v:0','-map','1:a:0','-t',str(N/30),'-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(HERE/'output'/(name+'.mp4'))]
        outputs[name]=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    font=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msyh.ttc'),17)
    audit=[];prev=None
    for n in range(N):
        ok,bgr=cap.read()
        if not ok:raise RuntimeError('source missing')
        rgb=e.frame(n);a=e.weights(n)
        assert np.max(np.abs(sum(a)-1))<1e-5 and min(float(x.min()) for x in a)>-1e-6
        audit.append(dict(frame=START+n,coverage=[float(x.mean()) for x in a],delta=0 if prev is None else float(np.abs(rgb.astype(float)-prev.astype(float)).mean()/255)))
        prev=rgb
        outputs['mask_three_colors_v3'].stdin.write(rgb.tobytes())
        outputs['mask_white_weight_v3'].stdin.write(e.frame(n,True).tobytes())
        original=cv2.cvtColor(cv2.resize(bgr,(W,H)),cv2.COLOR_BGR2RGB)
        comp=Image.fromarray(np.concatenate([original,rgb],axis=1));d=ImageDraw.Draw(comp)
        d.text((18,15),'原片 · 37.867–45.133 秒',font=font,fill='white')
        d.text((978,15),'v3 · 多条曲线夹缝 / 纵向漂移 / 独立旋转 / 环形 / 三层叠加',font=font,fill='white')
        d.text((978,506),f'原片 {(START+n)/30:.3f}s · 黑＝现实 / 白＝第二态 / 红＝第三态',font=font,fill='white')
        outputs['source_vs_mask_v3'].stdin.write(np.asarray(comp).tobytes())
    cap.release()
    for p in outputs.values():
        p.stdin.close()
        if p.wait()!=0:raise RuntimeError('ffmpeg failure')
    (HERE/'audit_v3.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('v3 exported',N,'frames')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');a=ap.parse_args()
    engine=CurvePairMasks();contact(engine)
    if a.render:render(engine)
