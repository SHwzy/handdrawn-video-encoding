"""Final approved timeline -> resolution-aware original-asset renderer."""
from pathlib import Path
import argparse,json,time,subprocess,sys,hashlib
import numpy as N
import cv2 as C
import runtime as rt
from loader import ModuleLoader

BOUNDS=[0,1800,2664,3540,4410,5460,6077,6900]
class Film:
    def __init__(self):self.loader=ModuleLoader();self.modules={};self.scenes={};self.man=None
    def module(self,key,path):
        if key not in self.modules:self.modules[key]=self.loader.load(rt.ROOT/path)
        return self.modules[key]
    def rgb(self,n):return rt.cv.cvtColor(rt.source_frame(n),C.COLOR_BGR2RGB)
    def mannequin(self,n):
        if self.man is None:
            m=self.module('man','headswap_hybrid_30_60/mannequin/render.py')
            meta=json.loads((rt.ROOT/'headswap_hybrid_30_60/mannequin/output/render_metadata_v4_parts.json').read_text())
            settings=meta['mask']['settings'];settings={k:v for k,v in settings.items() if k in {'seed','rotation_deg','mirror_x','mirror_y','variation'}}
            engine=m.PartwiseMasks(960,434,**settings);engine.groups[1]['half_width']*=meta['pattern_width_multiplier']
            plates=[]
            for name in ['reality','white','pattern']:
                im=rt.Image.open(rt.ROOT/'headswap_hybrid_30_60/mannequin/assets'/f'{name}_v1.png').convert('RGB')
                plates.append(rt.np.array(im.resize((round(im.width*434/im.height),434),rt.Image.Resampling.LANCZOS)))
            glyph=rt.Image.new('L',(960,434));d=rt.ImageDraw.Draw(glyph);font=rt.ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc'),12)
            for x,y,line in [(908,87,'従いたい心根を吐き出さぬように込めて'),(884,133,'胸の中が澱のように濁る')]:
                for j,ch in enumerate(line):d.text((x,y+12*j),ch,font=font,fill=255)
            self.man=(engine,plates,rt.np.asarray(glyph)[...,None].astype('float32')/255)
        engine,plates,glyph=self.man
        y,x=rt.grid((slice(0,434),slice(0,960))).astype('float32');pw=plates[0].shape[1];offset=pw-960-88.6*(n-1136)/217
        crops=[rt.cv.remap(p,x+offset,y,C.INTER_LINEAR,borderMode=C.BORDER_CONSTANT) for p in plates]
        weights=engine.weights(n-1136);active=sum(p.astype('float32')*q[...,None] for p,q in zip(crops,weights))
        active=active*(1-glyph)+(weights[0]+weights[2])[...,None]*255*glyph
        m=self.modules['a2'];return m.active_canvas(active)
    def frame(self,n):
        if n<900:
            m=self.module('a1','headswap_hybrid_30s/render.py')
            return rt.cv.cvtColor(m.render_frame(n,rt.source_frame(n).copy()),C.COLOR_BGR2RGB)
        if n<1800:
            m=self.module('a2','headswap_hybrid_30_60/render.py');s=next(s for s in m.CFG['shots'] if s['start']<=n<s['end']);id=s['id'];k=n-s['start']
            if id=='A':
                im=rt.cv.resize(m.load('../headswap_hybrid_30s/assets/two_people.png'),(960,540))[53:487]
                z=s['zoom_at_899']+(n-899)*s['zoom_velocity_per_frame'];return m.active_canvas(rt.cv.remap(im,(m.XX-480)/z+480,(m.YY-217)/z+217,C.INTER_LINEAR))
            if id=='T':return self.rgb(n)
            if id=='C':return self.mannequin(n)
            if id not in self.scenes:
                self.scenes[id]=m.ScenePlates(s) if s['mode']=='plates' else self.scenes.setdefault('finale',m.Finale([s for s in m.CFG['shots'] if s['id'] in 'FGHI']))
            return self.scenes[id].frame(k) if s['mode']=='plates' else self.scenes[id].frame(id,k)
        if n<2664:
            if n<1941:
                m=self.module('bopen','rebuild_60_90/opening_00_05/render_v3.py');a=m.frame(n)[0]
            elif n<1973:
                m=self.module('bseq','rebuild_60_90/segment_05_21/render_sequence_v5.py');a=m.rebuilt(n)
            elif 2034<=n<2102:
                m=self.module('bstand','rebuild_60_90/standing_fix/render.py');a=m.frame(n)
            elif n<2200:
                m=self.module('bbase','rebuild_60_90/render.py');a=m.frame(n,rt.source_frame(n).copy())[0]
            elif n<2443:
                m=self.module('bseq','rebuild_60_90/segment_05_21/render_sequence_v5.py');a=m.rebuilt(n)
            else:
                m=self.module('bend','rebuild_60_90/segment_21_29/render_segment_v4.py');a=m.frame(n)
            return rt.cv.cvtColor(a,C.COLOR_BGR2RGB)
        if n<3540:
            m=self.module('c','rebuild_88_118/render_v3.py')
            return m.base.frame(n) if 3061<=n<3073 or 3101<=n<3506 else self.rgb(n)
        if n<4410:return self.module('d','rebuild_118_147/render_v5.py').frame(n)
        if n<5460:
            m=self.module('e','rebuild_147_182/render_v5.py');return m.v.frame(n,rt.Capture(rt.SOURCE))
        if n<6077:
            rt.PROVIDERS[str((rt.ROOT/'rebuild_147_182/output/approved_147_182_v5.mp4').resolve())]=lambda k:rt.cv.cvtColor(self.frame(4410+k),C.COLOR_RGB2BGR)
            return self.module('f','rebuild_182_202_agent_20261002_a/render_v4.py').frame(n)
        return self.module('g','rebuild_202_end_agent_20261002_b/render_v3.py').render(n,self.rgb(n))
    def save_manifest(self):
        self.loader.save_log();records=[]
        for p,size in sorted(rt.READS.items()):
            path=Path(p);h=hashlib.sha256()
            with path.open('rb') as f:
                while b:=f.read(8*1024*1024):h.update(b)
            records.append({'file':p,'bytes':size,'sha256':h.hexdigest()})
        dest=rt.WORK/'logs'/f'{rt.JOB}_assets.json';dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps({'scope':'Actual loaded dependencies; accumulates with executed branch coverage','super_resolution':'adopted_h3_replacements' if rt.SR_MAP else 'original_media','files':records},ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=6900);ap.add_argument('--scale',type=int,default=2);ap.add_argument('--samples',nargs='*',type=int);ap.add_argument('--name',default='candidate_1080p');ap.add_argument('--lossless',action='store_true');a=ap.parse_args();rt.SCALE=a.scale;rt.JOB=a.name
    C.setNumThreads(4)
    film=Film();o=rt.WORK/'output';q=rt.WORK/'qa';o.mkdir(exist_ok=True);q.mkdir(exist_ok=True)
    if a.samples:
        for n in a.samples:
            t=time.time();im=film.frame(n);physical=rt.raw(im);assert physical.shape==(540*a.scale,960*a.scale,3),(n,im.shape,physical.shape,rt.density(im))
            ok,buf=C.imencode('.png',C.cvtColor(physical,C.COLOR_RGB2BGR));assert ok;buf.tofile(q/f'f{n}_s{a.scale}.png');print(json.dumps({'frame':n,'seconds':round(time.time()-t,3),'shape':physical.shape}),flush=True)
        film.save_manifest();return
    final=o/(a.name+'.mp4');dest=o/(a.name+'.partial.mp4');assert not final.exists() and not dest.exists()
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{960*a.scale}x{540*a.scale}','-r','30','-i','-','-ss',str(a.start/30),'-i',str(rt.SOURCE),'-map','0:v','-map','1:a','-t',str((a.end-a.start)/30),'-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(dest)]
    if a.lossless:
        i=cmd.index('-crf');cmd[i:i+2]=['-qp','0']
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE);t=time.time()
    try:
        for n in range(a.start,a.end):
            im=rt.raw(film.frame(n));assert im.shape==(540*a.scale,960*a.scale,3) and N.isfinite(im).all(),(n,im.shape)
            p.stdin.write(im.astype('uint8').tobytes())
            if n%30==0:print(json.dumps({'frame':n,'elapsed':round(time.time()-t,1)}),flush=True)
    finally:p.stdin.close();code=p.wait();film.save_manifest()
    assert code==0;dest.rename(final)
if __name__=='__main__':main()
