from pathlib import Path
import sys,json,argparse,subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/segment_30_60/mask_lab'))
from seeded_masks import SeededMasks
from partwise_masks import PartwiseMasks
W,H,AH,START,N=960,540,434,1136,218

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=20260930)
    ap.add_argument('--rotation-deg',type=float,default=0)
    ap.add_argument('--mirror-x',action='store_true')
    ap.add_argument('--mirror-y',action='store_true')
    ap.add_argument('--variation',type=float,default=.35)
    ap.add_argument('--pattern-width',type=float,default=1.7)
    ap.add_argument('--sway',action='store_true',help='Small animated rotation/scale on mask groups only; writes v2_sway outputs')
    ap.add_argument('--sway-strong',action='store_true',help='15 degree / 0.8-1.2 scale mask motion with procedural overscan')
    ap.add_argument('--parts-flow',action='store_true',help='Independent one-way rotation/scale for each ribbon and ring')
    a=ap.parse_args();settings=vars(a).copy();settings.pop('pattern_width')
    settings.pop('sway_strong')
    settings.pop('parts_flow')
    if a.sway_strong:a.sway=True;settings['sway']='strong'
    version='v3_sway' if a.sway_strong else ('v2_sway' if a.sway else 'v1')
    if a.parts_flow:version='v4_parts';settings['sway']=False
    shot_name=f'shot_{version}';comparison_name=f'source_vs_shot_{version}'
    engine=(PartwiseMasks if a.parts_flow else SeededMasks)(W,AH,**settings)
    engine.groups[1]['half_width']*=a.pattern_width
    plates=[];sizes=[]
    for name in ['reality','white','pattern']:
        im=Image.open(HERE/'assets'/f'{name}_v1.png').convert('RGB');sizes.append(im.size)
        width=round(im.width*AH/im.height)
        plates.append(np.array(im.resize((width,AH),Image.Resampling.LANCZOS)))
    assert len(set(sizes))==1,'Scene plates must share canvas'
    pw=plates[0].shape[1];shift=88.6
    assert pw-W>=shift,'Camera overscan insufficient'
    src=next(ROOT.glob('*.mp4'));cap=cv2.VideoCapture(str(src));cap.set(cv2.CAP_PROP_POS_FRAMES,START)
    font=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msyh.ttc'),17)
    lyricfont=ImageFont.truetype(__import__("builtins").portable_text('C:/Windows/Fonts/msgothic.ttc'),12)
    glyph=Image.new('L',(W,AH));d=ImageDraw.Draw(glyph)
    for x,y,line in [(908,87,'従いたい心根を吐き出さぬように込めて'),(884,133,'胸の中が澱のように濁る')]:
        for j,c in enumerate(line):d.text((x,y+12*j),c,font=lyricfont,fill=255)
    glyph=np.asarray(glyph)[...,None].astype(np.float32)/255
    writers={}
    for name,width in [(shot_name,W),(comparison_name,2*W)]:
        cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{H}','-r','30','-i','pipe:0','-ss',str(START/30),'-i',str(src),'-map','0:v','-map','1:a','-t',str(N/30),'-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(HERE/'output'/f'{name}.mp4')]
        writers[name]=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    contact=Image.new('RGB',(1280,4*200),'#17171c');cd=ImageDraw.Draw(contact)
    ids=[0,54,70,86,102,114,126,138,150,162,176,190,205,217]
    audit=[]
    yy,xx=np.mgrid[0:AH,0:W].astype(np.float32)
    for n in range(N):
        # Common camera motion for all scene states; artwork is never mirrored/rotated.
        offset=pw-W-shift*n/(N-1)
        crops=[cv2.remap(p,xx+offset,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for p in plates]
        weights=engine.weights(n)
        assert np.max(np.abs(sum(weights)-1))<1e-5
        active=sum(p.astype(np.float32)*q[...,None] for p,q in zip(crops,weights))
        textcolour=(weights[0]+weights[2])[...,None]*255
        active=active*(1-glyph)+textcolour*glyph
        out=np.zeros((H,W,3),np.uint8);out[53:487]=np.clip(active+.5,0,255).astype(np.uint8)
        ok,bgr=cap.read();assert ok
        original=cv2.cvtColor(cv2.resize(bgr,(W,H)),cv2.COLOR_BGR2RGB)
        comp=Image.fromarray(np.concatenate([original,out],1));d=ImageDraw.Draw(comp)
        d.text((18,15),'原片',font=font,fill='white');d.text((978,15),'爱音 · 三套实际画面 + 烟雾遮罩',font=font,fill='white')
        d.text((978,506),f'{(START+n)/30:.3f}s · seed {a.seed}',font=font,fill='white')
        writers[shot_name].stdin.write(out.tobytes());writers[comparison_name].stdin.write(np.asarray(comp).tobytes())
        if n in ids:
            i=ids.index(n);x=i%4*320;y=i//4*200
            contact.paste(Image.fromarray(out).resize((320,180)),(x,y));cd.text((x+5,y+182),f'{(START+n)/30:.3f}s',fill='white')
        audit.append(dict(frame=START+n,camera_x=float(offset),weights=[float(q.mean()) for q in weights],**({'sway':{g['id']:engine.sway_parameters(n,g) for g in engine.groups}} if a.sway else {})))
        if a.parts_flow:audit[-1]['parts']={g['id']:[engine.part_parameters(n,g,i) for i in range(g['count']+1)] for g in engine.groups}
    cap.release()
    for p in writers.values():
        p.stdin.close();assert p.wait()==0
    contact.save(HERE/'output'/f'contact_{version}.jpg',quality=95)
    metadata=dict(source_start=START,source_end_exclusive=START+N,frames=N,fps=30,asset_sizes=sizes,camera_shift_px=shift,pattern_width_multiplier=a.pattern_width,mask=engine.metadata(),audit=audit)
    (HERE/'output'/(f'render_metadata_{version}.json' if a.sway or a.parts_flow else 'render_metadata.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.sway or a.parts_flow:
        print('Exported sway trial: 218 frames, 30 fps; accepted v1 outputs preserved')
        return
    # Same instant, deliberately different stable shot settings, to demonstrate available variations.
    variants=[dict(seed=20260930,rotation_deg=0),dict(seed=73,rotation_deg=12,mirror_x=True),dict(seed=891,rotation_deg=-14,mirror_y=True)]
    vi=Image.new('RGB',(960,3*205),'#17171c');vd=ImageDraw.Draw(vi)
    for i,settings in enumerate(variants):
        e=SeededMasks(W,AH,**settings)
        for j,n in enumerate([90,125,170]):
            q=e.weights(n);rgb=np.uint8(np.stack([q[1]+q[2],q[1],q[1]],2)*255)
            vi.paste(Image.fromarray(rgb).resize((320,180)),(j*320,i*205))
        vd.text((5,i*205+183),str(settings),fill='white')
    vi.save(HERE/'output/mask_variants.jpg',quality=95)
    print('Exported 218 frames, 30 fps, three registered scene states')

if __name__=='__main__':main()
