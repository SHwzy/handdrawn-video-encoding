"""Cathedral shot: original-source frames [482, 808), at 30 fps.

Background is projected through a fixed room as the camera moves forward.
Character ground contact shares the floor projection. No source video reads.
Canvas: BGR uint8, 960x540. Active image is y=[53,487).
"""
from functools import lru_cache
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 960, 540
ACTIVE_Y, ACTIVE_H = 53, 434
START, CUT, END = 482, 698, 808
TRACKS=json.loads((ROOT/'camera_tracks.json').read_text(encoding='utf-8'))
PUSH=json.loads((ROOT/'push_tracks_v6.json').read_text(encoding='utf-8'))
SHAKE={k['frame']:(k['x'],k['y']) for k in json.loads((ROOT/'shake_tracks_v7.json').read_text(encoding='utf8'))['keys']}
_yy,_xx=np.indices((ACTIVE_H,WIDTH),dtype=np.float32)
_dx=_xx-480.0;_dy=_yy-PUSH['camera']['vanishing_y']
_u=_dx/PUSH['camera']['focal_px'];_v=_dy/PUSH['camera']['focal_px']


def _read(name, flags=cv2.IMREAD_UNCHANGED):
    image = cv2.imdecode(np.fromfile(ROOT / 'assets' / name, dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError(f'Could not read cathedral asset: {name}')
    return image


@lru_cache(maxsize=1)
def _assets():
    bg = _read('cathedral_background_v1.png', cv2.IMREAD_COLOR)
    fg = _read('sakiko_kneeling_v1.png').astype(np.float32) / 255.0
    if fg.shape[2] != 4:
        raise ValueError('The character sprite must have a real alpha channel.')
    # Premultiply BEFORE resampling. Transparent RGB cannot contaminate edges.
    fg[:, :, :3] *= fg[:, :, 3:4]
    title = _read('title_ja.png').astype(np.float32) / 255.0
    title[:, :, :3] *= title[:, :, 3:4]
    return bg, fg, title


def _smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _overlay(image, premult, transform=None, opacity=1.0):
    if transform is not None:
        premult = cv2.warpAffine(premult, transform, (WIDTH, ACTIVE_H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = premult[:, :, 3:4] * opacity
    return image * (1.0 - a) + premult[:, :, :3] * opacity * 255.0


def motion(source_frame):
    """Monotonic forward camera distance with continuous eased velocity."""
    name='wide' if source_frame<CUT else 'close'
    keys=PUSH[name]
    a,b=keys[0],keys[-1]
    for left,right in zip(keys,keys[1:]):
        if left['frame']<=source_frame<=right['frame']:
            a,b=left,right;break
    u=_smooth((source_frame-a['frame'])/max(1,b['frame']-a['frame']))
    times=np.array([k['frame'] for k in keys],float)
    distances=np.array([k['advance'] for k in keys],float)
    slopes=np.gradient(distances,times)
    ia=keys.index(a);ib=keys.index(b)
    t=float(np.clip((source_frame-a['frame'])/max(1,b['frame']-a['frame']),0,1))
    span=b['frame']-a['frame']
    advance=(2*t**3-3*t*t+1)*a['advance']+(t**3-2*t*t+t)*span*slopes[ia]+(-2*t**3+3*t*t)*b['advance']+(t**3-t*t)*span*slopes[ib]
    dots=a['dots']*(1-u)+b['dots']*u
    c=PUSH['camera'];vp=c['vanishing_y']
    if name=='wide':
        q=PUSH['wide_subject'];depth=q['depth']
        zoom=depth/(depth-advance);scale=q['scale_start']*zoom
        contact_y=vp+(q['contact_start']-vp)*zoom
        y=contact_y-1500*scale
        camera_y=0.
    else:
        # The close shot starts from a different camera setup at the same cut.
        delta=advance-PUSH['close'][0]['advance']
        q=PUSH['close_subject']
        zoom=q['depth']/(q['depth']-delta);scale=q['scale_start']*zoom
        # Translate camera vertically as it advances to hold the head's framing.
        camera_y=(q['head_top']-vp)*delta/c['focal_px']
        y=q['head_top']-q['head_asset_y']*scale
        contact_y=y+1500*scale
    return dict(shot=name,advance=advance,dots=dots,person_scale=scale,zoom=zoom,
                sprite_y=y,contact_y=contact_y,camera_y=camera_y)


def background_map(n,plane='wall'):
    m=motion(n);c=PUSH['camera']
    # Intersect each destination ray with the corridor's flat walls/floor/ceiling.
    # Each region is a plane homography: columns stay straight and meet the floor.
    z_side=c['half_width']/np.maximum(np.abs(_u),1e-6)
    z_floor=np.where(_v>0,(c['floor_height']-m['camera_y'])/np.maximum(_v,1e-6),1e6)
    z_ceiling=np.where(_v<0,(c['ceiling_height']+m['camera_y'])/np.maximum(-_v,1e-6),1e6)
    depth=z_floor if plane=='floor' else np.minimum(z_side,np.minimum(z_ceiling,c['back_depth']-m['advance']))
    factor=depth/(depth+m['advance'])
    cx,cy=c['plate_center'];s=c['plate_scale']
    return (cx+_dx*factor/s).astype(np.float32),(cy+(_dy*factor+c['focal_px']*m['camera_y']/(depth+m['advance']))/s).astype(np.float32)


@lru_cache(maxsize=1)
def _floor_mask():
    # The stepped boundary keeps column plinths in the rigid architecture plane.
    edge=[(0,635),(322,609),(326,647),(450,640),(452,608),(562,600),
          (566,608),(622,604),(625,590),(700,589),(887,585),
          (1074,589),(1149,590),(1152,604),(1208,608),(1212,600),
          (1322,608),(1324,640),(1448,647),(1452,609),(1773,635)]
    polygon=np.int32(edge+[(1773,886),(0,886)])
    mask=np.zeros((887,1774),np.float32);cv2.fillPoly(mask,[polygon],1)
    return cv2.GaussianBlur(mask,(0,0),1.2)


def projected_background(n):
    bg,_,_=_assets();wx,wy=background_map(n,'wall');fx,fy=background_map(n,'floor')
    walls=cv2.remap(bg,wx,wy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,0,255))
    floor=cv2.remap(bg,fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,0,255))
    # Architectural silhouettes (including column bases) stay on wall planes.
    # The full floor projection sits behind them, rather than cutting columns
    # with a separately moving floor matte.
    alpha=cv2.remap(_floor_mask(),wx,wy,cv2.INTER_LINEAR)[:,:,None]
    return np.uint8(np.clip(floor*alpha+walls*(1-alpha),0,255))


@lru_cache(maxsize=1)
def _dot_distance():
    spacing=PUSH['dot_grid']['spacing_px']
    u=(_xx+_yy)/np.sqrt(2);v=(_yy-_xx)/np.sqrt(2)
    x=np.remainder(u+spacing/2,spacing)-spacing/2
    y=np.remainder(v+spacing/2,spacing)-spacing/2
    return np.sqrt(x*x+y*y)


def halftone_background(image,strength):
    if strength<.001:return image
    luminance=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    # Original dots remain crisp while the architectural image is defocused.
    radius=.40+.62*(1-luminance)
    dots=np.clip((radius-_dot_distance())/PUSH['dot_grid']['edge_px']+.5,0,1)
    multiplier=1+strength*(.10-dots)
    return np.clip(image.astype(np.float32)*multiplier[:,:,None],0,255)


def render(source_frame):
    """Return one 960x540 BGR uint8 frame for source frame 482..807."""
    n = int(source_frame)
    if not START <= n < END:
        raise ValueError(f'Cathedral source frame must be {START}..{END - 1}, got {n}')
    bg, fg, title = _assets()
    m=motion(n)
    view=projected_background(n)
    view=cv2.GaussianBlur(view,(0,0),1.0 if n<CUT else 2.5)
    view = halftone_background(view,m['dots']).astype(np.float32)
    # The independent title sits behind the girl as in the source composition.
    if n<CUT:
        if n>=TRACKS['mosaic']['title_start']:
            z=.96+.04*m['zoom']
            view=_overlay(view,title,np.float32([[z,0,480*(1-z)],[0,z,217*(1-z)]]))
    scale=m['person_scale'];x=480-470*scale;y=m['sprite_y']
    sprite_matrix=np.float32([[scale,0,x],[0,scale,y]])
    view=_overlay(view,fg,sprite_matrix)
    if n>=CUT:
        # Source cuts abruptly to a defocused close-up, then pulls focus to the face.
        sigma=float(np.interp(n,[698,704,712,720,734,743,807],[6,5,3.5,2,.5,0,0]))
        if sigma>.05:view=cv2.GaussianBlur(view,(0,0),sigma)
    mosaic=TRACKS['mosaic'];i=n-mosaic['start']
    if 0<=i<len(mosaic['blocks_px']):
        block=mosaic['blocks_px'][i]
        small=cv2.resize(view,(max(1,WIDTH//block),max(1,ACTIVE_H//block)),interpolation=cv2.INTER_AREA)
        view=cv2.resize(small,(WIDTH,ACTIVE_H),interpolation=cv2.INTER_NEAREST)
    view=apply_shake(view,n)
    result = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    result[ACTIVE_Y:ACTIVE_Y + ACTIVE_H] = np.clip(view + 0.5, 0, 255).astype(np.uint8)
    return result


def apply_shake(active,n):
    if n<690:return active
    dx,dy=SHAKE.get(n,(0.,0.))
    # Stronger portrait entry, easing back during the push. The eye cut uses
    # a distinctly smaller residual, fading further towards the preview end.
    if n<808:
        gain=float(np.interp(n,[690,698,743,807],[2.,2.,1.8,1.5]))
        z=1.02
    else:
        gain=float(np.interp(n,[808,839,862],[.35,.20,.20]))
        z=1.01
    dx*=gain;dy*=gain
    M=np.float32([[z,0,480*(1-z)+dx],[0,z,217*(1-z)+dy]])
    return cv2.warpAffine(active,M,(960,434),flags=cv2.INTER_LINEAR)


@lru_cache(maxsize=1)
def _eye_asset():
    im=cv2.imdecode(np.fromfile(ROOT.parent/'assets/maid_eye.png',np.uint8),cv2.IMREAD_COLOR)
    return cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)[53:487]


def eye_state(n):
    keys=TRACKS['eye_proxy']['keys']
    values={k:float(np.interp(n,[v['frame'] for v in keys],[v[k] for v in keys]))
            for k in ('zoom','x','y','roll','sigma')}
    q=float(np.clip((n-808)/(862-808),0,1))
    values.update(zoom=1.04+.015*q,x=-2+4*q,y=-2+4*q,roll=-.15+.3*q)
    # A bounded pan over the existing eye material, with no invented edge pixels.
    corners=np.float32([[[0,0],[959,0],[959,433],[0,433]]])
    zoom=values['zoom']
    while True:
        M=cv2.getRotationMatrix2D((480,217),values['roll'],zoom)
        M[:,2]+=np.array([values['x'],values['y']])
        inverse=cv2.invertAffineTransform(M)
        q=cv2.transform(corners,inverse)[0]
        if q[:,0].min()>=1 and q[:,0].max()<=958 and q[:,1].min()>=1 and q[:,1].max()<=432:break
        zoom+=.015
        if zoom>2:raise RuntimeError('Eye material cannot cover motion')
    return M,values


def render_eye(n):
    # The SAME transparent portrait as the preceding shot. One continuous
    # framing path travels from her mouth to the eye at image-left, with ease-out.
    bg,fg,_=_assets()
    u=float(np.clip((n-808)/34.,0,1))
    progress=1-(1-u)**4
    mouth=np.array([476.,370.]);eye=np.array([400.,285.])
    target=mouth*(1-progress)+eye*progress
    scale=3.40+.25*progress
    # Preserve the preceding portrait's orientation: uniform scale and pan only.
    destination=np.array([740.,200.-20*progress])
    offset=destination-scale*target
    M=np.float32([[scale,0,offset[0]],[0,scale,offset[1]]])
    # Continue the cathedral plate behind the magnified portrait. Background
    # stays defocused, so camera reframing reveals real environment at left.
    active=projected_background(807)
    active=cv2.GaussianBlur(active,(0,0),7.)
    active=halftone_background(active,.18).astype(np.float32)
    active=_overlay(active,fg,M)
    # Focus and directional blur settle as the fast movement slows down.
    speed=4*(1-u)**3/34
    if speed>.004:
        offsets=np.linspace(-1,1,7)*speed*65
        acc=np.zeros_like(active)
        for offset in offsets:
            T=np.float32([[1,0,offset*.20],[0,1,offset]])
            acc+=cv2.warpAffine(active,T,(960,434),flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        active=acc/7
    sigma=3.8*(1-progress)
    if sigma>.05:active=cv2.GaussianBlur(active,(0,0),sigma)
    active=apply_shake(active,n)
    out=np.zeros((540,960,3),np.uint8)
    out[53:487]=np.clip(active+.5,0,255).astype(np.uint8)
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=630)
    parser.add_argument('--out', type=Path, default=ROOT / 'qa' / 'preview.png')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(args.out.suffix or '.png', render(args.frame))
    if not ok:
        raise RuntimeError('Could not encode preview.')
    encoded.tofile(str(args.out))
