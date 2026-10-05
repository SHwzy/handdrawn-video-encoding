"""Stable per-shot mask variants. Randomness is sampled once, never per frame."""
import copy
import numpy as np
import cv2
from renderer_v3 import CurvePairMasks,CFG

class SeededMasks(CurvePairMasks):
    def __init__(self,w=960,h=434,seed=20260930,rotation_deg=0,mirror_x=False,mirror_y=False,variation=.35,sway=False):
        super().__init__(w,h)
        self.seed=seed
        self.groups=copy.deepcopy(CFG['groups'])
        rng=np.random.default_rng(seed)
        self.variants={}
        self.description=dict(seed=seed,rotation_deg=rotation_deg,mirror_x=mirror_x,mirror_y=mirror_y,variation=variation,frame_randomness=False)
        for g in self.groups:
            entries=[]
            for i in range(g['count']):
                entries.append(dict(phase=float(rng.uniform(-.7,.7)*variation),width=float(1+rng.uniform(-.2,.2)*variation),spin=float(rng.uniform(-4,4)*variation),x=float(rng.uniform(-.05,.05)*variation),y=float(rng.uniform(-.08,.08)*variation),delay=float(rng.uniform(-4,4)*variation)))
            self.variants[g['id']]=entries
        self.transform=cv2.getRotationMatrix2D((w/2,h/2),rotation_deg,1.)
        self.mirror_x,self.mirror_y=mirror_x,mirror_y
        self.sway=sway
        if sway:self.description['sway']=dict(rotation_amplitude_deg=1.8,rotation_period_seconds=4.2,scale_amplitude=.015,scale_bias=.03,scale_period_seconds=3.7,independent_group_phases=True,coverage_scale_guard=True)
        if sway=='strong':self.description['sway']=dict(rotation_amplitude_deg=15,scale_range=[.8,1.2],procedural_padding_px=280,independent_group_timing=True,coverage_scale_guard=False)

    def sway_parameters(self,n,g):
        phase={'white':0.,'pattern':1.1,'return':2.3}[g['id']]
        progress=np.clip((n-g['start'])/(g['end']-g['start']),0.,1.)
        if self.sway=='strong':
            # A smooth time warp changes group timing, while preserving extrema and endpoints.
            shift={'white':0.,'pattern':.10,'return':-.10}[g['id']]
            p=progress+shift*np.sin(2*np.pi*progress)
            p=p*p*(3-2*p)
            return float(15*np.sin(2*np.pi*p)),float(1+.2*np.sin(4*np.pi*p))
        envelope=float(np.sin(np.pi*progress)**2)
        t=(n-g['start'])/30
        angle=float(1.8*np.sin(2*np.pi*t/4.2+phase)*envelope)
        scale=1+envelope*(.03+.015*np.sin(2*np.pi*t/3.7+phase+.6))
        # Uniform overscan ensures all inverse-mapped corners stay inside the mask.
        a=np.deg2rad(angle);safe=abs(np.cos(a))+max(self.w/self.h,self.h/self.w)*abs(np.sin(a))
        scale=float(max(scale,safe+.001*envelope))
        return angle,scale

    def pair(self,n,g,i):
        v=self.variants[g['id']][i]
        local=copy.copy(g)
        local['phase']+=v['phase']
        local['half_width']*=v['width']
        local['spin']+=v['spin']
        q,curves=super().pair(n-v['delay'],local,i)
        delta=np.array([v['x']*self.w,v['y']*self.h])
        return q,[c+delta for c in curves]

    def group(self,n,g):
        if self.sway=='strong':
            pad=280
            m=super().group(n,g,padding=pad)
            fixed=cv2.getRotationMatrix2D((self.w/2+pad,self.h/2+pad),self.description['rotation_deg'],1.)
            m=cv2.warpAffine(m,fixed,(self.w+2*pad,self.h+2*pad),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            if self.mirror_x:m=m[:,::-1]
            if self.mirror_y:m=m[::-1]
            angle,scale=self.sway_parameters(n,g)
            transform=cv2.getRotationMatrix2D(((self.w-1)/2+pad,(self.h-1)/2+pad),angle,scale)
            transform[:,2]-=pad
            return cv2.warpAffine(m,transform,(self.w,self.h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        m=super().group(n,g)
        m=cv2.warpAffine(m,self.transform,(self.w,self.h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        if self.mirror_x:m=m[:,::-1]
        if self.mirror_y:m=m[::-1]
        if self.sway:
            angle,scale=self.sway_parameters(n,g)
            transform=cv2.getRotationMatrix2D(((self.w-1)/2,(self.h-1)/2),angle,scale)
            m=cv2.warpAffine(m,transform,(self.w,self.h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        return np.ascontiguousarray(m)

    def weights(self,n):
        white,pattern,back=[self.group(n,g) for g in self.groups]
        red=white*pattern*(1-back)
        pale=white*(1-pattern)*(1-back)
        return 1-pale-red,pale,red

    def metadata(self):
        return dict(settings=self.description,per_curve=self.variants)
