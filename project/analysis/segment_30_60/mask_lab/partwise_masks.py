"""Independent monotonic transforms for each ribbon/ring; no group oscillation."""
import cv2
import numpy as np
from seeded_masks import SeededMasks
from renderer_v3 import ease

class PartwiseMasks(SeededMasks):
    def __init__(self,*args,**kwargs):
        kwargs['sway']=False
        super().__init__(*args,**kwargs)
        rng=np.random.default_rng(self.seed+4711)
        self.parts={}
        for gi,g in enumerate(self.groups):
            records=[]
            for i in range(g['count']+1):
                turn=1 if (i+gi)%2==0 else -1
                grow=1 if (i+2*gi)%3==0 else -1
                records.append(dict(kind='ribbon' if i<g['count'] else 'ring',angle_end_deg=float(turn*rng.uniform(10,15)),scale_end=float(1+grow*rng.uniform(.15,.2)),pivot_y=float(rng.uniform(.32,.68)),drift_y=float(rng.uniform(-.05,.05)*self.h)))
            # Include the requested full-scale extremes without synchronizing other parts.
            records[0].update(angle_end_deg=15.,scale_end=1.2)
            records[1].update(angle_end_deg=-15.,scale_end=.8)
            self.parts[g['id']]=records
        self.description['partwise_motion']=dict(rotation_range_deg=[-15,15],scale_range=[.8,1.2],independent_ribbon_and_ring=True,monotonic=True,return_to_identity=False,group_sway=False)

    def progress(self,n,g,i,ring=False):
        if ring:return float(ease((n-g['start']-10)/(g['end']-g['start']-17)))
        delay=i*g['stagger']+self.variants[g['id']][i]['delay']
        return float(ease((n-g['start']-delay)/(g['end']-g['start']-delay)))

    def part_parameters(self,n,g,i):
        p=self.parts[g['id']][i]
        q=self.progress(n,g,i,p['kind']=='ring')
        return q*p['angle_end_deg'],1+q*(p['scale_end']-1)

    def transform_part(self,points,n,g,i,pivot,q):
        p=self.parts[g['id']][i]
        angle,scale=self.part_parameters(n,g,i)
        a=np.deg2rad(angle);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        return (points-pivot)@rot.T*scale+pivot+np.array([0,q*p['drift_y']])

    def pair(self,n,g,i):
        v=self.variants[g['id']][i]
        q=self.progress(n,g,i)
        p=g['phase']+v['phase']+i*1.57
        center=-.1+i*1.2/(g['count']-1)
        z=self.z
        cx=center+.115*np.sin(z*4.6+p)+.035*np.sin(z*8-p)
        cy=z+q*g['vertical_speed']*(g['end']-g['start'])/30*(.72+.12*(i%3))
        pts=np.column_stack([cx*self.w,cy*self.h])
        taper=.58+.28*np.sin(z*2.1+p)+.14*np.sin(z*5-p)
        gain=1+2.1*q**4 if g['id']!='pattern' else 1
        breadth=q*g['half_width']*v['width']*self.w*taper*gain
        curves=[]
        part_pivot=np.array([center*self.w,self.parts[g['id']][i]['pivot_y']*self.h])
        for side in [-1,1]:
            c=pts.copy()
            c[:,0]+=side*breadth*(1+.1*side*np.sin(p))
            c[:,1]+=side*q*self.h*.055*np.sin(z*2.8+p)
            # Fixed initial orientation plus one-way opening of the two boundaries.
            a=np.deg2rad((g['spin']+v['spin'])*np.sin(p*.35)+side*q*(5.5+2.5*np.sin(p)))
            pivot=np.array([center*self.w,(.5+side*.1)*self.h])
            rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            c=(c-pivot)@rot.T+pivot+np.array([v['x']*self.w,v['y']*self.h])
            curves.append(self.transform_part(c,n,g,i,part_pivot,q))
        return q,curves

    def group(self,n,g):
        mask=np.zeros((self.h,self.w),np.uint8)
        for i in range(g['count']):
            q,(left,right)=self.pair(n,g,i)
            if q<.0002:continue
            cv2.fillPoly(mask,[np.rint(np.concatenate([left,right[::-1]])).astype(np.int32)],255,lineType=cv2.LINE_AA)
        i=g['count'];q=self.progress(n,g,i,True)
        if q>0:
            x,y,rx,ry=g['rings'][0]
            center=np.array([x*self.w,y*self.h])
            deform=1+.18*np.cos(self.circle)+.1*np.sin(2*self.circle)
            outer=np.column_stack([np.cos(self.circle)*rx*self.w*deform,np.sin(self.circle)*ry*self.h*deform])
            inner=outer*(1-.77*q)
            inner+=np.array([.035*self.w*q*np.sin(g['phase']),.045*self.h*q*np.cos(g['phase'])])
            outer=self.transform_part(outer+center,n,g,i,center,q)
            inside=self.transform_part(inner+center,n,g,i,center,q)
            ring=np.zeros_like(mask);hole=np.zeros_like(mask)
            cv2.fillPoly(ring,[np.rint(outer).astype(np.int32)],255,lineType=cv2.LINE_AA)
            cv2.fillPoly(hole,[np.rint(inside).astype(np.int32)],255,lineType=cv2.LINE_AA)
            core_gate=float(ease(q*4))*float(1-ease((q-.7)/.3))
            if core_gate>0:
                core=self.transform_part(inner*.72*core_gate+center,n,g,i,center,q)
                cv2.fillPoly(mask,[np.rint(core).astype(np.int32)],0,lineType=cv2.LINE_AA)
            mask=np.maximum(mask,np.minimum(ring,255-hole))
        mx,my=self.maps[g['id']]
        m=cv2.remap(mask,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)/255
        # Only fixed shot-level setup remains; no animated whole-mask transform.
        m=cv2.warpAffine(m,self.transform,(self.w,self.h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        if self.mirror_x:m=m[:,::-1]
        if self.mirror_y:m=m[::-1]
        return np.ascontiguousarray(m)

    def metadata(self):
        result=super().metadata();result['per_part']=self.parts
        return result
