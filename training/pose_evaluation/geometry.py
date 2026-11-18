"""NumPy-only geometric estimator. No labels, recipes or simulator dependency.

Inputs are a semantic prediction, observed optical-z/validity and calibrated
OpenCV camera K/world transform. Dimensions are fixed nominal task knowledge.
The capsule's unobservable axial roll is never estimated.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Calibration:
    K: np.ndarray
    world_from_camera: np.ndarray

    def __post_init__(self):
        k=np.asarray(self.K,dtype=float);t=np.asarray(self.world_from_camera,dtype=float)
        if k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all():
            raise ValueError('Invalid calibration shape/values')
        if k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):raise ValueError('Invalid intrinsics')
        if not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(t[:3,:3])<0:
            raise ValueError('Invalid camera rigid transform')
        object.__setattr__(self,'K',k.copy());object.__setattr__(self,'world_from_camera',t.copy())


def calibration_from_annotation(annotation):
    """Strict allowlist at legacy dataset boundary; never forward the annotation."""
    if annotation['camera_convention']!='OpenCV: +x right, +y down, +z forward; column-vector transforms':
        raise ValueError('Unknown camera convention')
    if annotation['depth_definition']!='optical-axis z, metres; observed invalid=0':raise ValueError('Unknown depth convention')
    return Calibration(annotation['K'],annotation['world_from_camera'])


def backproject(depth,valid,mask,calibration):
    depth=np.asarray(depth);valid=np.asarray(valid);mask=np.asarray(mask,dtype=bool)
    if depth.ndim!=2 or depth.shape!=valid.shape or mask.shape!=depth.shape:raise ValueError('Image shape mismatch')
    v,u=np.nonzero(mask & (valid>0) & np.isfinite(depth) & (depth>0))
    rays=np.stack([u,v,np.ones(len(u))],axis=1)@np.linalg.inv(calibration.K).T
    local=rays*depth[v,u,None]
    t=calibration.world_from_camera
    return local@t[:3,:3].T+t[:3,3]


def largest_component(mask):
    """Deterministic 8-connectivity; ties retain raster-first component."""
    mask=np.asarray(mask,dtype=bool);seen=np.zeros_like(mask);best=[]
    h,w=mask.shape
    for flat in np.flatnonzero(mask):
        y,x=divmod(int(flat),w)
        if seen[y,x]:continue
        seen[y,x]=True;stack=[(y,x)];component=[]
        while stack:
            y,x=stack.pop();component.append((y,x))
            for yy in range(max(0,y-1),min(h,y+2)):
                for xx in range(max(0,x-1),min(w,x+2)):
                    if mask[yy,xx] and not seen[yy,xx]:seen[yy,xx]=True;stack.append((yy,xx))
        if len(component)>len(best):best=component
    result=np.zeros_like(mask)
    if best:
        y,x=np.array(best).T;result[y,x]=True
    return result


def erode_cross(mask):
    m=np.pad(mask,1)
    return m[1:-1,1:-1]&m[:-2,1:-1]&m[2:,1:-1]&m[1:-1,:-2]&m[1:-1,2:]


def axis_error_deg(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.linalg.norm(a)<1e-12 or np.linalg.norm(b)<1e-12:raise ValueError('Zero axis')
    return float(np.degrees(np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0,1))))


def _reject(reason,quality=None):
    return {'accepted':False,'reasons':[reason],'position_m':None,'axis_world':None,'quality':quality or {}}


def _finish(position,axis,quality,reasons):
    return {'accepted':not reasons,'reasons':reasons,'position_m':np.asarray(position).tolist(),
            'axis_world':np.asarray(axis).tolist(),'quality':quality}


def _huber_cost(residual,delta=.002):
    a=np.abs(residual)
    return float(np.mean(np.where(a<=delta,.5*a*a,delta*(a-.5*delta))))


def _capsule_fit(points,center,axis,radius,half_segment):
    """Robust finite-difference LM, 3 centre coordinates + 2 axis coordinates."""
    basis=np.eye(3)[np.argmin(np.abs(axis))]
    u=np.cross(axis,basis);u/=np.linalg.norm(u);v=np.cross(axis,u)
    def decode(q):
        a=axis+u*q[3]+v*q[4];a/=np.linalg.norm(a)
        return center+q[:3]*.01,a
    def residual(q):
        c,a=decode(q);p=points-c;s=np.clip(p@a,-half_segment,half_segment)
        return np.linalg.norm(p-s[:,None]*a,axis=1)-radius
    q=np.zeros(5);lam=1e-5
    def jacobian(q,r):
        return np.stack([(residual(q+np.eye(5)[j]*1e-4)-r)/1e-4 for j in range(5)],axis=1)
    for _ in range(30):
        r=residual(q);j=jacobian(q,r);weights=np.minimum(1,.002/np.maximum(np.abs(r),1e-12))
        lhs=j.T@(weights[:,None]*j)+np.eye(5)*lam
        step=np.linalg.solve(lhs,-j.T@(weights*r));step=np.clip(step,-.25,.25)
        candidate=q+step;candidate[:3]=np.clip(candidate[:3],-1.5,1.5);candidate[3:]=np.clip(candidate[3:],-.8,.8)
        if _huber_cost(residual(candidate))<_huber_cost(r):
            q=candidate;lam=max(lam*.4,1e-9)
            if np.linalg.norm(step)<1e-5:break
        else:lam=min(lam*5,1)
    r=residual(q);singular=np.linalg.svd(jacobian(q,r),compute_uv=False)
    c,a=decode(q)
    return c,a,r,float(singular[-1]/max(singular[0],1e-12))


def capsule_from_points(points,camera_position,protocol):
    dims=protocol['nominal_dimensions_m'];r=dims['capsule_radius'];half=dims['capsule_half_segment'];length=2*(half+r)
    center=np.median(points,axis=0);_,eigenvectors=np.linalg.eigh(np.cov((points-center).T))
    axis=eigenvectors[:,-1];eigenvalues=np.linalg.eigvalsh(np.cov((points-center).T))
    ratio=float(eigenvalues[-1]/max(eigenvalues[-2],1e-12))
    if ratio<protocol['capsule_min_pca_ratio']:return _reject('capsule_axis_poorly_separated',{'pca_ratio':ratio})
    span=points@axis;center+=((np.quantile(span,.99)+np.quantile(span,.01))/2-center@axis)*axis
    away=center-np.asarray(camera_position);away-=away@axis*axis
    if np.linalg.norm(away)<1e-9:return _reject('capsule_end_on')
    center+=r*np.pi/4*away/np.linalg.norm(away)
    center,axis,residual,jratio=_capsule_fit(points,center,axis,r,half)
    span=float(np.quantile(points@axis,.99)-np.quantile(points@axis,.01))
    view=center-np.asarray(camera_position);view/=np.linalg.norm(view)
    end_on=float(abs(axis@view));med=float(np.median(np.abs(residual)));p90=float(np.quantile(np.abs(residual),.9))
    behind_surface=float((center-np.median(points,axis=0))@view)
    quality={'pca_ratio':ratio,'axial_span_m':span,'nominal_length_m':length,'end_on_cosine':end_on,
             'surface_residual_median_m':med,'surface_residual_p90_m':p90,'scaled_jacobian_ratio':jratio,
             'center_behind_visible_surface_m':behind_surface}
    reasons=[]
    if not protocol['capsule_min_axial_span_fraction']*length<=span<=protocol['capsule_max_axial_span_fraction']*length:reasons.append('capsule_incomplete_or_wrong_extent')
    if end_on>protocol['capsule_max_end_on_cosine']:reasons.append('capsule_end_on')
    if med>protocol['capsule_fit_median_limit_m'] or p90>protocol['capsule_fit_p90_limit_m']:reasons.append('capsule_surface_fit_inconsistent')
    if jratio<protocol['capsule_min_scaled_jacobian_ratio']:reasons.append('capsule_fit_poorly_conditioned')
    if not -.002<=behind_surface<=2*r+.002:reasons.append('capsule_wrong_visible_surface_side')
    return _finish(center,axis,quality,reasons)


def fixture_from_points(points,protocol):
    """Horizontal fixture: robust minimum-area footprint and floor-height mode.

    No table-height lookup. Floor reference is recovered from observed height
    clusters and the declared rim-to-floor height. Tilt is assumed, not measured.
    """
    dims=protocol['nominal_dimensions_m'];outer=np.asarray(dims['fixture_cavity'])+2*dims['fixture_wall_thickness']
    best=None
    # A one-degree grid has <=0.5 degree quantization before measurement error.
    for yaw in np.radians(np.arange(180)):
        a=np.array([np.cos(yaw),np.sin(yaw)]);b=np.array([-a[1],a[0]])
        coordinates=points[:,:2]@np.stack([a,b],axis=1)
        lo,hi=np.quantile(coordinates,[.01,.99],axis=0);extent=hi-lo
        area=float(np.prod(extent))
        if best is None or area<best[0]:best=(area,a,b,lo,hi,extent)
    _,a,b,lo,hi,extent=best
    xy=((lo+hi)/2)@np.stack([a,b])
    if extent[0]<extent[1]:a=b;extent=extent[::-1]
    # Mode of overlapping 3-mm horizontal bands. A floor and rim candidate are
    # scored against the complete declared set of horizontal surface levels.
    z=points[:,2];height=dims['fixture_wall_height']
    candidates=np.linspace(np.quantile(z,.05)-height,np.quantile(z,.95),80)
    losses=[np.mean(np.minimum((z-floor)**2,(z-floor-height)**2).clip(max=.005**2)) for floor in candidates]
    floor=float(candidates[int(np.argmin(losses))]);spread=float(np.quantile(z,.95)-np.quantile(z,.05))
    # Both floor and rim must be supported. Otherwise their height interpretation
    # is ambiguous by an entire wall height; reject rather than use table truth.
    floor_support=float(np.mean(np.abs(z-floor)<.0025));rim_support=float(np.mean(np.abs(z-floor-height)<.0025))
    quality={'footprint_extent_m':extent.tolist(),'nominal_outer_extent_m':outer.tolist(),
             'height_spread_m':spread,'floor_support_fraction':floor_support,'rim_support_fraction':rim_support}
    reasons=[]
    if np.any(extent<outer*protocol['fixture_min_extent_fraction']) or np.any(np.abs(extent-outer)>protocol['fixture_extent_tolerance_m']):reasons.append('fixture_incomplete_or_wrong_extent')
    if spread>protocol['fixture_height_spread_limit_m']:reasons.append('fixture_height_inconsistent')
    if floor_support<.15 or rim_support<.05:reasons.append('fixture_floor_rim_height_ambiguous')
    return _finish(np.array([*xy,floor]),np.array([*a,0]),quality,reasons)


def estimate_pose(mask,depth,valid,calibration,target,protocol):
    if target not in ('capsule','fixture'):raise ValueError('Unknown target')
    mask=np.asarray(mask,dtype=bool);component=largest_component(mask);count=int(component.sum())
    if count<protocol['min_component_pixels']:return _reject('too_few_mask_pixels',{'component_pixels':count})
    fraction=count/max(int(mask.sum()),1)
    if fraction<protocol['min_component_fraction']:return _reject('fragmented_mask',{'largest_component_fraction':fraction})
    if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():return _reject('cropped_at_image_border')
    # Capsule silhouette mixes background depth; fixture rims can be only one
    # pixel thick, so preserve their points and explicitly check floor/rim support.
    interior=erode_cross(component) if target=='capsule' else component
    depth_valid=(np.asarray(valid)>0)&np.isfinite(depth)&(depth>0)&(depth<=protocol['max_depth_m'])
    points=backproject(depth,depth_valid,interior,calibration)
    if len(points)<protocol['min_depth_points']:return _reject('too_few_interior_depth_points',{'depth_points':len(points)})
    # Gross contamination guard; no target position/workspace prior.
    med=np.median(points,axis=0);limit=.06 if target=='capsule' else .10
    keep=np.linalg.norm(points-med,axis=1)<limit
    if np.mean(keep)<.9:return _reject('depth_component_not_compact')
    points=points[keep]
    if len(points)<protocol['min_depth_points']:return _reject('too_few_compact_depth_points')
    n=protocol['max_fit_points']
    if len(points)>n:points=points[np.linspace(0,len(points)-1,n,dtype=int)]
    result=capsule_from_points(points,calibration.world_from_camera[:3,3],protocol) if target=='capsule' else fixture_from_points(points,protocol)
    result['quality'].update({'component_pixels':count,'largest_component_fraction':fraction,'fit_points':len(points)})
    return result
