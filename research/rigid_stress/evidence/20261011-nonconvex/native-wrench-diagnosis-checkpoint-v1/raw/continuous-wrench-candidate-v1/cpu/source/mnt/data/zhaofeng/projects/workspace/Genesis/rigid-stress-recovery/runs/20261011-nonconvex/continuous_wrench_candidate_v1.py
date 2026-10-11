"""Quadrants conservative continuous disk/triangle wrench-separation bounds."""
import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, func_closest_triangle
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionState, func_traction_support_range, func_traction_source_friction

@qd.kernel
def kernel_continuous_wrench_bound(
    contacts: StressContactState,
    sources: StressSourceState,
    tractions: StressTractionState,
    info: StressSurfaceInfo,
    rays: qd.Tensor,
    bounds: qd.Tensor,
    margins: qd.Tensor,
    valid: qd.Tensor,
    visited: qd.Tensor,
    workspace: qd.Tensor,
    depths: qd.Tensor,
    epsilon: float,
    max_depth: int,
    max_nodes: int,
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0],contacts.valid.shape[1]):
        valid[i_c,i_b]=False
        bounds[i_c,i_b]=gs.qd_float(float('inf'))
        margins[i_c,i_b]=0.0
        visited[i_c,i_b]=0
        ray=rays[i_c,i_b]
        radius=contacts.radius[i_c,i_b]
        scale=qd.max(contacts.force[i_c,i_b].norm(),sources.anchor_moment[i_c,i_b].norm()/radius)
        if contacts.valid[i_c,i_b] and ray.norm()>0.0 and scale>0.0:
            ray/=ray.norm()
            target=qd.Vector.zero(gs.qd_float,6)
            target[0:3]=contacts.force[i_c,i_b]/scale
            target[3:6]=sources.anchor_moment[i_c,i_b]/(radius*scale)
            margins[i_c,i_b]=target.dot(ray)
            if target.dot(ray)>1e-8:
                resolved=True
                maximum=gs.qd_float(-float('inf'))
                count=0
                faces=0
                start,end=func_traction_support_range(i_c,i_b,tractions)
                for i_s in range(start,end):
                    center=contacts.center[i_s,i_b]
                    support_radius=contacts.radius[i_s,i_b]
                    mu=func_traction_source_friction(i_s,i_b,contacts.friction[i_c,i_b],tractions)
                    geometric_roundoff=128.0*epsilon*(center.norm()+support_radius)
                    for i_f in range(info.face_origin.shape[0]):
                        if resolved:
                            closest,_=func_closest_triangle(i_f,center,info)
                            if (closest-center).norm()<=support_radius+geometric_roundoff:
                                faces+=1
                                workspace[0,i_c,i_b]=qd.Vector([0.0,0.0,1.0,0.0,0.0,1.0])
                                depths[0,i_c,i_b]=0
                                size=1
                                while size>0 and resolved:
                                    size-=1
                                    bary=workspace[size,i_c,i_b]
                                    depth=depths[size,i_c,i_b]
                                    count+=1
                                    if count>max_nodes:
                                        resolved=False
                                    else:
                                        origin,edges=info.face_origin[i_f],info.face_edges[i_f]
                                        points=qd.Matrix.zero(gs.qd_float,3,3)
                                        for a in qd.static(range(3)):
                                            points[a,:]=origin+bary[2*a]*edges[0,:]+bary[2*a+1]*edges[1,:]
                                        nearest=qd.Vector.zero(gs.qd_float,3)
                                        for k in qd.static(range(3)):
                                            low=qd.min(points[0,k],points[1,k],points[2,k])
                                            high=qd.max(points[0,k],points[1,k],points[2,k])
                                            nearest[k]=qd.max(low-center[k],center[k]-high,0.0)
                                        if nearest.norm()<=support_radius+geometric_roundoff:
                                            normal=-info.face_normal[i_f]
                                            upper=gs.qd_float(-float('inf'))
                                            for a in qd.static(range(3)):
                                                relative=(points[a,:]-contacts.center[i_c,i_b])/radius
                                                value=ray[0:3]+ray[3:6].cross(relative)
                                                compression=value.dot(normal)
                                                polar=compression+mu*(value-compression*normal).norm()
                                                allowance=128.0*epsilon*(1.0+mu)*(1.0+value.norm()+relative.norm())
                                                upper=qd.max(upper,polar+allowance)
                                            if upper<0.0:
                                                maximum=qd.max(maximum,upper)
                                            elif depth>=max_depth or size+4>workspace.shape[0]:
                                                resolved=False
                                            else:
                                                a,b,c=bary[0:2],bary[2:4],bary[4:6]
                                                ab,bc,ca=0.5*(a+b),0.5*(b+c),0.5*(c+a)
                                                workspace[size,i_c,i_b]=qd.Vector([a[0],a[1],ab[0],ab[1],ca[0],ca[1]])
                                                workspace[size+1,i_c,i_b]=qd.Vector([ab[0],ab[1],b[0],b[1],bc[0],bc[1]])
                                                workspace[size+2,i_c,i_b]=qd.Vector([ca[0],ca[1],bc[0],bc[1],c[0],c[1]])
                                                workspace[size+3,i_c,i_b]=qd.Vector([ab[0],ab[1],bc[0],bc[1],ca[0],ca[1]])
                                                for child in qd.static(range(4)):
                                                    depths[size+child,i_c,i_b]=depth+1
                                                size+=4
                visited[i_c,i_b]=count
                bounds[i_c,i_b]=maximum
                valid[i_c,i_b]=resolved and faces>0 and maximum<0.0
