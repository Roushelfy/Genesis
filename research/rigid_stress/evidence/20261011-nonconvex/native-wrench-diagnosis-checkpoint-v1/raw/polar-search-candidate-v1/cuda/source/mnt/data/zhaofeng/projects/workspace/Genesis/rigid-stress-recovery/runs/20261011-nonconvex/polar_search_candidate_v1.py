"""Bounded native dual-cone separation search; certificate validation stays separate."""
import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionState, StressTractionInfo, func_coulomb_projection, func_traction_direction,
    func_traction_geometry, func_traction_support_range, func_traction_source_friction,
)

@qd.func
def func_polar_penalty(i_c:int,i_b:int,ray:qd.types.vector(6),target:qd.types.vector(6),
                      tau:float, count:int, contacts:StressContactState,tractions:StressTractionState,info:StressTractionInfo):
    delta=ray-target
    energy=0.5*delta.dot(delta)
    gradient=delta
    hessian=qd.Matrix.identity(gs.qd_float,6)
    start,end=func_traction_support_range(i_c,i_b,tractions)
    for i_s in range(start,end):
        mu=func_traction_source_friction(i_s,i_b,contacts.friction[i_c,i_b],tractions)
        for i_q in range(tractions.sample_count[i_s,i_b]):
            if tractions.profile[i_q,i_s,i_b]>0.0:
                position,normal=func_traction_geometry(i_q,i_s,i_b,tractions,info)
                relative=(position-contacts.center[i_c,i_b])/contacts.radius[i_c,i_b]
                value=ray[0:3]+ray[3:6].cross(relative)
                projected,jacobian=func_coulomb_projection(value,normal,mu)
                operator=qd.Matrix.zero(gs.qd_float,6,3)
                for a in qd.static(range(3)):
                    operator[a,a]=1.0
                operator[3,1],operator[3,2]=-relative[2],relative[1]
                operator[4,0],operator[4,2]=relative[2],-relative[0]
                operator[5,0],operator[5,1]=-relative[1],relative[0]
                energy+=0.5*tau*projected.dot(projected)/count
                gradient+=tau*(operator@projected)/count
                hessian+=tau*(operator@jacobian@operator.transpose())/count
    return energy,gradient,hessian

@qd.kernel
def kernel_polar_search(contacts:StressContactState,sources:StressSourceState,tractions:StressTractionState,
                        info:StressTractionInfo,rays:qd.Tensor,valid:qd.Tensor,
                        iterations:qd.Tensor,polar_bounds:qd.Tensor,margins:qd.Tensor,epsilon:float):
    for i_c,i_b in qd.ndrange(contacts.valid.shape[0],contacts.valid.shape[1]):
        rays[i_c,i_b]=qd.Vector.zero(gs.qd_float,6)
        valid[i_c,i_b]=False
        iterations[i_c,i_b]=0
        scale=qd.max(contacts.force[i_c,i_b].norm(),sources.anchor_moment[i_c,i_b].norm()/contacts.radius[i_c,i_b])
        if contacts.valid[i_c,i_b] and scale>0:
            target=qd.Vector.zero(gs.qd_float,6)
            target[0:3]=contacts.force[i_c,i_b]/scale
            target[3:6]=sources.anchor_moment[i_c,i_b]/(contacts.radius[i_c,i_b]*scale)
            start,end=func_traction_support_range(i_c,i_b,tractions)
            count=0
            normal_sum=qd.Vector.zero(gs.qd_float,3)
            for i_s in range(start,end):
                for i_q in range(tractions.sample_count[i_s,i_b]):
                    if tractions.profile[i_q,i_s,i_b]>0.0:
                        _,normal=func_traction_geometry(i_q,i_s,i_b,tractions,info)
                        normal_sum+=normal
                        count+=1
            if count>0 and normal_sum.norm()>0:
                normal_reference=normal_sum.normalized()
                depth=gs.qd_float(float('inf'))
                for i_s in range(start,end):
                    mu=func_traction_source_friction(i_s,i_b,contacts.friction[i_c,i_b],tractions)
                    for i_q in range(tractions.sample_count[i_s,i_b]):
                        if tractions.profile[i_q,i_s,i_b]>0.0:
                            _,normal=func_traction_geometry(i_q,i_s,i_b,tractions,info)
                            compression=normal_reference.dot(normal)
                            depth=qd.min(depth,compression-mu*(normal_reference-compression*normal).norm())
                ray=qd.Vector.zero(gs.qd_float,6)
                tau=gs.qd_float(100.0)
                found=False
                for _phase in range(5):
                    if not found:
                        proposal=ray
                        energy=gs.qd_float(0.0)
                        gradient_squared=gs.qd_float(float('inf'))
                        alpha=gs.qd_float(1.0)
                        slope=gs.qd_float(0.0)
                        direction=ray
                        trial=False
                        active=True
                        for _ in range(128):
                            if active:
                                trial_energy,gradient,hessian=func_polar_penalty(i_c,i_b,proposal,target,tau,count,contacts,tractions,info)
                                iterations[i_c,i_b]+=1
                                squared=gradient.dot(gradient)
                                allowance=64.0*epsilon*(1.0+qd.abs(energy)+qd.abs(trial_energy))
                                accepted=not trial or trial_energy<=energy-1e-4*alpha*slope
                                accepted=accepted or (trial_energy<=energy+allowance and squared<gradient_squared)
                                if accepted:
                                    ray,energy,gradient_squared=proposal,trial_energy,squared
                                    if gradient.norm()<=1e-9:
                                        active=False
                                    else:
                                        direction,positive=func_traction_direction(hessian,gradient,0.0)
                                        slope=gradient.dot(direction)
                                        if not positive or slope<=0.0 or not slope<qd.math.inf:
                                            direction=gradient/qd.max(hessian.trace(),1.0)
                                            slope=gradient.dot(direction)
                                        fraction=qd.min(1.0,2.0*qd.max(1.0,ray.norm())/qd.max(direction.norm(),1e-30))
                                        direction*=fraction
                                        slope*=fraction
                                        alpha=1.0
                                        proposal=ray-direction
                                        trial=True
                                else:
                                    alpha*=0.5
                                    proposal=ray-alpha*direction
                        maximum=gs.qd_float(-float('inf'))
                        for i_s in range(start,end):
                            mu=func_traction_source_friction(i_s,i_b,contacts.friction[i_c,i_b],tractions)
                            for i_q in range(tractions.sample_count[i_s,i_b]):
                                if tractions.profile[i_q,i_s,i_b]>0.0:
                                    position,normal=func_traction_geometry(i_q,i_s,i_b,tractions,info)
                                    relative=(position-contacts.center[i_c,i_b])/contacts.radius[i_c,i_b]
                                    value=ray[0:3]+ray[3:6].cross(relative)
                                    compression=value.dot(normal)
                                    maximum=qd.max(maximum,compression+mu*(value-compression*normal).norm())
                        adjusted=ray
                        if depth>128.0*epsilon:
                            adjusted[0:3]-=qd.max(maximum+1e-12,0.0)/depth*normal_reference
                        if adjusted.norm()>0:
                            adjusted/=adjusted.norm()
                            maximum=gs.qd_float(-float('inf'))
                            for i_s in range(start,end):
                                mu=func_traction_source_friction(i_s,i_b,contacts.friction[i_c,i_b],tractions)
                                for i_q in range(tractions.sample_count[i_s,i_b]):
                                    if tractions.profile[i_q,i_s,i_b]>0.0:
                                        position,normal=func_traction_geometry(i_q,i_s,i_b,tractions,info)
                                        relative=(position-contacts.center[i_c,i_b])/contacts.radius[i_c,i_b]
                                        value=adjusted[0:3]+adjusted[3:6].cross(relative)
                                        compression=value.dot(normal)
                                        maximum=qd.max(maximum,compression+mu*(value-compression*normal).norm())
                            if maximum<-128.0*epsilon and target.dot(adjusted)>1e-8*qd.sqrt(2.0):
                                rays[i_c,i_b]=adjusted
                                valid[i_c,i_b]=True
                                polar_bounds[i_c,i_b]=maximum
                                margins[i_c,i_b]=target.dot(adjusted)
                                found=True
                        tau*=100.0
