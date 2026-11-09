"""Joint RMS trust constraints inside the fixed-wavefield material solve.

The baseline independently rescales r, beta and t after a coupled GN solve.
That can spoil the pressure-block step. Here three nonnegative dual shifts
enforce the three RMS bounds in the same coupled positive normal equations.
No additional propagation or material derivative probes are used.
"""
import numpy as np
from .wri_material import MaterialVariables,MaterialNormal as BaseNormal
from .coupled_coordinates import pcg,dot


class MaterialNormal(BaseNormal):
    def solve(self,current,moments,tv_weight=.001,damping_relative=.01,maximum=64,
              trust=(.03,.04,.10),callback=None):
        tv,g_tv,h_tv,tv_local=self.vars.tv(current,tv_weight)
        gradient=-self.rhs+g_tv;gradient[:,:80]=0;right=-gradient
        data_diag=np.maximum(self.diagonal,0)
        reference=[]
        for a in data_diag:
            positive=a[80:][a[80:]>0]
            reference.append(max(float(np.median(positive)) if positive.size else 0.,1e-18))
        base=damping_relative*np.maximum(data_diag,np.array(reference)[:,None,None]*1e-4)
        base[:,:80]=1.
        unknown=np.asarray(self.vars.unknown,np.float64)
        target=np.asarray(trust,np.float64)/.1
        multipliers=np.zeros(3,np.float64);linear_records=[]

        def rms(q):return np.sqrt(np.mean(q[:,80:]**2,axis=(-2,-1)))

        def linear(shifts,warm=None,label=''):
            damping=base+shifts[:,None,None]*unknown
            diagonal=np.maximum(data_diag+tv_local[:3]+damping,1e-30)
            scales=1/np.sqrt(diagonal)
            a=tv_local[3]*scales[0]*scales[1]/1.00001
            b=tv_local[4]*scales[0]*scales[2]/1.00001
            c=(self.normal['hbt']+tv_local[5])*scales[1]*scales[2]/1.00001
            determinant=np.maximum(1+2*a*b*c-a*a-b*b-c*c,1e-12)
            inv=np.stack([1-c*c,1-b*b,1-a*a,b*c-a,a*c-b,a*b-c])/determinant
            def precondition(rhs):
                y=rhs*scales
                return np.stack([inv[0]*y[0]+inv[3]*y[1]+inv[4]*y[2],
                    inv[3]*y[0]+inv[1]*y[1]+inv[5]*y[2],
                    inv[4]*y[0]+inv[5]*y[1]+inv[2]*y[2]])*scales
            def operator(q):return self.apply(q)+h_tv(q)+damping*q
            initial=np.zeros_like(right) if warm is None else warm
            residual=right-operator(initial) if warm is not None else right
            correction,history=pcg(operator,residual,precondition,maximum=maximum,
                tolerance=.01,callback=callback)
            solution=initial+correction;solution[:,:80]=0
            residual=right-operator(solution)
            relative=np.sqrt(max(dot(residual,precondition(residual)),0.)/
                             max(dot(right,precondition(right)),1e-300))
            linear_records.append({'label':label,'dual_shifts':shifts.tolist(),
                'iterations':len(history),'global_preconditioned_relative_residual':float(relative),
                'RMS_in_normalized_coordinates':rms(solution).tolist()})
            return solution,history

        direction,initial_history=linear(multipliers,label='unconstrained_GN')
        raw_rms=rms(direction)
        # A bracket scale only; every proposed multiplier uses the full coupled operator.
        full_diagonal=data_diag+tv_local[:3]+base
        effective=full_diagonal.copy()
        cross=self.normal['hbt']+tv_local[5]
        effective[2]-=cross*cross/np.maximum(full_diagonal[1],1e-30)
        effective=np.maximum(effective,np.array(reference)[:,None,None]*1e-6)
        effective_reference=np.array([float(np.median(a[80:])) for a in effective])
        for sweep in range(6):
            for coordinate in range(3):
                current_rms=rms(direction)
                if multipliers[coordinate]==0 and current_rms[coordinate]<=target[coordinate]*1.001:
                    continue
                zero=multipliers.copy();zero[coordinate]=0
                at_zero,_=linear(zero,warm=direction,label=f'trust{sweep}_coordinate{coordinate}_zero')
                if rms(at_zero)[coordinate]<=target[coordinate]:
                    multipliers=zero;direction=at_zero;continue
                low=0.;high=max(multipliers[coordinate],effective_reference[coordinate]*
                    max(rms(at_zero)[coordinate]/target[coordinate]-1,.1),1e-20)
                high_direction=at_zero
                for bracket in range(24):
                    shifted=multipliers.copy();shifted[coordinate]=high
                    high_direction,_=linear(shifted,warm=high_direction,
                        label=f'trust{sweep}_coordinate{coordinate}_bracket{bracket}')
                    if rms(high_direction)[coordinate]<=target[coordinate]:break
                    high*=4
                else:raise RuntimeError('Could not bracket a positive material trust multiplier')
                warm=high_direction
                for iteration in range(12):
                    mid=(low+high)/2;shifted=multipliers.copy();shifted[coordinate]=mid
                    trial,_=linear(shifted,warm=warm,
                        label=f'trust{sweep}_coordinate{coordinate}_bisect{iteration}')
                    value=rms(trial)[coordinate];warm=trial
                    if value>target[coordinate]:low=mid
                    else:high=mid;high_direction=trial
                    if value<=target[coordinate] and value>=target[coordinate]*.99:break
                multipliers[coordinate]=high;direction=high_direction
            values=rms(direction)
            active=multipliers>0
            if np.all(values<=target*1.005) and np.all(values[active]>=target[active]*.98):break
        final_rms=rms(direction)
        # One common safeguard preserves beta/t coupling; no component is rescaled separately.
        common_scale=min(1.,1/max(float(np.max(final_rms/target)),1e-30))
        direction=direction*common_scale;kind='RMS_constrained_conditional_GN'
        if dot(gradient,direction)>=0:
            direction=-gradient/np.maximum(full_diagonal,1e-30)
            direction*=min(1.,1/max(float(np.max(rms(direction)/target)),1e-30))
            kind='common_scaled_diagonal_descent_fallback'
        delta=.1*direction
        candidates=[];accepted=current.copy()
        for line in range(9):
            alpha=.5**line;proposal=self.vars.project(current+alpha*delta)
            actual_rms=np.sqrt(np.mean((proposal-current)[:,80:]**2,axis=(-2,-1)))
            slope=dot(gradient,(proposal-current)/.1)
            if slope>=0 or np.any(actual_rms>np.asarray(trust)*1.01):
                candidates.append({'alpha':alpha,'slope':slope,'actual_RMS':actual_rms.tolist(),'accepted':False});continue
            density=self.data_change_density(current,proposal)
            pressure=self.pressure_change(current,proposal,moments)
            proposed_tv=self.vars.tv(proposal,tv_weight)[0]
            change=density+pressure+proposed_tv-tv;ok=change<=1e-4*slope
            row={'alpha':alpha,'density_cost_change':density,'pressure_cost_change':pressure,
                'TV_change':proposed_tv-tv,'fixed_field_objective_change':change,'slope':slope,
                'actual_RMS':actual_rms.tolist(),'accepted':bool(ok)}
            candidates.append(row)
            if ok:accepted=proposal;break
        return accepted,{'direction_kind':kind,'CG_history':initial_history,'linear_solves':linear_records,
            'raw_unconstrained_RMS_r_beta_t':(.1*raw_rms).tolist(),
            'trust_RMS_r_beta_t':list(trust),'trust_dual_shifts':multipliers.tolist(),
            'common_safety_scale':common_scale,'line_search':candidates,
            'accepted':bool(candidates and candidates[-1]['accepted']),
            'update_RMS_r_beta_eta':np.sqrt(np.mean((accepted-current)[:,80:]**2,axis=(-2,-1))).tolist(),
            'gradient_norm_r_beta_eta_coordinates':np.sqrt(np.sum(gradient**2,axis=(-2,-1))).tolist(),
            'TV_before':tv,'damping_reference_per_parameter':reference,
            'truth_read':False,'propagation_in_material_step':False}
