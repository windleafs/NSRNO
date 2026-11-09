"""Constant-memory field CG with true full-RHS convergence measurements."""
import torch
from .wavefield_step import dot,penalty_scale


@torch.no_grad()
def update_wavefield(operator,rhs,target,response,current,dual_equation,dual_data,
                     penalty,forward,adjoint,maximum=96,tolerance=.003,return_residuals=False,normal_preconditioner=None):
    shifted_equation=rhs+dual_equation;shifted_data=target+dual_data
    equation=shifted_equation-operator(current)
    receivers=shifted_data-operator.readout(current,response)
    def gradient(eq,data):return operator.readout_adjoint(data,response)+penalty*operator.adjoint(eq)
    full_right=gradient(shifted_equation,shifted_data)
    physical_den=max(float(dot(full_right,full_right).real),1e-30)
    pre_right=adjoint(full_right)
    pre_den=max(float(dot(pre_right,pre_right).real),1e-30)
    del full_right,pre_right
    def normal(z):
        field=forward(z)
        return adjoint(gradient(operator(field),operator.readout(field,response)))
    initial_gradient=gradient(equation,receivers)
    right=adjoint(initial_gradient)
    initial_physical=float((dot(initial_gradient,initial_gradient).real/physical_den).sqrt())
    del initial_gradient
    precondition=(lambda value:value) if normal_preconditioner is None else normal_preconditioner
    correction=torch.zeros_like(right);residual=right.clone();pre_residual=precondition(residual);direction=pre_residual.clone()
    rho=dot(residual,pre_residual).real
    norm=dot(residual,residual).real;initial=max(float(norm),1e-30)
    initial_pre=float((norm/pre_den).sqrt());history=[]
    converged=initial_physical<=tolerance and initial_pre<=tolerance;last_check=0
    if not converged and float(norm)>0:
        for iteration in range(1,maximum+1):
            image=normal(direction);product=dot(direction,image)
            if not float(product.real)>0 or abs(float(product.imag))>.02*abs(float(product.real)):
                raise RuntimeError(f'Field normal equation lost positive Hermitian curvature: {product}')
            if not float(rho)>0:raise RuntimeError('Normal preconditioner lost positive curvature')
            alpha=rho/product.real;correction.add_(direction,alpha=float(alpha))
            residual.add_(image,alpha=-float(alpha));next_norm=dot(residual,residual).real
            row={'iteration':iteration,'relative_normal_residual':float((next_norm/initial).sqrt()),
                 'recursive_preconditioned_global_normal_residual':float((next_norm/pre_den).sqrt())}
            # A fresh physical residual avoids mistaking recurrence drift or a
            # small preconditioned gradient for a solved physical normal system.
            check=iteration%16==0 or iteration==maximum or (iteration-last_check>=4 and row['recursive_preconditioned_global_normal_residual']<=tolerance)
            if check:
                last_check=iteration
                candidate=current+forward(correction)
                actual_gradient=gradient(shifted_equation-operator(candidate),
                    shifted_data-operator.readout(candidate,response))
                actual_residual=adjoint(actual_gradient)
                true_norm=dot(actual_residual,actual_residual).real
                drift=float((dot(actual_residual-residual,actual_residual-residual).real/true_norm.clamp_min(1e-30)).sqrt())
                row.update(physical_global_normal_residual=float((dot(actual_gradient,actual_gradient).real/physical_den).sqrt()),
                    preconditioned_global_normal_residual=float((true_norm/pre_den).sqrt()),residual_drift=drift)
                converged=row['physical_global_normal_residual']<=tolerance and row['preconditioned_global_normal_residual']<=tolerance
                del actual_gradient,candidate
                history.append(row)
                if converged:break
                if drift>.05:
                    residual=actual_residual;pre_residual=precondition(residual)
                    direction=pre_residual.clone();rho=dot(residual,pre_residual).real;norm=true_norm
                    row['residual_replaced']=True;continue
            else:history.append(row)
            pre_residual=precondition(residual);next_rho=dot(residual,pre_residual).real
            direction.mul_(float(next_rho/rho)).add_(pre_residual);rho=next_rho;norm=next_norm
    final=current+forward(correction)
    after_equation=shifted_equation-operator(final)
    after_receivers=shifted_data-operator.readout(final,response)
    actual_gradient=gradient(after_equation,after_receivers);pre_gradient=adjoint(actual_gradient)
    physical=float((dot(actual_gradient,actual_gradient).real/physical_den).sqrt())
    preconditioned=float((dot(pre_gradient,pre_gradient).real/pre_den).sqrt())
    before=float(dot(receivers,receivers).real+penalty*dot(equation,equation).real)
    after=float(dot(after_receivers,after_receivers).real+penalty*dot(after_equation,after_equation).real)
    if after>before*(1+1e-4)+1e-20:raise RuntimeError(f'Linear field cost increased: {before} -> {after}')
    physical_data=after_receivers-dual_data;physical_equation=after_equation-dual_equation
    record={'augmented_field_cost_before':before,'augmented_field_cost_after':after,
        'data_residual_squared':float(dot(physical_data,physical_data).real),
        'equation_residual_squared':float(dot(physical_equation,physical_equation).real),
        'iterations':len(history),'history':history,
        'initial_physical_global_normal_residual':initial_physical,
        'initial_preconditioned_global_normal_residual':initial_pre,
        'physical_global_normal_residual':physical,'preconditioned_global_normal_residual':preconditioned,
        'normalization':'full shifted normal-equation RHS; never the warm correction RHS',
        'converged_to_requested_normal_tolerance':physical<=tolerance and preconditioned<=tolerance}
    if return_residuals:return final,record,after_equation,after_receivers
    return final,record
