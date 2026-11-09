"""Constant-memory field update for a fixed material; no material truth."""
import torch


def dot(a,b):
    return (a.conj()*b).sum(dtype=torch.complex128)


@torch.no_grad()
def penalty_scale(operator,forward,adjoint,response,iterations=8,fraction=.01):
    """Approximate ||C B||² from a 192-channel power iteration, not a material probe."""
    t=torch.arange(len(operator.g.rx_cols),device=operator.g.mask.device,dtype=torch.float32)
    vector=torch.exp(1j*t*1.61803398875);vector/=vector.norm()
    value=0.
    for _ in range(iterations):
        field=adjoint(operator.readout_adjoint(vector,response))
        result=operator.readout(forward(field),response)
        value=float(dot(vector,result).real)
        vector=result/result.norm().clamp_min(1e-30)
    return max(fraction*value,1e-10),value


@torch.no_grad()
def update_wavefield(operator,rhs,target,response,current,dual_equation,dual_data,
                     penalty,forward,adjoint,maximum=32,tolerance=1e-3,return_residuals=False):
    """Solve one augmented linear field problem by a short-recurrence complex CG.

    Material/field alternation and multiplier updates belong in the caller.
    Both residuals are explicit: fitting RF alone cannot qualify a model.
    """
    equation=rhs+dual_equation-operator(current)
    receivers=target+dual_data-operator.readout(current,response)
    right=adjoint(operator.readout_adjoint(receivers,response)+penalty*operator.adjoint(equation))
    def normal(z):
        field=forward(z)
        return adjoint(operator.readout_adjoint(operator.readout(field,response),response)+
                       penalty*operator.adjoint(operator(field)))
    correction=torch.zeros_like(right)
    residual=right.clone();direction=residual.clone()
    norm=dot(residual,residual).real
    initial=float(norm);history=[]
    if initial>0:
        for iteration in range(1,maximum+1):
            image=normal(direction)
            product=dot(direction,image)
            if not float(product.real)>0 or abs(float(product.imag))>.02*abs(float(product.real)):
                raise RuntimeError(f'Field normal equation lost positive Hermitian curvature: {product}')
            alpha=norm/product.real
            correction.add_(direction,alpha=float(alpha))
            residual.add_(image,alpha=-float(alpha))
            next_norm=dot(residual,residual).real
            relative=float(torch.sqrt(next_norm/max(initial,1e-30)))
            history.append({'iteration':iteration,'relative_normal_residual':relative})
            if relative<=tolerance:break
            direction.mul_(float(next_norm/norm)).add_(residual)
            norm=next_norm
    delta=forward(correction)
    final=current+delta
    before=float(dot(receivers,receivers).real+penalty*dot(equation,equation).real)
    after_receivers=target+dual_data-operator.readout(final,response)
    after_equation=rhs+dual_equation-operator(final)
    after=float(dot(after_receivers,after_receivers).real+penalty*dot(after_equation,after_equation).real)
    if after>before*(1+1e-4)+1e-20:
        raise RuntimeError(f'Linear field solve increased its actual objective: {before} -> {after}')
    physical_data=after_receivers-dual_data
    physical_equation=after_equation-dual_equation
    record={'augmented_field_cost_before':before,'augmented_field_cost_after':after,
                  'data_residual_squared':float(dot(physical_data,physical_data).real),
                  'equation_residual_squared':float(dot(physical_equation,physical_equation).real),
                  'iterations':len(history),'history':history,'converged_to_requested_normal_tolerance':bool(history and history[-1]['relative_normal_residual']<=tolerance)}
    if return_residuals:return final,record,after_equation,after_receivers
    return final,record
