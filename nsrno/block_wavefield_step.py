"""Block CG across all transmissions; full native fields and receiver data."""
import torch


def gram(a,b):
    x=a.flatten(1);y=b.flatten(1);result=torch.zeros((len(x),len(y)),dtype=torch.complex128,device=x.device)
    for k in range(0,x.shape[1],131072):
        result.add_(x[:,k:k+131072].to(torch.complex128).conj()@y[:,k:k+131072].to(torch.complex128).T)
    return result


def mix(fields,matrix):
    return (matrix.T.to(fields.dtype)@fields.flatten(1)).reshape_as(fields)


def solve_spd(matrix,right):
    if float(torch.linalg.vector_norm(matrix-matrix.mH))>.02*max(float(torch.linalg.vector_norm(matrix)),1e-30):
        raise RuntimeError('Block field curvature lost Hermitian symmetry')
    matrix=(matrix+matrix.mH)*.5
    eigen,vectors=torch.linalg.eigh(matrix)
    maximum=float(eigen[-1]);minimum=float(eigen[0])
    if minimum < -maximum*1e-5:raise RuntimeError('Block field normal lost positive curvature')
    inverse=torch.where(eigen>maximum*1e-11,1/eigen.clamp_min(maximum*1e-11),0.)
    return vectors@(inverse[:,None]*(vectors.mH@right))


@torch.no_grad()
def update_block(operator,rhs,target,response,current,penalty,forward,adjoint,precondition,
                 maximum=32,tolerance=.003):
    def gradient(eq,rx):return operator.readout_adjoint(rx,response)+penalty*operator.adjoint(eq)
    full=gradient(rhs,target);den=gram(full,full).diagonal().real.clamp_min(1e-30)
    pre_full=adjoint(full);pre_den=gram(pre_full,pre_full).diagonal().real.clamp_min(1e-30)
    del full,pre_full
    eq=rhs-operator(current);rx=target-operator.readout(current,response)
    before=gram(rx,rx).diagonal().real+penalty*gram(eq,eq).diagonal().real
    right=adjoint(gradient(eq,rx));correction=torch.zeros_like(right);residual=right.clone()
    z=precondition(residual);direction=z.clone();rho=gram(residual,z);history=[]
    def normal(value):
        f=forward(value);return adjoint(gradient(operator(f),operator.readout(f,response)))
    for iteration in range(1,maximum+1):
        image=normal(direction);curvature=gram(direction,image)
        alpha=solve_spd(curvature,rho);correction.add_(mix(direction,alpha));residual.sub_(mix(image,alpha))
        row={'iteration':iteration,'recursive_preconditioned_global_normal_residual':
            (gram(residual,residual).diagonal().real/pre_den).sqrt().tolist()}
        if iteration%4==0 or iteration==maximum:
            candidate=current+forward(correction);er=rhs-operator(candidate);dr=target-operator.readout(candidate,response)
            grad=gradient(er,dr);rtrue=adjoint(grad)
            row.update(physical_global_normal_residual=(gram(grad,grad).diagonal().real/den).sqrt().tolist(),
                preconditioned_global_normal_residual=(gram(rtrue,rtrue).diagonal().real/pre_den).sqrt().tolist(),
                augmented_cost=(gram(dr,dr).diagonal().real+penalty*gram(er,er).diagonal().real).tolist())
            drift=(gram(rtrue-residual,rtrue-residual).diagonal().real/
                gram(rtrue,rtrue).diagonal().real.clamp_min(1e-30)).sqrt().max()
            row['residual_drift']=float(drift)
            converged=max(row['physical_global_normal_residual'])<=tolerance and max(row['preconditioned_global_normal_residual'])<=tolerance
            del candidate,er,dr,grad
            history.append(row)
            if converged:break
            if iteration%16==0 and float(drift)>.05:
                residual=rtrue;z=precondition(residual);direction=z.clone();rho=gram(residual,z)
                row['residual_replaced']=True;continue
        else:history.append(row)
        z=precondition(residual);new_rho=gram(residual,z)
        beta=solve_spd(rho,new_rho);direction=z+mix(direction,beta);rho=new_rho
    final=current+forward(correction);eq=rhs-operator(final);rx=target-operator.readout(final,response)
    grad=gradient(eq,rx);rtrue=adjoint(grad)
    physical=(gram(grad,grad).diagonal().real/den).sqrt().tolist()
    pre=(gram(rtrue,rtrue).diagonal().real/pre_den).sqrt().tolist()
    after=gram(rx,rx).diagonal().real+penalty*gram(eq,eq).diagonal().real
    if bool((after>before*(1+1e-4)+1e-20).any()):raise RuntimeError('Block field solve increased a shot objective')
    return final,{'iterations':len(history),'history':history,'physical_global_normal_residual':physical,
        'preconditioned_global_normal_residual':pre,'augmented_cost_before':before.tolist(),
        'augmented_cost_after':after.tolist(),'converged':max(physical)<=tolerance and max(pre)<=tolerance},eq,rx
