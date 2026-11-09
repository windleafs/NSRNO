"""Independent dense CPU proof check of the residual-augmented metric."""
from pathlib import Path
import argparse
import json
import numpy as np
from .native_gn_spectral import SpectralDataMetric
from .native_gn_residual_metric import ResidualAwareMetric


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    rng=np.random.default_rng(1794004)
    shape=(3,82,4)
    mask=np.broadcast_to(np.arange(82)[:,None]>=80,shape[1:])
    active=np.broadcast_to(mask,shape)
    n=int(active.sum())
    def pack(x):
        result=np.zeros(shape)
        result[active]=x
        return result
    F=rng.normal(size=(40,n))/np.sqrt(n)
    H=F.T@F
    residual=rng.normal(size=40)
    gradient=F.T@residual
    energy=float(residual@residual)
    D=rng.normal(size=(n,5))
    Y=H@D
    symbol=np.broadcast_to(np.diag([.8,1.2,.5]).astype(np.complex128),(82,3,3,3)).copy()
    base=SpectralDataMetric(symbol,mask)
    prior=.03
    pairs=[(pack(D[:,j]),pack(Y[:,j])) for j in range(5)]
    model=ResidualAwareMetric(base,mask,lambda x:prior*x*mask,
        np.full(shape,prior)*mask,pairs,pack(gradient),energy)
    basis=[pack(np.eye(n)[:,j]) for j in range(n)]
    matrix=np.column_stack([model.apply(e)[active] for e in basis])
    data=matrix-prior*np.eye(n)
    secant=float(np.linalg.norm(matrix@D-(H+prior*np.eye(n))@D)/np.linalg.norm(Y))
    cauchy_min=float(np.linalg.eigvalsh(data-np.outer(gradient,gradient)/energy).min())
    minimum_data_loss=.5*energy-.5*float(gradient@np.linalg.solve(data,gradient))
    assert secant<1e-12 and cauchy_min>=-1e-12 and minimum_data_loss>=-1e-12
    assert np.linalg.eigvalsh(matrix).min()>0
    d=pack(rng.normal(size=n))
    update=model.update(d,pack((H+prior*np.eye(n))@d[active]))
    assert update['last_secant_relative_error']<1e-12
    report={'kind':'residual_augmented_native_metric_CPU_dense_check',
        'all_saved_secant_relative_error':secant,
        'minimum_eigenvalue_above_gradient_Cauchy_bound':cauchy_min,
        'minimum_linearized_data_loss':minimum_data_loss,
        'new_secant_relative_error':update['last_secant_relative_error'],
        'GPU_used':False,'patient_waves':0,'passed':True}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
