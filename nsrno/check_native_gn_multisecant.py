"""One independent dense CPU check of the spectral multisecant algebra."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .native_gn_spectral import SpectralDataMetric
from .native_gn_multisecant import MultisecantMetric


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    started=time.time()
    rng=np.random.default_rng(1794003)
    shape=(3,82,4)
    mask=np.broadcast_to(np.arange(shape[1])[:,None]>=80,shape[1:])
    active=np.broadcast_to(mask,shape)
    n=int(active.sum())
    def pack(x):
        value=np.zeros(shape)
        value[active]=x
        return value
    kz=2*np.pi*np.arange(82)/82
    kx=2*np.pi*np.arange(3)/4
    factor=1+.1*np.cos(kz[:,None])+.2*np.cos(kx[None,:])
    matrix=np.array([[1.,.1,.02],[.1,.6,.03],[.02,.03,1.5]])
    symbol=factor[...,None,None]*matrix[None,None,:,:].astype(np.complex128)
    symbol[...,0,1]+=.08j*np.sin(kz[:,None])
    symbol[...,1,0]-=.08j*np.sin(kz[:,None])
    base=SpectralDataMetric(symbol,mask)
    basis=[pack(np.eye(n)[:,j]) for j in range(n)]
    A=np.column_stack([base.apply(e)[active] for e in basis])
    q=rng.normal(size=(n,n))
    H=q@q.T/n+.2*np.eye(n)
    directions=rng.normal(size=(n,5))
    images=H@directions
    prior=.07
    pairs=[(pack(directions[:,j]),pack(images[:,j])) for j in range(5)]
    model=MultisecantMetric(base,mask,lambda x:prior*x*mask,
        np.full(shape,prior)*mask,pairs)
    actual=np.column_stack([model.apply(e)[active] for e in basis])
    AD=A@directions
    oracle=A-AD@np.linalg.solve(directions.T@AD,AD.T)+images@np.linalg.solve(
        directions.T@images,images.T)+prior*np.eye(n)
    relative=float(np.linalg.norm(actual-oracle)/np.linalg.norm(oracle))
    secant=float(np.linalg.norm(actual@directions-(H+prior*np.eye(n))@directions)/np.linalg.norm(H@directions))
    diagonal=float(np.max(abs(np.diag(actual)-model.diagonal[active])))
    minimum=float(np.linalg.eigvalsh(actual).min())
    assert relative<1e-12 and secant<1e-12 and diagonal<1e-12 and minimum>0
    d=pack(rng.normal(size=n))
    update=model.update(d,pack((H+prior*np.eye(n))@d[active]))
    final=np.column_stack([model.apply(e)[active] for e in basis])
    assert np.linalg.eigvalsh(final).min()>0
    assert update['last_secant_relative_error']<1e-12
    report={'kind':'independent_dense_CPU_check_for_spectral_multisecant_metric',
        'active_dimension':n,'native_patient_waves':0,'GPU_used':False,
        'dense_formula_relative_error':relative,'all_saved_secant_relative_error':secant,
        'diagonal_absolute_error':diagonal,'minimum_eigenvalue':minimum,
        'new_secant_relative_error':update['last_secant_relative_error'],
        'passed':True,'seconds':time.time()-started}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
