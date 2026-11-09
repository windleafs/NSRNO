"""One independent dense check of source-feature and residual combination."""
from pathlib import Path
import argparse
import json
import numpy as np
from .native_gn_spectral import SpectralDataMetric
from .native_gn_enriched_metric import SourceEnrichedMetric


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    rng=np.random.default_rng(1794005)
    shape=(3,82,4)
    mask=np.broadcast_to(np.arange(82)[:,None]>=80,shape[1:])
    active=np.broadcast_to(mask,shape)
    n=int(active.sum())
    def pack(x):
        out=np.zeros(shape)
        out[active]=x
        return out
    F=rng.normal(size=(66,n))/np.sqrt(n)
    H=F.T@F
    residual=rng.normal(size=66)
    gradient=F.T@residual
    energy=float(residual@residual)
    D=rng.normal(size=(n,5));Y=H@D
    pairs=[(pack(D[:,j]),pack(Y[:,j])) for j in range(5)]
    features=[]
    for j in [3,4]:
        for shot in range(22):
            block=F[3*shot:3*shot+3]
            features.append({'name':f'shot{shot:02d}','direction_index':j,
                'vector':pack(block.T@(block@D[:,j]))})
    symbol=np.broadcast_to(np.diag([.8,1.2,.5]).astype(np.complex128),(82,3,3,3)).copy()
    prior=.03
    model=SourceEnrichedMetric(SpectralDataMetric(symbol,mask),mask,
        lambda x:prior*x*mask,np.full(shape,prior)*mask,pairs,
        pack(gradient),energy,features)
    basis=[pack(np.eye(n)[:,j]) for j in range(n)]
    matrix=np.column_stack([model.apply(e)[active] for e in basis])
    feature_matrix=sum((np.outer(v[active],v[active]) for v in model.positive),np.zeros((n,n)))
    source_matrix=sum((np.outer(v[active],v[active]) for v in model.source_only_vectors),np.zeros((n,n)))
    aggregate=Y@np.linalg.solve(D.T@Y,Y.T)
    source_lower=float(np.linalg.eigvalsh(H-aggregate-source_matrix).min())
    feature_upper=float(np.linalg.eigvalsh(2*H-feature_matrix).min())
    secant=float(np.linalg.norm(matrix@D-(H+prior*np.eye(n))@D)/np.linalg.norm(Y))
    data=matrix-prior*np.eye(n)
    gradient_bound=float(np.linalg.eigvalsh(data-np.outer(gradient,gradient)/energy).min())
    diagonal=float(np.max(abs(np.diag(matrix)-model.diagonal[active])))
    assert secant<1e-10 and source_lower>-1e-10 and feature_upper>-1e-10
    assert gradient_bound>-1e-10 and diagonal<1e-10 and np.linalg.eigvalsh(matrix).min()>0
    d=pack(rng.normal(size=n))
    update=model.update(d,pack((H+prior*np.eye(n))@d[active]))
    assert update['last_secant_relative_error']<1e-10
    report={'kind':'native_source_plus_residual_sketch_independent_CPU_dense_check',
        'all_saved_secant_relative_error':secant,'source_lower_bound_margin':source_lower,
        'feature_upper_2H_margin':feature_upper,'gradient_Cauchy_bound_margin':gradient_bound,
        'diagonal_absolute_error':diagonal,'new_secant_relative_error':update['last_secant_relative_error'],
        'source_modes_retained':model.report['source_conditional_modes'],
        'GPU_used':False,'native_patient_waves':0,'passed':True}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
