"""One CPU algebra check against independent dense split-update matrices."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .native_gn_source_metric import SourceResolvedBFGS
from .native_gn_coordinates import dot


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    started=time.time()
    rng=np.random.default_rng(1794001)
    shape=(3,82,4)
    mask=np.broadcast_to(np.arange(shape[1])[:,None]>=80,shape[1:])
    active=np.broadcast_to(mask,shape)
    n=int(active.sum())
    def pack(x):
        out=np.zeros(shape)
        out[active]=x
        return out
    base=np.diag(np.linspace(.4,1.2,n))
    model=SourceResolvedBFGS(lambda x:pack(base@x[active]),
        pack(np.diag(base))+~mask,mask)
    dense=base.copy()
    basis=[pack(np.eye(n)[:,j]) for j in range(n)]
    rows=[]
    for iteration in range(3):
        d=pack(rng.normal(size=n))
        components=[]
        positive_dense=np.zeros((n,n))
        y=np.zeros(shape)
        for shot in range(22):
            a=rng.normal(size=(n,4))
            H=a@a.T/(n*22)+np.eye(n)*.03/22
            vector=pack(H@d[active]).astype(np.float32)
            components.append({'name':f'shot_{shot:02d}','vector':vector})
            y+=vector
            dense_vector=vector[active].astype(np.float64)
            positive_dense+=np.outer(dense_vector,dense_vector)/dot(d,vector)
        prior=pack(.1*d[active])
        components.append({'name':'regularizer','vector':prior})
        y+=prior
        positive_dense+=np.outer(prior[active],prior[active])/dot(d,prior)
        v=dense@d[active]
        common=dense-np.outer(v,v)/(d[active]@v)
        aggregate=common+np.outer(y[active],y[active])/dot(d,y)
        expected=common+positive_dense
        update=model.update(d,y,components)
        actual=np.column_stack([model.apply(e)[active] for e in basis])
        relative=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
        minimum=float(np.linalg.eigvalsh((actual+actual.T)/2).min())
        gain=float(np.linalg.eigvalsh((actual-aggregate+(actual-aggregate).T)/2).min())
        assert relative<1e-12
        assert minimum>0 and gain>-1e-12*np.linalg.norm(actual)
        assert update['last_secant_relative_error']<1e-10
        assert np.max(abs(actual-actual.T))<1e-12
        rows.append({'iteration':iteration+1,'dense_relative_error':relative,
            'minimum_eigenvalue':minimum,'minimum_eigenvalue_of_split_gain':gain,
            'secant_relative_error':update['last_secant_relative_error']})
        dense=actual
    report={'kind':'CPU_dense_oracle_for_new_source_resolved_metric',
        'patient_wave_solves':0,'GPU_used':False,'native_data_read':False,
        'active_dimension':n,'per_update_data_components':22,'updates':rows,
        'passed':True,'seconds':time.time()-started}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
