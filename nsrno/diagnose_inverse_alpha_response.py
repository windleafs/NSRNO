"""Inspect existing independent-alpha input/output changes on CPU only."""
from pathlib import Path
import argparse
import json
import numpy as np
from .native_gn_json import atomic


def rms(x):return float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/inverse_angular_spectral_v1/index.json'))
    p.add_argument('--model',type=Path,default=Path('out/v5/inverse_angular_spectral_uno_v1'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('Read the existing diagnosis instead of repeating it')
    index=json.loads(args.index.read_text());groups={}
    for row in index['records']:
        if row['kind']=='alpha' and 'pair_group' in row:groups.setdefault(row['pair_group'],[]).append(row)
    z=np.arange(216)*.0002;reports=[]
    for name,members in groups.items():
        if len(members)!=2:continue
        minus,plus=sorted(members,key=lambda r:r['pair_sign'])
        target0=np.load(minus['label']);target1=np.load(plus['label'])
        target_delta=[rms(target1[j,5:]-target0[j,5:]) for j in range(3)]
        item={'pair_group':name,'role':minus['role'],'case':minus['case'],
            'target_difference_rms_c_rho_alpha':target_delta,'sources':{}}
        for source in sorted(set(minus['sources'])&set(plus['sources'])):
            q0=np.load(minus['sources'][source]).astype(np.float64)
            q1=np.load(plus['sources'][source]).astype(np.float64)
            rf_relative=rms(q1-q0)/max(rms(.5*(q0+q1)),1e-30)
            f0=np.load(minus['features'][source]).astype(np.float64).reshape(9,8,216,192)
            f1=np.load(plus['features'][source]).astype(np.float64).reshape(9,8,216,192)
            depths={}
            for low,high in [(1,5),(5,12),(12,25),(25,43.01)]:
                mask=(z>=low/1000)&(z<high/1000)
                values=[]
                for j in range(6):
                    a=f0[:,j,mask];b=f1[:,j,mask]
                    values.append({'feature_type':j,'mean_RMS':rms(.5*(a+b)),
                        'difference_RMS':rms(b-a),'unchanged_fraction':float(np.mean(a==b))})
                depths[f'{low}-{min(high,43):g}']=values
            item['sources'][source]={'observed_IQ_relative_difference':rf_relative,
                'IQ_relative_difference_per_unit_target_alpha_RMS':rf_relative/max(target_delta[2],1e-30),
                'feature_changes_by_depth_mm':depths}
        p0=args.model/'evaluation'/f"{minus['id']}.prediction.npy"
        p1=args.model/'evaluation'/f"{plus['id']}.prediction.npy"
        if p0.exists() and p1.exists():
            delta=np.load(p1)[:,5:].astype(np.float64)-np.load(p0)[:,5:]
            predicted=[rms(v) for v in delta]
            item['predicted_difference_rms_c_rho_alpha']=predicted
            item['alpha_RMS_response_gain']=predicted[2]/max(target_delta[2],1e-30)
        reports.append(item)
    summary={}
    for role in sorted({r['role'] for r in reports}):
        selected=[r for r in reports if r['role']==role]
        source_summary={}
        for source in ['hf','lf']:
            rows=[r['sources'][source] for r in selected if source in r['sources']]
            if rows:source_summary[source]={'pairs':len(rows),
                'mean_IQ_relative_difference':float(np.mean([r['observed_IQ_relative_difference'] for r in rows])),
                'mean_IQ_difference_per_unit_target_alpha_RMS':float(np.mean([r['IQ_relative_difference_per_unit_target_alpha_RMS'] for r in rows]))}
        gains=[r['alpha_RMS_response_gain'] for r in selected if 'alpha_RMS_response_gain' in r]
        summary[role]={'pairs':len(selected),'sources':source_summary,
            'mean_alpha_RMS_response_gain':float(np.mean(gains)) if gains else None}
    report={'CPU_only':True,'patient_wave_solves':0,'new_inference_passes':0,
        'source_index':str(args.index),'feature_types':['mean_real','mean_imag','angular_real','angular_imag','log1p_RMS','coherence'],
        'summary':summary,'pairs':reports}
    atomic(args.out,report)
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'}),flush=True)


if __name__=='__main__':main()
