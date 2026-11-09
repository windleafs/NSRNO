from pathlib import Path
import json,math
import numpy as np
out=Path('out/v5/forward_round8_dense_full'); rows=[]
for part in range(3): rows.extend(json.load(open(out/f'forward_part{part}.json')))
summary={}
for sid in ('train_000','train_200'):
    summary[sid]={}
    methods=('before','after','after64','converged') if sid=='train_200' else ('before','after')
    for method in methods:
        values=[r for r in rows if r['id']==sid and r['method']==method]
        bins={r['bin'] for r in values}
        if len(values)!=294 or bins!=set(range(241,535)): raise RuntimeError('Incomplete dense coverage '+sid+' '+method)
        band_energy=sum(r['target_sq'] for r in values); full_energy=values[0]['full_rf_target_energy']
        summary[sid][method]={'frequencies':294,'angles':11,'observations':3234,
            'rf_relative_rms':math.sqrt(sum(r['rf_error_sq'] for r in values)/sum(r['target_sq'] for r in values)),
            'full_rf_relative_rms':math.sqrt((sum(r['rf_error_sq'] for r in values)+max(0.,full_energy-band_energy))/full_energy),
            'retained_rf_energy_fraction':band_energy/full_energy,
            'equal_frequency_rf_rms':math.sqrt(np.mean([r['rf_relative_rms']**2 for r in values])),
            'pde_residual_rms':math.sqrt(np.mean([r['pde_residual']**2 for r in values])),
            'pde_residual_max':max(r['pde_residual'] for r in values),'seconds_total':sum(r['seconds'] for r in values)}
result={'summary':summary,'passed_for_reconstruction':summary['train_200']['converged']['full_rf_relative_rms']<=.05 and summary['train_200']['after64']['full_rf_relative_rms']<=.05,
    'accuracy_criterion':'both trained64-step output and converged output must have<=5-percent completeRF error on all294x11 observations of train_200',
    'note':'Neural32-step and converged physical output are reported separately; all294x11 are evaluated, including missing-band energy in full RF error.'}
(out/'forward_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if not result['passed_for_reconstruction']: raise RuntimeError('Forward accuracy insufficient; diagnose before reconstruction')
