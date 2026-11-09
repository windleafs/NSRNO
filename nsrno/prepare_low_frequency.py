"""Convert completed paired RF to the lossless V5-compatible IQ representation."""
from argparse import ArgumentParser
from pathlib import Path
import json,time,os
import numpy as np
from .dataset import read_json,write_json
from .rf_representation import encode

def main():
    p=ArgumentParser(); p.add_argument('--raw',type=Path,default=Path('out/v5/low_frequency_3mhz_v1'))
    p.add_argument('--out',type=Path,default=Path('out/v5/forward_cache_3mhz_v1'))
    p.add_argument('--wait',action='store_true'); args=p.parse_args()
    cache=Path('out/v5/forward_cache_v1').resolve(); high=read_json(cache/'index.json')
    manifest=read_json(args.raw/'index.json'); args.out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); records=[]; pending=list(manifest['records'])
    while pending:
        for row in list(pending):
            path=args.raw/row['rf']; receipt=path.with_suffix('.json')
            if not receipt.exists() or not path.exists(): continue
            original=read_json(receipt)
            if not original['medium_exactly_matches_v5']: raise ValueError('Unpaired media')
            with np.load(path,allow_pickle=False) as data: iq,error=encode(data['rf']); trefs=data['source_tref_s'].tolist()
            out=args.out/(row['id']+'.iq.npy'); temporary=out.with_suffix('.partial.npy')
            with temporary.open('wb') as handle:
                np.save(handle,iq); handle.flush(); os.fsync(handle.fileno())
            temporary.replace(out)
            record={**row,'medium':str(cache/row['medium']),'iq':out.name,'source_tref_s':trefs,
                    'rf_representation_relative_rms':error,'raw_rf':str(path.resolve()),
                    'low_frequency_receipt':str(receipt.resolve())}
            write_json(args.out/(row['id']+'.json'),record); records.append(record); pending.remove(row)
            print(json.dumps({'prepared':len(records),'total':len(manifest['records']),'id':row['id'],'representation_error':error}),flush=True)
        if not pending: break
        if not args.wait: raise RuntimeError(f'{len(pending)} records still generating; use --wait')
        if time.monotonic()-start>14400: raise TimeoutError('Generation not complete after four hours')
        time.sleep(15)
    lookup={r['id']:r for r in records}; records=[lookup[r['id']] for r in manifest['records']]
    index={**high,'acquisition':manifest['acquisition'],'records':records,
           'paired_source_index':str((cache/'index.json').resolve()),'raw_index':str((args.raw/'index.json').resolve())}
    # Distinguish this cache provenance from the high-frequency source manifest.
    from hashlib import sha256
    index['source_sha256']=sha256((args.raw/'index.json').read_bytes()).hexdigest()
    write_json(args.out/'index.json',index)
    result=dict(records=len(records),training=sum(r['role']=='train' for r in records),
        validation=sum(r['role']=='validation' for r in records),
        max_rf_representation_relative_rms=max(r['rf_representation_relative_rms'] for r in records))
    write_json(args.out/'completion.json',result); print(json.dumps(result),flush=True)

if __name__=='__main__': main()
