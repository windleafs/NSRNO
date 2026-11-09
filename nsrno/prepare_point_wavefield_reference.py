"""Prepare the same reference generation for existing 90-us point data."""
from pathlib import Path
import ast,hashlib,json


def main():
    root=Path('nsrno');name='generate_lf_wavefield_reference.py'
    raw=(root/name).read_bytes()
    record=json.loads(Path('out/v5/conditioned_lf_full_v1/run.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=record['sources_sha256'][name]:
        raise RuntimeError('Preserve the actual LF generation source')
    source=raw.decode()
    def once(old,new):
        nonlocal source
        if source.count(old)!=1:raise RuntimeError('Unexpected source context: '+old)
        source=source.replace(old,new,1)
    once('from .native_capture import capture','from .native_capture_normalized import capture')
    once("Path('out/v5/conditioned_lf_full_v1')","Path('out/v5/conditioned_point_full_v1')")
    once("TimeAcquisition('lf','train_200'","TimeAcquisition('point3','train_200'")
    once('bins=np.arange(91,271)','bins=np.arange(136,406)')
    source=source.replace('/2401','/3601')
    once("    low=json.loads(Path('out/v5/low_frequency_3mhz_v1/index.json').read_text())",
         "    low=json.loads(Path('out/v5/low_frequency_3mhz_v1/index.json').read_text())\n    point_metadata=json.loads(Path('out/v5/point_tx_3mhz_v1/index.json').read_text())")
    once("**source_index['acquisition']","**point_metadata['acquisition']")
    once("    directory=args.out/'reference';directory.mkdir(exist_ok=True)",
         "    spec.pop('iq_shape',None)\n    spec['source']='point3'\n    directory=args.out/'reference';directory.mkdir(exist_ok=True)")
    once("'all_angles':11","'all_transmitters':11")
    once("'frequencies':180","'frequencies':270")
    once("'native_capture.py',Path(__file__).name]","'native_capture.py','native_capture_normalized.py',Path(__file__).name]")
    once("for r in source_index['records']","for r in point_metadata['records']")
    once("'source':'lf'","'source':'point3'")
    once("'resolved through nsrno.v5_observations.plane_observation, no material arrays'",
         "'existing point_tx_3mhz_v1 RF, no material arrays'")
    ast.parse(source)
    path=root/'generate_point_wavefield_reference.py'
    if path.exists():raise FileExistsError('Preserve previous point reference generator')
    path.write_text(source)
    print(path,hashlib.sha256(source.encode()).hexdigest())


if __name__=='__main__':main()
