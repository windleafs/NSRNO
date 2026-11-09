"""Run the identical post-hoc objective without per-process huge-page compaction."""
import ctypes
import runpy

if __name__ == '__main__':
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
        raise RuntimeError('Could not disable transparent huge pages for this process')
    runpy.run_module('nsrno.assess_cmpe_objective', run_name='__main__')
