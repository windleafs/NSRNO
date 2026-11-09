from pathlib import Path
import os,sys,time
root=Path('out/v5/conditioned_wave_v1'); started=time.monotonic()
while not all((root/f'teachers_complete_{j}.json').exists() for j in range(3)):
    if time.monotonic()-started>1800: raise TimeoutError('Teacher generation incomplete after 30 minutes')
    time.sleep(10)
print('All 432 reference fields complete; starting 240-step training.',flush=True)
os.execv(sys.executable,[sys.executable,'-m','nsrno.train_preconditioner','--steps','240'])
