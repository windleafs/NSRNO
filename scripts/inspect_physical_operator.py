from pathlib import Path
from nsrno.fullwave import initialize
gen=initialize(); case=gen.geometry_case(); solver=gen.solver_for(case,case['maps'],gen.absorption_model(case))
Path('out/v5/wave_diagnosis_v1/physical_operator.c').write_text(str(solver.op))
