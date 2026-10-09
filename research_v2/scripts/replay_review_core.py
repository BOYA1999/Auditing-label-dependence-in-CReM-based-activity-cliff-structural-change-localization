import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,'-B',str(ROOT/'research_v2/scripts/analyze_review_revision.py')],cwd=ROOT,check=True)
p=ROOT/'research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008'
import pandas as pd
rows=pd.read_csv(p/'primary_summary.csv')
checks={}
for name,expected in [('raw_nap',-0.005232719803153247),('pooled_exact_nap',-0.00879496753922941),('endpoint_exact_nap',-0.0069910516781531915)]:
 r=rows[rows.rationale.eq('boundary_extended')&rows.denominator.eq('supported')&rows.metric.eq(name)].iloc[0]
 checks[name]=bool(abs(r.estimate-expected)<1e-12 and r.datasets==27 and r.components==617)
validation=json.loads((p/'validation.json').read_text())
checks['endpoint_null_exhaustive']=validation['endpoint_null_exact_expectation']['max_absolute_error']<1e-12
checks['native_mapping_no_hidden_recovery']=validation['mapping']['eligible_pairs']==611 and validation['mapping']['native_right_misses']==6
checks['matched_generator_identity']=validation['denominator_identity_max_error']<1e-12
report={'status':'pass' if all(checks.values()) else 'failure','checks':checks,
 'scope':'Core statistics recomputed from restored frozen atom scores, pairs, masks and selected mutants; not a model refit or independent biological validation.'}
(ROOT/'core_replay_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert all(checks.values())
