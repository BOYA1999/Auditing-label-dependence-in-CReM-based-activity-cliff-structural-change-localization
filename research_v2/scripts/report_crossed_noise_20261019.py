import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_v2/artifacts/experiment/crossed_20261019'
OUT = ROOT / 'research_v2/artifacts/analysis/extension_20261019/noise'


def main():
    summary = pd.read_csv(OUT / 'summary.csv')
    frame = pd.read_csv(OUT / 'anchor_paired_distances.csv.gz',usecols=['scenario','domain','anchor_seed','row_id','dataset','component_id','n_atoms','matched_distance','observed_ratio','permuted_ratio','observed_zero_denominator','permuted_zero_denominator','observed_undefined_reference','permuted_undefined_reference'])
    table = []
    for scenario in summary.scenario.unique():
        selected = frame[frame.domain.eq('supported') & frame.scenario.eq(scenario)]
        if scenario.startswith('anchor_seed_'):
            selected = frame[frame.domain.eq('supported') & frame.scenario.eq('full') & frame.anchor_seed.eq(int(scenario.split('_')[-1]))]
        values = summary[summary.domain.eq('supported') & summary.scenario.eq(scenario)].set_index('metric')
        for reference in ['observed','permuted']:
            row = {'scenario':scenario,'initialization_noise_reference':reference,'matched_label_distance':values.loc['matched_distance','estimate'],'reference_distance':values.loc[reference+'_noise','estimate'],'repeated_anchor_permutation_rows':len(selected),'matched_distance_undefined':int(selected.matched_distance.isna().sum()),'ratio_defined_rows':int(selected[reference+'_ratio'].notna().sum()),'zero_denominator_rows':int(selected[reference+'_zero_denominator'].sum())}
            row['ratio_undefined_rows'] = int(selected[reference+'_ratio'].isna().sum())
            row['reference_undefined_rows'] = int(selected[reference+'_undefined_reference'].sum())
            for metric in ['excess','ratio']:
                for field in ['estimate','ci_low','ci_high']:
                    row[metric+'_'+field]=float(values.loc[reference+'_'+metric,field])
            table.append(row)
    pd.DataFrame(table).to_csv(OUT / 'Table_S48_noise_summary.csv',index=False)
    shutil.copy2(RUN/'backend_difference_summary.csv',OUT/'Table_S52_backend_sensitivity.csv')
    molecules=pd.read_csv(OUT/'molecule_configuration_summary.csv.gz')
    molecules=molecules[molecules.scenario.eq('full') & molecules.domain.eq('supported')]
    counts=molecules.groupby(['dataset','component_id']).matched_distance.count()
    counts[counts.eq(0)].rename('defined_molecule_configuration_rows').to_csv(OUT/'unsupported_correlation_components.csv')
    eligibility=[]
    for path in sorted((RUN/'atom_scores').glob('*.npz')):
        dataset,rep,pred=path.stem.split('--');d=np.load(path)
        for j,ident in enumerate(d['row_ids']):
            a,b=d['offsets'][j:j+2];mask=d['supported_flat'][a:b];n=int(mask.sum())
            obs=d['observed_scores'][:,a:b][:,mask];perm=d['scores'][:,:,a:b][:,:,mask]
            eligibility.append({'dataset':dataset,'row_id':str(ident),'representation':rep,'predictor':pred,'supported_atoms':n,'insufficient_atoms':n<2,'observed_constant_vectors':int(np.sum(np.ptp(obs,axis=-1)==0)) if n>=2 else 0,'permuted_constant_vectors':int(np.sum(np.ptp(perm,axis=-1)==0)) if n>=2 else 0})
    pd.DataFrame(eligibility).to_csv(OUT/'map_eligibility.csv',index=False)
    costs=[]
    for name,path in [('720 new fits and cache validation',RUN/'training_manifest.json'),('source-compatible atom export',RUN/'atom_export_manifest_source.json'),('uniform-backend atom export',RUN/'atom_export_manifest.json'),('noise analysis',OUT/'manifest.json')]:
        costs.append({'stage':name,'elapsed_seconds':json.loads(path.read_text())['elapsed_seconds']})
    pd.DataFrame(costs).to_csv(OUT/'recorded_run_costs.csv',index=False)
    files=['Table_S48_noise_summary.csv','Table_S52_backend_sensitivity.csv','map_eligibility.csv','unsupported_correlation_components.csv','recorded_run_costs.csv','independent_verification.json']
    (OUT/'report_manifest.json').write_text(json.dumps({'status':'pass','files':{f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in files},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':
    main()
