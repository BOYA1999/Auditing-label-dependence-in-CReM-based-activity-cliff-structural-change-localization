import hashlib,heapq,json,urllib.request,urllib.parse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem,RDLogger

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_v2/artifacts/analysis/deep_construct_20261009_v1'
RDLogger.DisableLog('rdApp.*')
revision='e86ec35687b0c0ebd1ecc151610a08224d8cf851'
name='validation_datasets/Pattern 1/validation_molecules.parquet'
url=f'https://huggingface.co/datasets/dmtsh/wombat-smiles/resolve/{revision}/'+urllib.parse.quote(name,safe='/')
file=OUT/'external_source.parquet'
tree=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/dmtsh/wombat-smiles/tree/{revision}?recursive=true',timeout=120))
metadata=next(x for x in tree if x['path']==name)
expected=metadata['lfs']['oid']
if not file.is_file():
    print('Downloading pinned external source',flush=True)
    with urllib.request.urlopen(url,timeout=120) as source,file.open('wb') as target:
        while chunk:=source.read(1024*1024):target.write(chunk)
assert file.stat().st_size==157820200
assert hashlib.sha256(file.read_bytes()).hexdigest()==expected
print('External source hash verified',flush=True)
schema=pq.read_schema(file)
column=next(n for n in schema.names if n.lower() in {'smiles','canonical_smiles'})
raw=pq.read_table(file,columns=[column]).column(column).to_pylist()
excluded=set()
benchmark_files=list((ROOT/'external/MoleculeACE/MoleculeACE/Data/benchmark_data').glob('*.csv'))
assert len(benchmark_files)==30
for source in benchmark_files:
    frame=pd.read_csv(source)
    column2=next(n for n in frame.columns if n.lower() in {'smiles','canonical_smiles'})
    for value in frame[column2].dropna():
        mol=Chem.MolFromSmiles(value)
        if mol:excluded.add(Chem.MolToSmiles(mol,isomericSmiles=True))
assert excluded
print('Ranking source strings before eligibility checks',flush=True)
ranked=heapq.nsmallest(4096,set(raw),key=lambda s:hashlib.sha256(('deep-construct-20261009:'+s).encode()).hexdigest())
selected=[];seen=set()
for value in ranked:
    mol=Chem.MolFromSmiles(value)
    if mol is None or len(Chem.GetMolFrags(mol))!=1 or not 8<=mol.GetNumHeavyAtoms()<=60:continue
    if any(a.GetAtomicNum() not in {6,7,8,9,15,16,17,35,53} for a in mol.GetAtoms()):continue
    smiles=Chem.MolToSmiles(mol,isomericSmiles=True)
    if smiles in excluded or smiles in seen:continue
    key=hashlib.sha256(('deep-construct-20261009:'+value).encode()).hexdigest()
    selected.append((key,smiles,value));seen.add(smiles)
    if len(selected)==256:break
assert len(selected)==256
rows=[{'row_id':f'WOMBAT:{i}','dataset':'WOMBAT_Pattern1_source','molecule_index':i,'canonical_smiles':s,'source_smiles':v,'component_id':i,'component_split':'test','selection_hash':k} for i,(k,s,v) in enumerate(selected)]
pd.DataFrame(rows).to_csv(OUT/'selected_external_molecules.csv',index=False)
sources=[ROOT/'research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1/prepared/molecules.csv',ROOT/'research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1/prepared/sentinel_pairs.csv',ROOT/'research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1/full/counterfactuals/scored_mutants.csv',ROOT/'research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1/full/crem_db/replacement_library.db',OUT/'protocol.md',OUT/'selected_external_molecules.csv']
record={'dataset_revision':revision,'source_url':url,'source_bytes':file.stat().st_size,'source_sha256':expected,'source_rows':len(raw),'source_column':column,'dataset_license':'cc0-1.0','excluded_benchmark_structures':len(excluded),'selected':256,'canonicalization_changed_selected_strings':sum(s!=v for _,s,v in selected),'selection_uses_localization':False,'selection_rule':'First 256 eligible distinct current-canonical structures in global SHA256 rank of distinct source SMILES strings; eligibility checked within first 4096; no generation outcomes used.','files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(OUT/'input_provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ['source_rows','excluded_benchmark_structures','selected']}),flush=True)
