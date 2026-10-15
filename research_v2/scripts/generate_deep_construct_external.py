import csv,hashlib,json,random,time,zlib
from pathlib import Path
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from generate_counterfactuals import candidate_masks,crem_core,DEFAULT_DB

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_v2/artifacts/analysis/deep_construct_20261009_v1'
RDLogger.DisableLog('rdApp.*')
selected=OUT/'selected_external_molecules.csv'
sample=pd.read_csv(selected)
selection_hash=hashlib.sha256(selected.read_bytes()).hexdigest()
generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
columns=['row_id','dataset','mask_id','mask_atoms','mask_size','mutant_smiles','parent_mutant_tanimoto','reaction_smarts','true_removed_atoms','outside_removed_atoms','retained_element_changes','true_removed_halogen','true_inserted_halogen','true_removed_heteroatom','true_inserted_heteroatom','elemental_accounting_valid']
started=time.monotonic();completed=0;scored=0;coverage=[];molecule_status=[]
with (OUT/'external_scored_mutants.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
    for r in sample.itertuples():
        if time.monotonic()-started>45*60:break
        mol=Chem.MolFromSmiles(r.canonical_smiles);fp=generator.GetFingerprint(mol)
        masks=candidate_masks(mol,1,8);mol_scored=0
        for mask_id,mask in enumerate(masks):
            random.seed(zlib.crc32(f'{r.row_id}:{mask}'.encode()))
            unique={};failure=''
            try:
                for value,rxn,product in crem_core.mutate_mol(mol,str(DEFAULT_DB),radius=1,min_size=1,max_size=8,min_inc=-2,max_inc=2,max_replacements=8,replace_ids=mask,symmetry_fixes=True,min_freq=2,return_rxn=True,return_mol=True):
                    mutant=Chem.MolFromSmiles(value)
                    if mutant is None or len(Chem.GetMolFrags(mutant))!=1:continue
                    canonical=Chem.MolToSmiles(mutant,isomericSmiles=True)
                    if canonical==r.canonical_smiles:continue
                    similarity=DataStructs.TanimotoSimilarity(fp,generator.GetFingerprint(mutant))
                    if similarity<.60:continue
                    retained={a.GetIntProp('react_atom_idx'):a for a in product.GetAtoms() if a.HasProp('react_atom_idx')}
                    removed=sorted(set(range(mol.GetNumAtoms()))-set(retained))
                    inserted=[a for a in product.GetAtoms() if not a.HasProp('react_atom_idx')]
                    changed=sum(mol.GetAtomWithIdx(i).GetAtomicNum()!=a.GetAtomicNum() for i,a in retained.items())
                    trace={}
                    valid=True
                    for name,elements in [('halogen',{9,17,35,53}),('heteroatom',{7,8,15,16})]:
                        lost=sum(mol.GetAtomWithIdx(i).GetAtomicNum() in elements for i in removed)
                        gained=sum(a.GetAtomicNum() in elements for a in inserted)
                        retained_delta=sum(int(a.GetAtomicNum() in elements)-int(mol.GetAtomWithIdx(i).GetAtomicNum() in elements) for i,a in retained.items())
                        delta=sum(a.GetAtomicNum() in elements for a in mutant.GetAtoms())-sum(a.GetAtomicNum() in elements for a in mol.GetAtoms())
                        valid=valid and delta==gained-lost+retained_delta
                        trace[f'true_removed_{name}']=lost;trace[f'true_inserted_{name}']=gained
                    unique[canonical]={'row_id':r.row_id,'dataset':r.dataset,'mask_id':mask_id,'mask_atoms':json.dumps(mask),'mask_size':len(mask),'mutant_smiles':canonical,'parent_mutant_tanimoto':similarity,'reaction_smarts':rxn,'true_removed_atoms':json.dumps(removed),'outside_removed_atoms':len(set(removed)-set(mask)),'retained_element_changes':changed,'elemental_accounting_valid':valid,**trace}
            except Exception as error:
                unique={};failure=f'{type(error).__name__}:{error}'
            rows=sorted(unique.values(),key=lambda x:hashlib.sha256(f"{r.row_id}:{mask_id}:{x['mutant_smiles']}".encode()).hexdigest())[:4]
            writer.writerows(rows);handle.flush();scored+=len(rows);mol_scored+=len(rows)
            coverage.append({'row_id':r.row_id,'mask_id':mask_id,'mask_atoms':json.dumps(mask),'mask_size':len(mask),'supported_unique_mutants':len(unique),'retained_mutants':len(rows),'failure':failure})
        completed+=1;molecule_status.append({'row_id':r.row_id,'candidate_masks':len(masks),'retained_mutants':mol_scored,'completed':True})
        if completed%16==0:print(f'External generation {completed}/256; retained mutants {scored}; elapsed {time.monotonic()-started:.1f}s',flush=True)
for r in sample.iloc[completed:].itertuples():molecule_status.append({'row_id':r.row_id,'candidate_masks':None,'retained_mutants':0,'completed':False})
pd.DataFrame(coverage).to_csv(OUT/'external_mask_coverage.csv',index=False)
pd.DataFrame(molecule_status).to_csv(OUT/'external_generation_molecules.csv',index=False)
assert hashlib.sha256(selected.read_bytes()).hexdigest()==selection_hash
record={'status':'completed' if completed==256 else 'partial_resource_limit','selected':256,'completed':completed,'retained_mutants':scored,'selection_sha256_before_generation':selection_hash,'failures':sum(bool(x['failure']) for x in coverage),'unsupported_molecules':sum(x['completed'] and not x['retained_mutants'] for x in molecule_status),'elapsed_seconds':time.monotonic()-started,'native_trace':'CReM return_rxn/return_mol; RDKit react_atom_idx','original_generator_reference':'research_v2/scripts/generate_counterfactuals.py','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'interpretation_amendment_sha256':hashlib.sha256((OUT/'interpretation_amendment.md').read_bytes()).hexdigest()}
(OUT/'external_generation_summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2),flush=True)
