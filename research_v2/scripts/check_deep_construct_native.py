import json,random,zlib
from pathlib import Path
import pandas as pd
from rdkit import Chem,RDLogger
import generate_counterfactuals as g

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_v2/artifacts/analysis/deep_construct_20261009_v1'
RDLogger.DisableLog('rdApp.*')
cache=pd.read_csv(OUT/'external_scored_mutants.csv')
sample=pd.read_csv(OUT/'selected_external_molecules.csv').set_index('row_id')
records=[]
cases=cache[cache.outside_removed_atoms.gt(0)].drop_duplicates('row_id').head(8)
for r in cases.itertuples():
    smiles=sample.loc[r.row_id,'canonical_smiles'];mask=tuple(json.loads(r.mask_atoms))
    collected=[]
    for method,fragmenter in [('optimized',g.fragment_mol_fast),('upstream',g.ORIGINAL_FRAGMENT_MOL)]:
        g.crem_core.__fragment_mol=fragmenter
        mol=Chem.MolFromSmiles(smiles);random.seed(zlib.crc32(f'{r.row_id}:{mask}'.encode()))
        products={}
        for value,product in g.crem_core.mutate_mol(mol,str(g.DEFAULT_DB),radius=1,min_size=1,max_size=8,min_inc=-2,max_inc=2,max_replacements=8,replace_ids=mask,symmetry_fixes=True,min_freq=2,return_mol=True):
            retained={a.GetIntProp('react_atom_idx') for a in product.GetAtoms() if a.HasProp('react_atom_idx')}
            products[Chem.MolToSmiles(Chem.MolFromSmiles(value),isomericSmiles=True)]=sorted(set(range(mol.GetNumAtoms()))-retained)
        collected.append(products)
        records.append({'row_id':r.row_id,'mask_id':r.mask_id,'method':method,'target_product_reproduced':r.mutant_smiles in products,'target_removed_atoms':products.get(r.mutant_smiles),'allowed_mask':list(mask),'generated_unique':len(products)})
    records[-1]['optimized_upstream_same_smiles']=set(collected[0])==set(collected[1])
    records[-1]['optimized_upstream_same_trace']=collected[0]==collected[1]
(OUT/'upstream_trace_check.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
print(json.dumps(records,indent=2),flush=True)
