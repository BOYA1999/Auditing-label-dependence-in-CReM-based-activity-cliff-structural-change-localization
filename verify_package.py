import hashlib,json,re
from pathlib import Path
root=Path(__file__).resolve().parent
entries=[line.split('  ',1) for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines()]
errors=[]
for expected,name in entries:
 p=(root/name).resolve()
 if not p.is_relative_to(root) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append(name)
expected_set={name for _,name in entries}
extension=json.loads((root/'EXTENSION_INPUTS_MANIFEST.json').read_text(encoding='utf-8'))
core=json.loads((root/'CORE_INPUTS_MANIFEST.json').read_text(encoding='utf-8'))
package=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
for name,item in {**core,**extension}.items():
 p=root/name
 if not p.is_file() or p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:errors.append('core:'+name)
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}
allowed=expected_set|set(core)|set(extension)|{'MANIFEST.sha256','CORE_INPUTS_MANIFEST.json','CORE_INPUTS_README.md','core_replay_result.json','rereview_replay_result.json','research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008/count_oracle_matched_summary.csv'}
if actual-allowed or expected_set-actual:errors.append('unexpected inventory:'+str(sorted(actual^allowed)))
for name in expected_set:
 p=root/name
 if p.suffix.lower() in {'.docx','.doc','.tex','.pptx','.xlsx','.pem','.key'}:errors.append('prohibited artifact:'+name)
 if p.suffix.lower() in {'.pdf','.png','.tiff','.tif','.svg'} and name not in package['scientific_figures']:errors.append('unlisted scientific figure:'+name)
 if p.suffix.lower() in {'.py','.md','.json','.txt','.csv'}:
  s=p.read_text(encoding='utf-8',errors='replace')
  if re.search(r'(?i)[A-Z]:[\\/]+(?:Users|software)[\\/]+',s):errors.append('local path:'+name)
if len(list((root/'figure_source_data').glob('*.csv')))!=package['figure_sources']:errors.append('figure count')
tables=root/'research_v2/artifacts/analysis/xai_sanity_cliffs_supplement_tables_v1'
if len(list(tables.glob('*.csv')))!=package['supplementary_tables']:errors.append('table count')
for item in json.loads((tables/'manifest.json').read_text(encoding='utf-8'))['tables']:
 if hashlib.sha256((tables/item['file']).read_bytes()).hexdigest()!=item['sha256']:errors.append('table hash:'+item['file'])
result={'status':'pass' if not errors else 'failure','code_payload_files':len(entries),'core_inputs':len(core),'extension_inputs':len(extension),'errors':errors}
print(json.dumps(result,indent=2));assert not errors
