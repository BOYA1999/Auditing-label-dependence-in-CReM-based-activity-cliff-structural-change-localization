import io,json,xml.etree.ElementTree as ET
from pathlib import Path
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research_v2/artifacts/analysis/deep_construct_20261009_v1'
OUT=ROOT/'research_v2/artifacts/analysis/minor_revision_20261015/extension'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.4,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
summary=pd.read_csv(OUT/'common_population_summary.csv').set_index('concept').loc['heteroatom_count']
old=[json.loads((BASE/f'evaluation_summary_{c}.json').read_text(encoding='utf8'))['primary'] for c in ['development','external']]
fig,axes=plt.subplots(1,2,figsize=(6.69,2.6),gridspec_kw={'width_ratios':[1,1.05]})
fig.subplots_adjust(left=.10,right=.985,bottom=.23,top=.82,wspace=.37)
for i,r in enumerate(old):
    axes[0].errorbar(i,r['paired_contrast'],yerr=[[r['paired_contrast']-r['ci_low']],[r['ci_high']-r['paired_contrast']]],fmt='o',color='#255b83',capsize=4,markersize=5)
axes[0].axhline(0,color='#999999',lw=.7);axes[0].axhline(.1,color='#999999',lw=.7,ls='--')
axes[0].set(xticks=[0,1],xticklabels=['Development\n(n = 308)','External\n(n = 256)'],xlim=(-.5,1.5),ylim=(-.02,.42),ylabel='Paired nAP difference')
axes[0].set_title('A  Occupancy − net response',loc='left',fontsize=9.1,pad=13,fontweight='bold')
variants=['net','occupancy','native_removal'];colors=['#255b83','#bc7d32','#16806a']
for i,(v,c) in enumerate(zip(variants,colors)):
    value=summary[v+'_nap'];lo=summary[v+'_nap_ci_low'];hi=summary[v+'_nap_ci_high']
    axes[1].bar(i,value,width=.6,color=c)
    axes[1].errorbar(i,value,yerr=[[value-lo],[hi-value]],fmt='none',color='#333333',capsize=3,lw=.8)
    axes[1].text(i,hi+.024,f'{value:.3f}',ha='center',fontsize=8)
axes[1].set(xticks=range(3),xticklabels=['Net\nchange','Region\noccupancy','Native\nremoval'],ylim=(0,.50),ylabel='Molecule median nAP')
axes[1].set_title('B  Same 251 supported structures',loc='left',fontsize=9.1,pad=13,fontweight='bold')
buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig)
NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS)
svg=ET.Element('{'+NS+'}svg',{'width':'170mm','height':'216mm','viewBox':'0 0 760 966'})
ET.SubElement(svg,'{'+NS+'}rect',{'x':'0','y':'0','width':'760','height':'966','fill':'white'})
top=ET.fromstring(buf.getvalue());top.set('x','0');top.set('y','0');top.set('width','760');top.set('height','295');svg.append(top)
def text(x,y,value,size=16,bold=False,color='#222222',anchor='start'):
    t=ET.SubElement(svg,'{'+NS+'}text',{'x':str(x),'y':str(y),'font-family':'DejaVu Sans','font-size':str(size),'font-weight':'bold' if bold else 'normal','fill':color,'text-anchor':anchor});t.text=value
def molecule(mol,highlight,color,x,y):
    drawer=rdMolDraw2D.MolDraw2DSVG(220,185);options=drawer.drawOptions();options.fixedFontSize=17;options.bondLineWidth=1.3;options.padding=.12;options.useBWAtomPalette()
    drawer.DrawMolecule(mol,highlightAtoms=highlight,highlightAtomColors={i:color for i in highlight},highlightAtomRadii={i:.34 for i in highlight},highlightBonds=[])
    drawer.FinishDrawing();element=ET.fromstring(drawer.GetDrawingText());element.set('x',str(x));element.set('y',str(y));svg.append(element)
    return drawer
def arrow(x1,x2,y):
    ET.SubElement(svg,'{'+NS+'}path',{'d':f'M {x1} {y} L {x2-7} {y}','stroke':'#666666','stroke-width':'1.5'})
    ET.SubElement(svg,'{'+NS+'}path',{'d':f'M {x2-8} {y-4} L {x2} {y} L {x2-8} {y+4} Z','fill':'#666666'})
examples={r['event']:r for r in json.loads((OUT/'molecular_examples.json').read_text(encoding='utf8'))}
for event,y,letter,title in [('creation',323,'C','Creation at a currently negative site'),('cancellation',637,'D','Deletion–insertion cancellation')]:
    r=examples[event];parent=Chem.MolFromSmiles(r['parent_smiles']);product=next(iter(Chem.SDMolSupplier(str(OUT/f'example_{event}_product.sdf'))))
    rdDepictor.Compute2DCoords(parent)
    atom_map=[(a['parent_atom_index'],a['product_atom_index']) for a in r['product_atom_provenance'] if a['parent_atom_index'] is not None]
    rdDepictor.GenerateDepictionMatching2DStructure(product,parent,atom_map)
    text(22,y,f'{letter}  {title}',18,True)
    text(130,y+33,'Parent',16,True,anchor='middle');text(380,y+33,'Retained mutant',16,True,anchor='middle');text(630,y+33,'Parent score assignment',16,True,anchor='middle')
    molecule(parent,r['deleted_parent_atoms'],(.94,.70,.39),20,y+44)
    molecule(product,r['inserted_product_atoms'],(.42,.80,.68),270,y+44)
    molecule(Chem.Mol(parent),r['allowed_parent_atoms'],(.48,.69,.87),520,y+44)
    arrow(240,263,y+132);arrow(490,513,y+132)
    if event=='creation':
        text(130,y+250,'I deleted; 0 heteroatoms lost',14,anchor='middle')
        text(380,y+250,'1 N inserted; |Δcount| = 1',14,anchor='middle')
        text(630,y+250,'Region mean = 1 (1 mutant)',14,anchor='middle')
    else:
        text(130,y+250,'1 N deleted',14,anchor='middle')
        text(380,y+250,'1 N inserted; |Δcount| = 0',14,anchor='middle')
        text(630,y+250,'Region mean = 2/3 (3 mutants)',14,anchor='middle')
    text(130,y+273,'Orange: allowed = deleted',13,color='#83501d',anchor='middle')
    text(380,y+273,'Green: all inserted atoms',13,color='#166b58',anchor='middle')
    text(630,y+273,'Blue: receives region response',13,color='#255b83',anchor='middle')
ET.SubElement(svg,'{'+NS+'}path',{'d':'M 22 616 L 738 616','stroke':'#dddddd','stroke-width':'1'})
text(22,950,'Heteroatoms: N, O, P and S. Structures illustrate saved events; they are not representative samples.',12)
data=ET.tostring(svg,encoding='utf-8',xml_declaration=True);(OUT/'figure5_revised.svg').write_bytes(data)
cairosvg.svg2pdf(bytestring=data,write_to=str(OUT/'figure5_revised.pdf'))
cairosvg.svg2png(bytestring=data,write_to=str(OUT/'figure5_revised.png'),output_width=2008,output_height=2552)
Image.open(OUT/'figure5_revised.png').convert('RGB').save(OUT/'figure5_revised.tiff',dpi=(300,300),compression='tiff_lzw')
(OUT/'figure5_source.json').write_text(json.dumps({'panel_A':old,'panel_B':summary.to_dict(),'panels_C_D':list(examples.values()),'dimensions_mm':[170,216],'scope':'Panels A and B have different explicitly labelled populations; C and D illustrate saved events. Single-mutant net effects and region means are distinct.'},indent=2)+'\n',encoding='utf8')
print('Figure 5 vector PDF/SVG and 300 dpi PNG/TIFF saved',flush=True)
