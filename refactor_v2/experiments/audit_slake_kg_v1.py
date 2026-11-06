"""Inventory source KG relations and literal TRAIN-domain coverage, without fitting."""
import csv,hashlib,io,json,re,zipfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];O=ROOT/'refactor_v2/reports/pf_slake_source_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    receipt=json.loads((O/'source_receipt.json').read_text());assert sha(O/'KG.zip')==receipt['archive_sha256']
    tables={}
    with zipfile.ZipFile(O/'KG.zip') as z:
        for name in ('en_organ.csv','en_organ_rel.csv','en_disease.csv'):
            rows=list(csv.reader(io.StringIO(z.read('KG/'+name).decode('utf-8-sig')),delimiter='#'))
            header=rows.pop(0);assert len(header)==3 and all(len(r)==3 for r in rows)
            tables[name]=[dict(zip(('subject','relation','object'),[s.strip() for s in r])) for r in rows]
    train_path=ROOT/'refactor_v2/reports/step4_localization_v3_1/train.json';train=json.loads(train_path.read_text())
    regions=sorted({term.strip() for r in train for term in r['region'].split(' and ') if r.get('region')})
    nodes=sorted({r['subject'] for table in tables.values() for r in table})
    terms=regions+['fracture']
    def contains(text,term):return bool(re.search(r'\b'+re.escape(term)+r's?\b',text,re.I))
    matches={term:[n for n in nodes if contains(n,term)] for term in terms}
    location=[r for r in tables['en_disease.csv'] if r['relation'].lower()=='location']
    fracture_nodes=matches['fracture'];fracture_relations=[r for r in location if r['subject'] in fracture_nodes]
    audit={'source_archive_sha256':sha(O/'KG.zip'),'TRAIN_manifest_sha256':sha(train_path),'declared_license':receipt['declared_license'],'tables':{n:{'rows':len(rs),'subjects':len({r['subject'] for r in rs}),'relations':dict(Counter(r['relation'] for r in rs))} for n,rs in tables.items()},'literal_TRAIN_domain_matches':matches,'disease_location_triples':len(location),'fracture_node_location_rows_verbatim':fracture_relations,'interpretation':'Literal label coverage only; no synonym expansion, invented edges or claims of clinical completeness. Original multi-location cells remain verbatim until a parsing specification is frozen. This is the author-linked SLAKE source, not proof of byte-identical LaPA processed adjacency.','exact_LaPA_processed_graph_verified':False,'graph_integration_released':False,'source_sha256':sha(Path(__file__))}
    with (O/'coverage_audit.json').open('x') as f:json.dump(audit,f,indent=2)
    with (O/'english_source_triples.json').open('x') as f:json.dump(tables,f,indent=2)
    with (O/'source_seal.json').open('x') as f:json.dump({'artifacts':{str(O/n):sha(O/n) for n in ('KG.zip','SOURCE_README.md','source_receipt.json','coverage_audit.json','english_source_triples.json')},'training_released':False},f,indent=2)
    print(json.dumps({'tables':audit['tables'],'matches':matches,'fracture_locations':fracture_relations,'exact_LaPA_processed_graph_verified':False}))
if __name__=='__main__':main()
