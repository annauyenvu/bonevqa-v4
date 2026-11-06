"""Source-only PF graph inventory and explicitly synthetic structural controls."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];S=ROOT/'refactor_v2/reports/pf_slake_source_v1';O=ROOT/'refactor_v2/reports/pf_slake_graph_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
def norm(s):return ' '.join(s.casefold().split())
def main():
    assert not O.exists();O.mkdir()
    try:
        for p,h in read(S/'source_seal.json')['artifacts'].items():assert sha(Path(p))==h,p
        tables=read(S/'english_source_triples.json');locations=[(i,r) for i,r in enumerate(tables['en_disease.csv']) if r['relation'].casefold()=='location']
        # Every label comes verbatim from source subjects or location objects.
        organs={norm(r['subject']):r['subject'] for r in tables['en_organ.csv']}
        organs.update({norm(r['object']):r['object'] for _,r in locations})
        diseases={norm(r['subject']):r['subject'] for r in tables['en_disease.csv']}
        nodes=[{'kind':kind,'label':v,'key':kind+':'+k} for kind,values in (('organ',organs),('disease',diseases)) for k,v in sorted(values.items())]
        index={n['key']:i for i,n in enumerate(nodes)};evidence=[]
        for row,r in locations:
            pair=[index['organ:'+norm(r['object'])],index['disease:'+norm(r['subject'])]]
            evidence.append({'edge':pair,'source_table':'en_disease.csv','source_row_zero_based':row,'original_relation':r})
        edges=sorted({tuple(r['edge']) for r in evidence});n=len(nodes)
        truth=set(edges);control=set(edges);rng=np.random.default_rng(20261028);swaps=0
        for _ in range(50*len(edges)):
            ordered=sorted(control);i,j=rng.choice(len(ordered),2,replace=False);a,b=ordered[i],ordered[j]
            if a[0]==b[0] or a[1]==b[1]:continue
            c,d=(a[0],b[1]),(b[0],a[1])
            if c in control or d in control:continue
            control.remove(a);control.remove(b);control.add(c);control.add(d);swaps+=1
        def degree(es):return np.bincount(np.array(list(es)).reshape(-1),minlength=n)
        assert len(control)==len(truth) and np.array_equal(degree(control),degree(truth)) and control!=truth
        self_edges=[[i,i] for i in range(n)]
        def message_edges(es):return self_edges+[[u,v] for a,b in sorted(es) for u,v in ((a,b),(b,a))]
        write(O/'nodes.json',nodes);write(O/'source_edge_evidence.json',evidence)
        write(O/'real_graph.json',{'message_edges':message_edges(truth),'medical_source_edges':[list(e) for e in edges],'self_loops_are_computational':True,'direction':'Source disease-location relation, symmetrized for message passing'})
        write(O/'graph_off.json',{'message_edges':self_edges,'medical_source_edges':[],'scope':'Experimental graph-off: self messages only, same node labels/order'})
        write(O/'degree_placebo.json',{'message_edges':message_edges(control),'synthetic_control_edges':[list(e) for e in sorted(control)],'scope':'Synthetic structural placebo only; these are not asserted medical relations'})
        summary={'nodes':n,'organ_nodes':len(organs),'disease_nodes':len(diseases),'original_location_rows':len(locations),'unique_medical_edges':len(edges),'duplicate_source_rows':len(locations)-len(edges),'isolated_source_nodes':int((degree(truth)==0).sum()),'placebo_seed':20261028,'successful_swaps':swaps,'placebo_degree_exact':True,'source_edges_retained_in_placebo':len(truth&control),'source_archive_sha256':sha(S/'KG.zip'),'TRAIN_or_evaluation_labels_used':False,'node_selection':'All English organ subjects, disease subjects and source location-object labels; no learned filtering, synonym edges or domain relabeling.','exact_LaPA_processed_graph_verified':False,'graph_model_gates_pending':True,'quality_released':False}
        write(O/'graph_audit.json',summary)
        write(O/'graph_seal.json',{'artifacts':{str(p):sha(p) for p in O.glob('*.json')},'source_code':{str(Path(__file__)):sha(Path(__file__))},'training_released':False})
        print(json.dumps(summary))
    except Exception as e:
        write(O/'preparation_failure.json',{'error':repr(e),'preserve_outputs':True});raise
if __name__=='__main__':main()
