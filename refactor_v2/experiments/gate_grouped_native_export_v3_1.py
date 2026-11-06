"""Updated-state checkpoint roundtrip gate; disposable weights never quality init."""
from pathlib import Path
p=Path(__file__).with_name('gate_grouped_native_v3_1.py')
s=p.read_text().replace("grouped_native_model_gate_v3_1'","grouped_native_export_gate_v3_1'")
assert s.count('return results')==1
s=s.replace('return results',"return json.loads(json.dumps(results,allow_nan=False,default=lambda value:value.tolist()))")
needle="assert len(after)==263"
addition="""assert len(after)==263
        path=OUT/f'DISPOSABLE_fold{f}.pt'
        if sys.argv[1]=='run':
            assert not path.exists()
            m.save(str(path),extra={'disposable_technical_gate':True,'quality_initialization_forbidden':True})
        payload=torch.load(path,map_location='cpu')
        assert payload['extra']['quality_initialization_forbidden']
        assert all(torch.equal(v,payload['state'][k]) for k,v in after.items())
        del m;gc.collect()
        set_seed(90211)
        m=BoneVQAModel.load(str(path),device='cpu').eval()
        assert all(torch.equal(v,m.trainable_state_dict()[k]) for k,v in after.items())
        with torch.no_grad():restored=m.predict_batch({'images':batch['images'],'questions':batch['questions']})
        normalize=lambda value:json.loads(json.dumps(value,allow_nan=False,default=lambda v:v.tolist()))
        assert normalize(restored)==normalize(pred)
        assert sha(c['init_from'])==p['dependencies'][c['init_from']]
"""
assert s.count(needle)==1
s=s.replace(needle,addition)
s=s.replace("'executor_integration_and_checkpoint_export_gate_pending':True", "'updated_checkpoint_roundtrip_passed':True,'executor_integration_pending':True")
exec(compile(s,str(p),'exec'))
