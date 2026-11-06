"""Preserve two pre-model failures; correct metadata binding and image adapter."""
import json
from pathlib import Path
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha


if __name__=='__main__':
    draft=ROOT/'refactor_v2/reports/NESTED_DESIGN_REQUIREMENTS_V3_1.json'
    assert sha(draft)=='0a225a950cc4d42339075c652be7a9bd07d4a4fcdbb15b29bf389b04ea56565c'
    design=json.loads(draft.read_text());base=ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_0'
    detector=base/'detector_split_v3.json'
    assert sha(detector)==design['folds']['0']['detector_metadata_sha256']
    det=json.loads(detector.read_text());train=design['folds']['0']['train']
    assert det['source_vqa']['train']=={'path':train['manifest'],'sha256':train['sha256']}
    assert sha(Path(train['manifest']))==train['sha256']
    inventory=ROOT/'refactor_v2/reports/fresh_exact_upstream_fracatlas_v3_1/image_hashes.jsonl'
    assert sha(inventory)=='edbdf69f9d347a7cbd3c79fbce5bfe3a0ae59dd9c830cce6ffa6767ebb4b940e'
    expected={r['path']:r['sha256'] for s in inventory.read_text().splitlines() if '\\fracatlas\\' in (r:=json.loads(s))['path']}
    image_root=ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'
    for image in {r['image'] for r in json.loads(Path(train['manifest']).read_text())}:
        adapted=image_root/image
        canonical=ROOT/'data'/('BONE_VQA\\BaoYen_work\\data\\fracatlas\\FracAtlas\\'+image.replace('/','\\'))
        assert adapted.resolve()==canonical and sha(adapted)==expected[str(canonical)]
    source=ROOT/'refactor_v2/experiments/gate_fullfold0_uniform_timing_v3_1.py'
    text=source.read_text()
    substitutions={
        "OUT=ROOT/'refactor_v2/reports/fullfold0_uniform_timing_gate_v3_1'":"OUT=ROOT/'refactor_v2/reports/fullfold0_uniform_timing_gate_v3_3'",
        "for p in (train,vocabpath,base/'detector_split_v3.json'):":"for p in (train,vocabpath):",
        "root=str(ROOT/'data'),":"root=str(ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'),",
        "[Path(__file__).resolve(),cfgpath,train,vocabpath,sealpath,":"[Path(__file__).resolve(),Path(SOURCE_CORE_PATH),Path(DRAFT_PATH),Path(DETECTOR_PATH),Path(INVENTORY_PATH),cfgpath,train,vocabpath,sealpath,"}
    for old,new in substitutions.items():
        assert text.count(old)==1,old
        text=text.replace(old,new)
    namespace={'__name__':'fullfold0_corrected_image_gate','__file__':str(Path(__file__).resolve()),
        'SOURCE_CORE_PATH':str(source),'DRAFT_PATH':str(draft),'DETECTOR_PATH':str(detector),'INVENTORY_PATH':str(inventory)}
    exec(compile(text,str(source),'exec'),namespace)
    namespace['main']()
