"""Preserved audit repair: explicit detector val -> model_val hash binding.

Only the auditor's in-memory role names change; original manifests untouched.
"""
from pathlib import Path
from experiments import audit_data_lineage_closure_v3_1 as core

IMPLEMENTATION=Path(core.__file__).resolve()
ORIGINAL_READ=core.read
core.__file__=str(Path(__file__).resolve())
core.OUT=core.REPORTS/'data_lineage_closure_v3_2.json'


def read(path):
    value=ORIGINAL_READ(path);path=Path(path)
    if path.name=='detector_split_v3.json':
        assert set(value['split_images'])=={'train','val','calibration','test'}
        for detector_role,qa_role in [('train','train'),('val','model_val'),
                                      ('calibration','calibration'),('test','test')]:
            source=value['source_vqa'][detector_role]
            expected=path.parent/('vqa_'+qa_role+'.json')
            assert Path(source['path']).resolve()==expected.resolve()
            assert source['sha256']==core.sha(expected)
        value['split_images']['model_val']=value['split_images'].pop('val')
    return value


core.read=read

if __name__=='__main__':
    core.DEPENDENCIES[str(IMPLEMENTATION)]=core.sha(IMPLEMENTATION)
    failure=core.REPORTS/'data_lineage_closure_v3_1_failure.json'
    record=ORIGINAL_READ(failure);assert record['exception']=="KeyError('model_val')"
    core.main()
