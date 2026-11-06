"""Exact B-pilot active-role proposals,not calibration/test or training release."""
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from src.data.fracatlas_vqa import FRACATLAS_ROOT
from src.data.sealed_manifest import load_sealed_records
from src.detector_protocol import load_detector_manifest
from src.preprocessing import canonicalize_raster,RASTER_PREPROCESSING_VERSION
from src.provenance import validate_detector_proposals
from experiments.vl_bounded_proposal_contract_v3_1 import sha,validate_roles,PilotPreparationDetector640,validate_prepared_proposals


@torch.no_grad()
def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/vl_pilot_proposals640_v3_1"
    if folder.exists():
        raise RuntimeError("Existing proposal preparation;preserve and inspect,no duplicate")
    if any(" -m src.train " in row for row in subprocess.check_output(["ps","-axo","command"],text=True).splitlines()):
        raise RuntimeError("Training active;do not compete")
    assert os.environ["HF_HUB_OFFLINE"]==os.environ["TRANSFORMERS_OFFLINE"]=="1" and os.environ["OMP_NUM_THREADS"]=="8"
    parent=root/"refactor_v2/reports/fracatlas_scale_pilot_v3"
    detector=root/"refactor_v2/reports/detector640_fold0_bounded_quality_v3_1"
    base=json.loads((detector/"execution_release.json").read_text())
    evaluation=detector/"exact_architecture_independent_verification.json"
    assert sha(evaluation)==json.loads((detector/"exact_architecture_verification_progress.json").read_text())["report_sha256"]
    assert json.loads(evaluation.read_text())["passed_artifact_and_exact_architecture_metric_checks"]
    assert json.loads((root/"refactor_v2/reports/vl_detector640_independent_v3/report.json").read_text())["passed"]
    done=json.loads((detector/"completion.json").read_text())
    seal=parent/"data_seal.json"
    roles={}
    image_paths={}
    for role,expected_rows,expected_images in (("train",1173,261),("model_val",142,32)):
        source=parent/(role+".json")
        rows=load_sealed_records(source,seal,root)
        active=sorted({r["image_id"] for r in rows})
        assert len(rows)==expected_rows and len(active)==expected_images
        roles[role]={"path":str(source),"sha256":sha(source),"QA_rows":len(rows),"image_ids":active}
        for row in rows:
            path=Path(FRACATLAS_ROOT)/row["image"]
            assert row["image_id"] not in image_paths or image_paths[row["image_id"]]==path
            image_paths[row["image_id"]]=path
    spec={"schema_version":4,"scope":"bounded_pilot_data_preparation_only","project_root":str(root),"source_QA_seal":{"path":str(seal),"sha256":sha(seal)},"parent_detector_manifest":{"path":base["manifest_path"],"sha256":base["manifest_sha256"]},"detector_checkpoint":{"path":str(root/"refactor_v2/detector640_fold0_bounded_quality_v3_1/last.pt"),"sha256":done["last_sha256"]},"roles":roles,"calibration_or_test_export_allowed":False,"training_released":False}
    folder.mkdir()
    roles_path=folder/"active_roles_manifest.json"
    roles_path.write_text(json.dumps(spec,indent=2)+"\n")
    validate_roles(roles_path)
    state={"status":"preparing","pid":os.getpid(),"scope":spec["scope"],"expected_images":293,"completed_images":0}
    def save():
        (folder/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    started=time.perf_counter()
    try:
        torch.set_num_threads(8)
        runtime=PilotPreparationDetector640(roles_path)
        preparation={"passed_role_and_source_gates":True,"active_roles_manifest_sha256":sha(roles_path),"runtime_source_sha256":sha(root/"refactor_v2/experiments/vl_bounded_proposal_contract_v3_1.py"),"exporter_source_sha256":sha(__file__),"checkpoint_sha256":done["last_sha256"],"training_released":False,"detector_eval_report_sha256":sha(evaluation)}
        (folder/"data_preparation_release.json").write_text(json.dumps(preparation,indent=2)+"\n")
        proposals={}
        perimage={}
        for role,source in roles.items():
            for iid in source["image_ids"]:
                path=image_paths[iid]
                with Image.open(path) as image:
                    image.load()
                    # No annotation,answer,type,region or question passed here.
                    boxes=runtime.predict(image)
                    width,height=canonicalize_raster(image).size
                assert len(boxes)<=3 and all(0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height for b in boxes)
                proposals[iid]=boxes
                perimage[iid]={"role":role,"width":width,"height":height,"input_image_path":str(path),"input_image_sha256":sha(path)}
                if len(proposals)%25==0:
                    state.update(status="exporting",completed_images=len(proposals),role=role,elapsed_seconds=time.perf_counter()-started)
                    save()
        assert len(proposals)==293
        path=folder/"boxes.json"
        path.write_text(json.dumps(proposals,indent=2)+"\n")
        _,foldids,_=load_detector_manifest(base["manifest_path"])
        artifact={"schema_version":4,"scope":spec["scope"],"complete":False,"complete_for_active_roles":True,"training_released":False,"coordinate_space":"exif_canonical_pixels","preprocessing_version":RASTER_PREPROCESSING_VERSION,"checkpoint":spec["detector_checkpoint"]["path"],"checkpoint_sha256":done["last_sha256"],"split_manifest":base["manifest_path"],"split_manifest_sha256":base["manifest_sha256"],"active_roles_manifest":str(roles_path),"active_roles_manifest_sha256":sha(roles_path),"proposal_sha256":sha(path),"coverage":293,"active_expected_coverage":293,"expected_coverage":len(set().union(*(set(v) for v in foldids.values()))),"score_threshold":0.05,"max_boxes":3,"detector_transform":runtime.provenance["detector_transform"],"normalization_spec_sha256":runtime.provenance["normalization_spec_sha256"],"runtime_source":str(root/"refactor_v2/experiments/vl_bounded_proposal_contract_v3_1.py"),"runtime_source_sha256":preparation["runtime_source_sha256"],"exporter_source":str(Path(__file__).resolve()),"exporter_source_sha256":preparation["exporter_source_sha256"]}
        path.with_suffix(".provenance.json").write_text(json.dumps({"artifact":artifact,"images":perimage},indent=2)+"\n")
        validate_prepared_proposals(path)
        try:
            validate_detector_proposals(path,require_complete=True)
        except RuntimeError as error:
            assert "chưa phủ toàn bộ sealed splits" in str(error)
            rejection=str(error)
        else:
            raise RuntimeError("Partial active-role preparation passed fullfold guard")
        summary={"complete_data_preparation":True,"scope":spec["scope"],"role_counts":{role:{"images":len(source["image_ids"]),"nonempty_proposals":sum(bool(proposals[i]) for i in source["image_ids"]),"empty_proposals":sum(not proposals[i] for i in source["image_ids"])} for role,source in roles.items()},"proposal_sha256":sha(path),"sidecar_sha256":sha(path.with_suffix(".provenance.json")),"active_roles_manifest_sha256":sha(roles_path),"fullfold_production_guard_rejection":rejection,"elapsed_seconds":time.perf_counter()-started,"calibration_or_test_exported":False,"scientific_VLP_training_or_production_released":False,"pending":"independent active-role source/coverage/negative-case verification and matched controls before newboundedexecutionrelease"}
        (folder/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        state.update(status="complete_data_preparation_pending_independent_contract_tests",completed_images=293,summary_sha256=sha(folder/"summary.json"))
        save()
        print(json.dumps(summary,indent=2))
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error))
        save()
        raise


if __name__=="__main__":
    main()
