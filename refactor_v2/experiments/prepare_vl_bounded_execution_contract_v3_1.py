"""Write capped candidates and test reject cases;never call a training loop."""
import copy
import json
import time
from pathlib import Path

from src.context_controls import ComponentBatchSampler,load_component_lookup
from src.data import build_dataset
from src.protocol import records_of
from src.train import build_split
from experiments.vl_bounded_execution_contract_v3_1 import ARMS,INIT,INIT_SHA,OPTIMIZER,phases,model_config,validate_plan
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/vl_bounded_execution_contract_v3_1"
    assert not folder.exists()
    started=time.perf_counter()
    proposals=root/"refactor_v2/reports/vl_pilot_proposals640_v3_1/boxes.json"
    control=root/"refactor_v2/reports/vl_matched_control_audit_v3_1"
    base=json.loads((root/"refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3.json").read_text())
    ds=build_split(build_dataset,base["data"]["dataset"],"train",None,True,base["data"]["loader_kwargs"])
    bc=base["train"]["component_batching"]
    lookup,_=load_component_lookup(bc["manifest"],bc["seal"],root)
    sampler=ComponentBatchSampler([lookup[r["image_id"]] for r in records_of(ds)],4,42)
    schedule=[]
    for epoch in range(4):
        sampler.set_epoch(epoch)
        batches=list(sampler)
        assert len(batches)==294 and sorted(i for b in batches for i in b)==list(range(1173))
        schedule.append({"global_epoch":epoch,"microbatches":294,"optimizer_steps":74,"all_rows_once":True,"ordered_indices":batches})
    folder.mkdir()
    configs={}
    for arm in ARMS:
        config={"scope":"bounded_VL_development_plan_v3_1","arm":arm,"seed":42,"recipe_status":"candidate_not_launch_sealed","quality_training_released":False,"phases":phases(arm),"total_optimizer_steps_cap":296,"optimizer":OPTIMIZER,"runtime":{"device":"cpu","dtype":"float32","torch_threads":8,"workers":0,"offline":True},"data_roles":{"train":{"QA_rows":1173,"images":261},"model_val":{"QA_rows":142,"images":32}},"classification_threshold":0.5,"fit_calibration":False,"initialization":{"path":str(root/INIT),"sha256":INIT_SHA,"kind":"ROCO_PREtask_NOT_smoke_or_tasktrained"},"model":model_config(arm),"roi_condition":"group_translation_location_placebo" if arm.endswith("location_placebo") else "real","primary_checkpoint":"LAST296cumulativeupdates","allowed_inference_keys":["images","questions","automatic_boxes"] if arm.startswith("V_") else ["images","questions"],"proposals":{"path":str(proposals),"sha256":sha(proposals)},"controls":{"report_sha256":sha(control/"report.json"),"geometry_path":str(control/"group_translation_location_placebo.json"),"geometry_sha256":sha(control/"group_translation_location_placebo.json")}}
        validate_plan(config)
        (folder/(arm+".json")).write_text(json.dumps(config,indent=2)+"\n")
        configs[arm]=config
    faults={}
    for name in ("extra_epoch","extra_steps","wrong_stage","wrong_phase_offset","different_seed","changed_lr","MPS_switch","GT_region_input","lens_inference_ROI","test_role_alias","calibration_fit","tasktrained_initialization","claimed_release"):
        cfg=copy.deepcopy(configs["L_real"])
        if name=="extra_epoch":cfg["phases"][0]["epochs"]=3
        elif name=="extra_steps":cfg["total_optimizer_steps_cap"]=297
        elif name=="wrong_stage":cfg["phases"][0]["stage"]=3
        elif name=="wrong_phase_offset":cfg["phases"][1]["global_epoch_offset"]=0
        elif name=="different_seed":cfg["seed"]=43
        elif name=="changed_lr":cfg["optimizer"]["lr"]=0.001
        elif name=="MPS_switch":cfg["runtime"]["device"]="mps"
        elif name=="GT_region_input":cfg["model"]["use_region_in_prompt"]=True
        elif name=="lens_inference_ROI":cfg["model"]["lens_training_only"]=False
        elif name=="test_role_alias":cfg["data_roles"]["test"]=cfg["data_roles"].pop("model_val")
        elif name=="calibration_fit":cfg["fit_calibration"]=True
        elif name=="tasktrained_initialization":cfg["initialization"]["kind"]="tasktrained_B"
        else:cfg["quality_training_released"]=True
        try:validate_plan(cfg,deep_sources=False)
        except AssertionError:faults[name]=True
        else:raise RuntimeError("Out-of-scope plan accepted:"+name)
    (folder/"component_epoch_schedule.json").write_text(json.dumps(schedule,indent=2)+"\n")
    report={"passed_plan_and_caps_tests":True,"scope":"candidateconfigandcontracttestsNOTexecutionrelease","config_sha256":{arm:sha(folder/(arm+".json")) for arm in ARMS},"sampler_schedule_sha256":sha(folder/"component_epoch_schedule.json"),"source_contract_sha256":sha(root/"refactor_v2/experiments/vl_bounded_execution_contract_v3_1.py"),"negative_cases_rejected":faults,"all4epochs_no_row_loss":True,"microbatches_per_epoch":294,"optimizer_steps_per_epoch":74,"elapsed_seconds":time.perf_counter()-started,"quality_training_released":False,"pending":["control-specific isolated-gradient/actual-update smoke on exactpilotTRAIN;discardupdatedweights","materializeandverifyfreshperarmfullinitialstates;sealallruntime/config/control/datahashes","dedicatedboundedexecutorcapvalidationandindependentverifybeforefirstqualityupdate;unmodifiedCLIguardremainsstrict"],"calibration_or_test_used":False}
    (folder/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
