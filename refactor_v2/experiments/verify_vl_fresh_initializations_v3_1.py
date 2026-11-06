"""Independent file-level verification,not qualitytraining authorization."""
import gc
import hashlib
import json
from pathlib import Path

import torch
from experiments.vl_bounded_execution_contract_v3_1 import ARMS,INIT,INIT_SHA,validate_plan
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/vl_control_update_gates_v3_1"
    output=folder/"independent_initialization_verification.json"
    assert not output.exists()
    report=json.loads((folder/"report.json").read_text())
    assert report["passed"] and sha(folder/"report.json")==json.loads((folder/"progress.json").read_text())["report_sha256"]
    assert sha(INIT)==INIT_SHA
    parent=torch.load(INIT,map_location="cpu",weights_only=False)
    expected=json.loads((root/"refactor_v2/reports/vl_matched_control_audit_v3_1/report.json").read_text())["initial_tensor_hashes"]
    checks={}
    for arm in ARMS:
        row=report["results"][arm]
        cfg_path=root/"refactor_v2/reports/vl_bounded_execution_contract_v3_1"/(arm+".json")
        assert sha(cfg_path)==row["config_sha256"]
        config=json.loads(cfg_path.read_text());validate_plan(config)
        assert sha(row["fresh_initialization_path"])==row["fresh_initialization_sha256"]
        payload=torch.load(row["fresh_initialization_path"],map_location="cpu",weights_only=False)
        assert payload["extra"]["scope"]=="freshPREtaskinitialization;zeroQAupdates"
        assert payload["extra"]["parent_ROCO_PREtask_sha256"]==INIT_SHA
        assert payload["config"]==config["model"]
        assert payload["answer_vocab"]==parent["answer_vocab"] and len(payload["answer_vocab"])==68
        hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in payload["state"].items()}
        assert hashes==expected[arm[0]] and all(torch.isfinite(v).all() for v in payload["state"].values())
        assert all(torch.equal(payload["state"][k],v) for k,v in parent["state"].items())
        checks[arm]={"passed":True,"initialization_sha256":row["fresh_initialization_sha256"],"all_initial_tensors":len(hashes),"all263Bshared_exact":True,"zeroQAupdates":True,"modelconfig_vocab_exact":True}
        del payload;gc.collect()
    output.write_text(json.dumps({"passed":True,"scope":"independentfreshinitialstatefilesonly;notqualityexecutionrelease","source_control_gates_sha256":sha(folder/"report.json"),"checks":checks,"calibration_or_test_used":False,"quality_training_released":False},indent=2)+"\n")
    print("Five fresh initializations independently verified;no quality training released")


if __name__=="__main__":
    main()
