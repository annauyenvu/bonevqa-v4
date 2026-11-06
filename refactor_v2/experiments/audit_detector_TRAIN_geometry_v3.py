"""Read-only TRAIN box survival at raster scales;not a resolution efficacy test."""
import json
import time
from pathlib import Path

import numpy as np
from scripts.train_detector_v2 import BoPhatHienDataset


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/detector_fold0_bounded_quality_v3"
    output=folder/"TRAIN_geometry_audit.json"
    assert not output.exists()
    release=json.loads((folder/"execution_release.json").read_text())
    manifest=json.loads(Path(release["manifest_path"]).read_text())
    ids=set(manifest["split_images"]["train"])
    started=time.perf_counter()
    summary={}
    for size in (320,640):
        ds=BoPhatHienDataset("train",canh_dai=size,tang_cuong=False,allowed_ids=ids)
        counts,widths,heights,areas=[],[],[],[]
        lost=[]
        for i,(iid,(_,raw)) in enumerate(ds.muc):
            if not raw:
                continue
            _,target,_=ds[i]
            boxes=target["boxes"].numpy()
            counts.append((len(raw),len(boxes)))
            if not len(boxes):
                lost.append(iid)
            for x1,y1,x2,y2 in boxes:
                widths.append(float(x2-x1))
                heights.append(float(y2-y1))
                areas.append(float((x2-x1)*(y2-y1)/(size*size)))
        summary[str(size)]={"TRAIN_positive_images":len(counts),"raw_GT_boxes":sum(c[0] for c in counts),"retained_boxes":sum(c[1] for c in counts),"images_losing_all_positive_boxes":lost,"box_width_pixels_p10_p50_p90":np.percentile(widths,[10,50,90]).tolist(),"box_height_pixels_p10_p50_p90":np.percentile(heights,[10,50,90]).tolist(),"box_canvas_area_fraction_p10_p50_p90":np.percentile(areas,[10,50,90]).tolist(),"boxes_min_side_below8pixels":sum(min(w,h)<8 for w,h in zip(widths,heights))}
    report={"scope":"TRAIN-only frozen annotation/raster diagnostic,no training or modelvalidation/test fitting","scales":summary,"elapsed_seconds":time.perf_counter()-started,"warning":"Changing padded input canvas alone does not certify effective detector resolution:MobileNet320 internal transform min_size remains320 unless explicitly changed and gated. No higher-resolution model has been trained or evaluated.","causal_interpretation":"Box survival/size distributions can identify preprocessing loss,not establish cause of low detectorIoU or guarantee a higher-resolution benefit.","calibration_or_test_evaluated":False}
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
