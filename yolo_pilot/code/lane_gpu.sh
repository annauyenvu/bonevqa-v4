#!/bin/zsh
# GPU lane: per fold, data preparation, Faster R-CNN reference boxes, the joint YOLOv8 detectors (outer + 3 inner
# cross-fit, MPS), their predictions and the V4 proposal files.  Fold order 1, 2, 3, 4.  Restartable: every step is
# skipped when its output exists; a detector folder without its "done" marker is moved aside and retrained.
set -e
LANE=gpu
source /Users/uyen/Downloads/BONEVQA/yolo_pilot/code/folds_lib.sh
FOLDS=(${=FOLDS:-1 2 3 4})
if pgrep -f "code/train_yolo\.py" > /dev/null; then stamp "a YOLOv8 training is already running: not starting" >> $QLOG; exit 1; fi
# one GPU lane at a time, also between trainings (added 2026-11-19: fold 4 started early, lane_sweep.sh starts it again)
GLOCK=$LANES/gpu_lane.pid
if [ -f $GLOCK ] && kill -0 $(cat $GLOCK) 2>/dev/null; then stamp "GPU lane already running (pid $(cat $GLOCK)): not starting" >> $QLOG; exit 0; fi
echo $$ > $GLOCK

yolo_train() {  # run name, data yaml (relative to yolo_pilot)
  local name=$1 data=$2
  cd $PILOT
  if [ ! -f runs/yolo/$name/done ]; then
    diskcheck
    if [ -d runs/yolo/$name ]; then mv runs/yolo/$name runs/yolo/${name}_incomplete_$(date +%s); fi
    stamp "train detector $name"
    $YPY code/train_yolo.py --name $name --data $data --epochs 60 --patience 20 --device mps > $LOG/train_$name.log 2>&1
    touch runs/yolo/$name/done
  fi
}
yolo_predict() {  # weights run name, output name, then predict_yolo arguments
  local name=$1 out=$2; shift 2
  cd $PILOT
  if [ ! -f runs/pred/${out}_cls0.json ]; then
    stamp "predict $out"
    $YPY code/predict_yolo.py --weights runs/yolo/$name/weights/best.pt --out runs/pred/$out "$@" --device cpu > $LOG/predict_$out.log 2>&1
  fi
}
prep() {  # fold: everything B needs and the YOLOv8 data of the fold
  local f=$1 DS=datasets/fa_fold$1
  cd $PILOT
  if [ ! -f $DS/gt.json ]; then stamp "prepare FracAtlas fold $f for YOLO"; python3 -I code/prep_fracatlas.py $f > $LOG/prep_f$f.log 2>&1; fi
  if [ ! -f $VGDIR/runs/crossfit_yolo/fold_$f/inner_split.json ]; then stamp "inner split fold $f"; v4 -m scripts.yolo_cf_split --fold $f > $LOG/split_f$f.log 2>&1; fi
  cd $PILOT
  if [ ! -d $DS/cf ]; then python3 -I code/make_cf_lists.py $f > $LOG/cflists_f$f.log 2>&1; fi
  if [ ! -f $DS/data_fa_btxrd_os3.yaml ]; then python3 -I code/prep_fold_yaml.py $f > $LOG/yaml_f$f.log 2>&1; fi
  if [ ! -f runs/frcnn/fold${f}_ep2_sealed.json ]; then
    stamp "Faster R-CNN reference boxes fold $f"
    ( source $VGDIR/env.sh > /dev/null; $PY $PILOT/code/frcnn_raw.py $f ) > $LOG/frcnn_raw_f$f.log 2>&1
  fi
  touch $LANES/prep_f$f.done
}

stamp "GPU lane start (folds: $FOLDS)" >> $QLOG
# preparation of every fold first (minutes each), so the CPU lane can train B while the detectors train
for f in $FOLDS; do prep $f >> $QLOG; done

for f in $FOLDS; do
  {
  cd $PILOT
  DS=datasets/fa_fold$f
  CFJ=$(cfj $f)
  BTP=$(btp $f)
  yolo_train fa_btxrd_os3_f$f $DS/data_fa_btxrd_os3.yaml
  yolo_predict fa_btxrd_os3_f$f f${f}_fa_btxrd --dirs $DS/images/val $DS/images/cal $DS/images/test \
    datasets/btxrd_pilot/images/val datasets/btxrd_pilot/images/test datasets/btxrd_pilot/images/external
  yolo_predict fa_btxrd_os3_f$f f${f}_fa_btxrd_train --dirs $DS/images/train
  yolo_predict fa_btxrd_os3_f$f f${f}_fa_btxrd_bttrain --dirs datasets/btxrd_pilot/images/train
  cd $PILOT
  if [ ! -f $CFJ/proposals_fold${f}_insample.json ]; then
    stamp "in-sample proposals fold $f"; python3 -I code/build_v4_proposals.py $f insample joint > $LOG/props_insample_f$f.log 2>&1
  fi
  if [ ! -f $BTP ]; then stamp "BTXRD proposals fold $f"; python3 -I code/build_btxrd_proposals.py $f > $LOG/props_btxrd_f$f.log 2>&1; fi
  for i in 0 1 2; do
    yolo_train cfj${i}_f$f $DS/cf/data_cfj$i.yaml
    yolo_predict cfj${i}_f$f f${f}_cfj$i --lists $DS/cf/inner${i}_heldout.txt --dirs $DS/images/val
  done
  cd $PILOT
  if [ ! -f $CFJ/proposals_fold$f.json ]; then
    stamp "cross-fitted proposals fold $f"; python3 -I code/build_v4_proposals.py $f full joint > $LOG/props_full_f$f.log 2>&1
  fi
  touch $LANES/props_f$f.done
  stamp "detectors and proposals fold $f done"
  } >> $QLOG
done
stamp "GPU lane done" >> $QLOG
