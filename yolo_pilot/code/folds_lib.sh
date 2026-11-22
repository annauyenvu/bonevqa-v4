#!/bin/zsh
# Shared helpers of the two-lane multi-fold queue (lane_gpu.sh, lane_cpu.sh).  Same commands and arguments as
# run_folds_compact.sh (fold 0 recipe, unchanged); only the scheduling differs.  Approved 2026-11-17: YOLOv8 on MPS
# may run at the same time as V4 training on CPU (thesis deadline 22/11).  Never two YOLOv8 or two V4 trainings at once.
PILOT=/Users/uyen/Downloads/BONEVQA/yolo_pilot
VGDIR=/Users/uyen/Downloads/BONEVQA/v4_grounded_next
LOG=$PILOT/logs/folds
LANES=$PILOT/runs/lanes          # marker files: prep_f<k>.done, props_f<k>.done
mkdir -p $LOG $LANES
YPY=$PILOT/venv/bin/python
BTQ=$PILOT/datasets/btxrd_vqa
QLOG=$PILOT/logs/run_folds_compact.log
stamp() { date "+%F %T [$LANE] $1" }
v4() { ( source $VGDIR/env.sh > /dev/null; cd $VGDIR; $PY "$@" ) }
diskcheck() {
  local free=$(df -g /Users/uyen | awk 'NR==2 {print $4}')
  # DISKMIN: 3 for the fold-0 sweep only (19/11, about 1 GB needed); 5 otherwise
  if [ "$free" -lt ${DISKMIN:-5} ]; then stamp "STOP: only ${free} GB free"; exit 2; fi
}
trained() { [ -f $VGDIR/runs/arms/$1/fold_$2/seed_$3/completed.json ] }
cfj() { echo $VGDIR/runs/crossfit_yolo_joint/fold_$1 }
btp() { echo $VGDIR/runs/crossfit_yolo_joint/btxrd_fold$1/proposals_btxrd_insample.json }
waitfor() {  # marker file: wait (the other lane produces it)
  local first=1
  while [ ! -f $1 ]; do
    [ $first = 1 ] && stamp "waiting for ${1:t}" && first=0
    sleep 300
  done
}
