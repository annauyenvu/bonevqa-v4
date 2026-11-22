#!/bin/zsh
# Fold 4, chained after lane_final.sh (user 21/11 00:2x: "chạy liên tục hết công suất, phê duyệt giúp tôi nếu cần").
# Scope-freeze rule 3 kept: no fold 4 detector next to a V4 training, so this waits until lane_final.sh has exited
# and has written the summary tables (JF folds 0-3 done).  Then, one after the other:
#   1. GPU lane fold 4: cfj1_f4 (was stopped at epoch 32/60, retrained from scratch), cfj2_f4, predictions, proposals
#   2. CPU lane fold 4: D2, J, evaluations, comparisons, trust layer (B fold 4 is already trained)
#   3. JF fold 4: FracAtlas r0.02 + BTXRD r0.02 (training) and the locked BTXRD inference level, then lane_final FOLDS=4
# Fold 4 results are for IEEE Access; the thesis tables stay on folds 0-3.  Restartable: every step skips finished work.
LANE=fold4
source /Users/uyen/Downloads/BONEVQA/yolo_pilot/code/folds_lib.sh
FINAL_PID=${FINAL_PID:-38492}
{
stamp "fold 4 chain: waiting for lane_final.sh (pid $FINAL_PID)"
while kill -0 $FINAL_PID 2>/dev/null; do sleep 120; done
if ! grep -q "\[final\] summary tables written" $QLOG; then stamp "lane_final.sh ended without the summary tables: fold 4 not started"; exit 2; fi
cd $PILOT
stamp "fold 4 chain: GPU lane"
FOLDS=4 ./code/lane_gpu.sh || { stamp "fold 4 GPU lane failed"; exit 3; }
[ -f $LANES/props_f4.done ] || { stamp "fold 4 proposals missing"; exit 3; }
stamp "fold 4 chain: CPU lane"
FOLDS=4 DISKMIN=5 ./code/lane_cpu.sh || { stamp "fold 4 CPU lane failed"; exit 4; }
SW=$VGDIR/runs/crossfit_yolo_joint_sweep
SEL=$VGDIR/runs/joint/final_btxrd_level.json
BL=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['chosen_level'])" $SEL)
[ -f $SW/r0.02/fold_4/proposals_fold4.json ] || { stamp "FracAtlas r0.02 proposals fold 4"; python3 -I code/sweep_proposals.py 4 0.02 > $LOG/sweep_props_r0.02_f4.log 2>&1; }
for l in 0.02 $BL; do
  [ -f $SW/r$l/btxrd_fold4/proposals_btxrd_insample.json ] || { stamp "BTXRD r$l proposals fold 4"; python3 -I code/build_btxrd_proposals.py 4 $l > $LOG/props_btxrd_r${l}_f4.log 2>&1; }
done
stamp "fold 4 chain: JF fold 4"
FOLDS=4 ./code/lane_final.sh || { stamp "JF fold 4 failed"; exit 5; }
stamp "fold 4 chain: done"
} >> $QLOG
