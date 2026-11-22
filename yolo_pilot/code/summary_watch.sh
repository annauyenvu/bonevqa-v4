#!/bin/zsh
# Re-run the summary tables (code/summary_final.py, read-only + compare_arms JF vs B) every time lane_final.sh stamps a
# finished fold, so runs/summary_final/bang_tong_hop.md is always current (scope freeze, 20/11/2026).
QLOG=/Users/uyen/Downloads/BONEVQA/yolo_pilot/logs/run_folds_compact.log
cd /Users/uyen/Downloads/BONEVQA/yolo_pilot
seen=0
while true; do
  n=$(grep -c "\[final\] final fold [0-3] \(results ready\|done (trust layer)\)" $QLOG)
  if [ $n -gt $seen ]; then python3 -I code/summary_final.py > logs/folds/summary_watch.log 2>&1; seen=$n; fi
  grep -q "\[final\] summary tables" $QLOG && exit 0
  sleep 120
done
