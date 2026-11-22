#!/bin/zsh
# Power guard: the 68 W adapter cannot feed a V4 training (CPU) and a YOLOv8 training (MPS) at the same time, so the
# battery drains even on AC.  When the battery falls to LOW %, the running V4 process(es) are paused (SIGSTOP);
# they are resumed (SIGCONT) at HIGH %.  YOLOv8 is never paused: the GPU lane sets the finish time of every fold.
# Pausing changes nothing in training (same updates, same data order); only wall-clock time.
# Marker runs/lanes/v4_paused exists while paused (lane_speed.py skips its "no update" alarm then).
LOW=${LOW:-35}; HIGH=${HIGH:-80}
LANES=/Users/uyen/Downloads/BONEVQA/yolo_pilot/runs/lanes
LOGF=/Users/uyen/Downloads/BONEVQA/yolo_pilot/logs/power_guard.log
PAT="scripts\.(train_arm|train_joint|eval_arm|eval_joint|reference_score|trust_fit|compare_arms|compare_btxrd)"
resume() { for p in $(pgrep -f "$PAT"); do kill -CONT $p 2>/dev/null; done; rm -f $LANES/v4_paused; }
trap 'resume; echo "$(date "+%F %T") guard stopped, V4 resumed" >> $LOGF; exit 0' INT TERM
echo "$(date '+%F %T') guard start (pause V4 at <= $LOW%, resume at >= $HIGH%)" >> $LOGF
while true; do
  b=$(pmset -g batt | grep -o '[0-9]*%' | tr -d %)
  if [ ! -f $LANES/v4_paused ] && [ "$b" -le $LOW ]; then
    pids=$(pgrep -f "$PAT" | tr '\n' ' ')
    if [ -n "$pids" ]; then
      for p in ${=pids}; do kill -STOP $p; done
      touch $LANES/v4_paused
      echo "$(date '+%F %T') battery $b%: paused V4 ($pids)" >> $LOGF
    fi
  elif [ -f $LANES/v4_paused ]; then
    # keep any V4 process paused (none can start: the CPU lane waits on the paused one)
    for p in $(pgrep -f "$PAT"); do kill -STOP $p 2>/dev/null; done
    # V4 alone does not drain the battery on AC: resume early once no YOLOv8 training runs (added 2026-11-19 14:05)
    if [ "$b" -ge $HIGH ] || { [ "$b" -ge 20 ] && pmset -g batt | grep -q "AC Power" && ! pgrep -f "code/train_yolo\.py" > /dev/null; }; then
      resume
      echo "$(date '+%F %T') battery $b%: resumed V4" >> $LOGF
    fi
  fi
  sleep 60
done
