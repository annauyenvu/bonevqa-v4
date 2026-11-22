#!/bin/zsh
# Added 2026-11-20 05:15 (charger unplugged 04:51 during YOLOv8 fold 4): pause the YOLOv8 training (SIGSTOP) at
# <= LOW % on battery so the run is not lost when the battery empties (lane_gpu.sh would retrain from scratch);
# resume (SIGCONT) once on AC. Pausing changes nothing in training, only wall-clock time. Exits when no YOLOv8 runs.
LOW=${LOW:-15}
LOGF=/Users/uyen/Downloads/BONEVQA/yolo_pilot/logs/power_guard.log
paused=0
echo "$(date '+%F %T') yolo guard start (pause YOLOv8 at <= $LOW% on battery, resume on AC)" >> $LOGF
while pgrep -f "code/train_yolo\.py" > /dev/null; do
  b=$(pmset -g batt | grep -o '[0-9]*%' | tr -d %)
  ac=$(pmset -g batt | grep -c "AC Power")
  if [ $paused = 0 ] && [ $ac = 0 ] && [ "$b" -le $LOW ]; then
    for p in $(pgrep -f "code/train_yolo\.py"); do kill -STOP $p; done; paused=1
    echo "$(date '+%F %T') battery $b%: paused YOLOv8" >> $LOGF
  elif [ $paused = 1 ] && [ $ac = 1 ]; then
    for p in $(pgrep -f "code/train_yolo\.py"); do kill -CONT $p; done; paused=0
    echo "$(date '+%F %T') battery $b% on AC: resumed YOLOv8" >> $LOGF
  fi
  sleep 60
done
echo "$(date '+%F %T') yolo guard: no YOLOv8 training, exit" >> $LOGF
