#!/bin/zsh
# Fold-0 pilot: two YOLOv8s detectors trained one after the other (never in parallel),
# then predictions on CPU.  Test images are only read by predict_yolo.py, after training.
set -e
cd /Users/uyen/Downloads/BONEVQA/yolo_pilot
PY=./venv/bin/python
EPOCHS=${EPOCHS:-60}
FA="datasets/fa_fold0/images/val datasets/fa_fold0/images/cal datasets/fa_fold0/images/test"
BT="datasets/btxrd_pilot/images/val datasets/btxrd_pilot/images/test datasets/btxrd_pilot/images/external"

date "+%F %T start fa_only"
$PY code/train_yolo.py --name fa_os3 --data datasets/fa_fold0/data_fa_os3.yaml --epochs $EPOCHS --patience 20 --device mps > logs/train_fa_os3.log 2>&1
date "+%F %T predict fa_only"
$PY code/predict_yolo.py --weights runs/yolo/fa_os3/weights/best.pt --out runs/pred/fa_only --dirs ${=FA} --device cpu > logs/predict_fa_only.log 2>&1

date "+%F %T start fa_btxrd"
$PY code/train_yolo.py --name fa_btxrd_os3 --data datasets/data_fa_btxrd_os3.yaml --epochs $EPOCHS --patience 20 --device mps > logs/train_fa_btxrd_os3.log 2>&1
date "+%F %T predict fa_btxrd"
$PY code/predict_yolo.py --weights runs/yolo/fa_btxrd_os3/weights/best.pt --out runs/pred/fa_btxrd --dirs ${=FA} ${=BT} --device cpu > logs/predict_fa_btxrd.log 2>&1
# the FracAtlas-only detector on BTXRD: how often does a fracture detector fire on tumour films?
$PY code/predict_yolo.py --weights runs/yolo/fa_os3/weights/best.pt --out runs/pred/fa_only_on_btxrd --dirs ${=BT} --device cpu > logs/predict_fa_only_on_btxrd.log 2>&1
date "+%F %T done"
