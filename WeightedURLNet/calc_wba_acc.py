import csv
from matplotlib import pyplot as plt
import os
import sys
import socket
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, roc_curve


test_result_file = sys.argv[1]
preset_weights = ''
if len(sys.argv) > 1:   # if not set, use rarity weights for WBA
    preset_weights = sys.argv[2]
    preset_weights = dict((i,float(w)) for i, w in enumerate(preset_weights.split(',')))
# first start with original
detection_stats = dict()  # {gt-label: {pred-label: count}}
gt_label, preds = [], []
total_cnt = 0
with open(test_result_file) as fh:
    for ln in fh.readlines()[1:]:
        total_cnt += 1
        tmp = ln.strip().split()
        gt = int(tmp[0])
        pred_label = int(tmp[1])
        gt_label.append(gt)
        preds.append(float(tmp[-1]))
        if gt not in detection_stats:
            detection_stats[gt] = dict()
        if pred_label not in detection_stats[gt]:
            detection_stats[gt][pred_label] = 0
        detection_stats[gt][pred_label] += 1

per_label_acc = dict()
WBA_weights = dict()
total_correct = 0
print(f'detection_stats: {detection_stats}')
for gt, preds in detection_stats.items():
    this_total_count = sum(preds.values())   # total count for this gt label
    WBA_weights[gt] = total_cnt / this_total_count
    this_correct = preds[gt] if gt in preds else 0
    per_label_acc[gt] = this_correct / this_total_count
    total_correct += this_correct
    print(f'For gt-label {gt}, pred-label-stats: {preds}, acc: {per_label_acc[gt]}')

print(f'before normalize, WBAweights: {WBA_weights}')
if len(preset_weights) > 0:
    WBA_weights = preset_weights
else: # rarity weights
    WBA_weights = dict((gt, weight/sum(WBA_weights.values())) for gt, weight in preset_weights.items())
print(f'after normalize, WBA weights: {WBA_weights}')

print(f'Overall acc: {total_correct/total_cnt}')
WBA_acc = sum([WBA_weights[gt] * per_label_acc[gt] for gt in WBA_weights.keys()])
print(f'WBA acc: {WBA_acc}')
