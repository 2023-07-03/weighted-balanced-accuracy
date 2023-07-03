import csv
from matplotlib import pyplot as plt
import os
import sys
import socket
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, roc_curve


test_result_file = sys.argv[1]
if len(sys.argv) > 2:
    test_result_file_with_weights = sys.argv[2]
output_folder = '../figures/'
# plot
fig, ax = plt.subplots(1, constrained_layout=True)
fig.set_figheight(3)
fig.set_figwidth(6)
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlabel("False Positive Rate (%)", fontsize=10)  # , color="blue", fontsize=14)
ax.set_ylabel("True Positive Rate (%)", fontsize=10)

keys = ['original', 'weighted']
input_files = {'original': test_result_file, 'weighted': test_result_file_with_weights}
line_colors = {'original': 'red', 'weighted': 'black'}


for kk in keys:
    # first start with original
    detection_stats = {'gt1': [], 'gt-1': []}
    gt_label, preds = [], []
    with open(input_files[kk]) as fh:
        for ln in fh.readlines()[1:]:
            tmp = ln.strip().split()
            gt = int(tmp[0])
            pred_label = int(tmp[1])
            gt_label.append(gt)
            preds.append(float(tmp[-1]))
            detection_stats[f'gt{gt}'].append(pred_label)

    detection_stats_cnt = {"gt1": {"pred=1": detection_stats["gt1"].count(1), "pred=-1": detection_stats["gt1"].count(-1)},
                           "gt-1": {"pred=1": detection_stats["gt-1"].count(1), "pred=-1": detection_stats["gt-1"].count(-1)}}
    print(f'{kk} detection stats: {detection_stats_cnt}')
    print(f'{kk} overall accuracy: {(detection_stats["gt1"].count(1)+detection_stats["gt-1"].count(-1))/(len(detection_stats["gt1"])+len(detection_stats["gt-1"]))}, '
          f'gt1 accuracy: {detection_stats["gt1"].count(1)/len(detection_stats["gt1"])}, '
          f'gt-1 accuracy: {detection_stats["gt-1"].count(-1)/len(detection_stats["gt-1"])}')

    fpr, tpr, thresholds = roc_curve(gt_label, preds)

    ax.plot([x*100 for x in fpr], [x*100 for x in tpr], color=line_colors[kk], label=kk)


    with open(os.path.join(output_folder, f'{kk}.tpr.fpr.csv'), 'w') as fh:
        fh.write('fpr\ttpr\n')
        for it, tp in enumerate(tpr):
            fh.write(f'{fpr[it]}\t{tpr[it]}\n')
ax.legend()
plt.savefig(os.path.join(output_folder, f'URLNet-test-tpr-fpr-compare.png'))
plt.show()
