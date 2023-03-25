import glob

from matplotlib import pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import math
import pandas as pd
import sys
import json
import numpy as np

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 3)


def plot_metrics(dfx, column_names_to_plot, gridspec_cols=2, idx=0):
    assert (gridspec_cols >= 2)  # not allowed single col gridspec plot
    assert (len(column_names_to_plot) / gridspec_cols > 1)  # not allowed single row gridspec plot
    gridspec_rows = math.ceil(len(column_names_to_plot) / gridspec_cols)

    fig, axes = plt.subplots(gridspec_rows, gridspec_cols, figsize=(6 * gridspec_cols, 4 * gridspec_rows))
    for i in range(gridspec_rows):
        for j in range(gridspec_cols):
            if idx < len(column_names_to_plot):
                c = column_names_to_plot[idx]
                idx += 1
                f = dfx[[c]]
                ax0 = axes[i][j]
                sns.lineplot(f, markers=True, ax=ax0, color='brown')
                ax0.set_title(c, size=16)
    fig.suptitle('Training result metrics', size=26)
    return fig


# YOLOv5 trainin quickstart: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

# Train from the beginning with the right split of training/validation datasets.
with open('config.json', 'r') as f:
    JSON_Obj = json.load(f)
YOLO('./weights/yolov8x.pt').train(
    data=JSON_Obj["data"],
    imgsz=JSON_Obj["imgsz"],
    epochs=JSON_Obj["epochs"],
    batch=JSON_Obj["batch"],
    patience=JSON_Obj["patience"],  # use `patience=0` to disable EarlyStopping
)

trains = glob.glob('./../../runs/detect/*')
if not trains:
    train_result_folder = "train"
else:
    train_result_folder = "train" + str(np.array([int(path.rsplit('train', 1)[1])
                                                  if path.rsplit('train', 1)[1] != '' else 0
                                                  for path in trains]).max() + 1)
# merging training results
files = ['./runs/detect/' + train_result_folder + '/results.csv',
         # '../../runs/detect/train30/results.csv',
         # '../../runs/detect/train31/results.csv',
         # '../../runs/detect/train33/results.csv',
         ]
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
df.columns = [x.strip(' ') for x in df.columns]
ordered_cols = [
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)',
    'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
]
result_figure = plot_metrics(df, ordered_cols, gridspec_cols=5)
result_figure.savefig("YOLOv8_training_results.png")
