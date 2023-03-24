from matplotlib import pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import math
import pandas as pd
import sys

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

# # load a pretrained model (recommended for training) and train the model
# YOLO('./weights/yolov8x.pt').train(data='BruceBeach39.yaml', epochs=3, imgsz=640)

# # 1 epoch for 10 min and 8hr * 6 =48 epochs during sleep.
# # continue from the last epoch. This is going to be stored in '/train30'
# YOLO('../../runs/detect/train28/weights/last.pt').train(data='BruceBeach39.yaml', epochs=50, imgsz=640)

# # train for 30min for 3 epochs. This is going to be stored in '/train31'
# YOLO('../../runs/detect/train31/weights/last.pt').train(data='BruceBeach39.yaml', epochs=3, imgsz=640)

# # train for 20min for 2 epochs. This is going to be stored in '/train33'
# YOLO('../../runs/detect/train31/weights/last.pt').train(data='BruceBeach39.yaml', epochs=2, imgsz=640)

# # train for 3 epochs. This is going to be stored in '/train34'.
# YOLO('../../runs/detect/train33/weights/last.pt').train(data='BruceBeach39.yaml', epochs=3, imgsz=640)

# # train for 3 epochs. This is going to be stored in '/train35'.
# YOLO('../../runs/detect/train34/weights/last.pt').train(data='BruceBeach39.yaml', epochs=3, imgsz=640)

# # train for 3 epochs. This is going to be stored in '/train36'.
# YOLO('../../runs/detect/train35/weights/last.pt').train(data='BruceBeach39.yaml', epochs=6, imgsz=640)

# Train from the beginning with the right split of training/validation datasets.
YOLO('./weights/yolov8x.pt').train(data='./models/YOLOv8/BruceBeach39.yaml', epochs=1, imgsz=640)

train_result_folder = '??????'
if len(sys.argv) == 2:
    train_result_folder = str(sys.argv[1])  # e.g. "train6"
# merging training results
files = ['../../runs/detect/' + train_result_folder + '/results.csv',
         # '../../runs/detect/train30/results.csv',
         # '../../runs/detect/train31/results.csv',
         # '../../runs/detect/train33/results.csv',
         # '../../runs/detect/train34/results.csv',
         # '../../runs/detect/train35/results.csv',
         # '../../runs/detect/train36/results.csv',
         ]
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
df.columns = [x.strip(' ') for x in df.columns]
ordered_cols = [
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)',
    'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
]
result_figure = plot_metrics(df, ordered_cols, gridspec_cols=5)
result_figure.savefig("YOLOv8_training_results.png")
