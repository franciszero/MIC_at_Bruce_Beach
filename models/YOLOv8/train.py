import glob
import json

import pandas as pd
import numpy as np
from ultralytics import YOLO

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 3)

trains = glob.glob('../../runs/detect/*')
if not trains:
    train_result_folder = "train"
else:
    train_result_folder = "train" + str(np.array([int(path.rsplit('train', 1)[1])
                                                  if path.rsplit('train', 1)[1] != '' else 0
                                                  for path in trains]).max() + 1)

# YOLOv5 trainin quickstart: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

# Train from the beginning with the right split of training/validation datasets.
with open('config.json', 'r') as f:
    JSON_Obj = json.load(f)
# YOLO('./weights/yolov8x.pt').train(
YOLO(JSON_Obj["weights_file"]).train(
    data=JSON_Obj["data"],
    imgsz=JSON_Obj["imgsz"],
    epochs=JSON_Obj["epochs"],
    batch=JSON_Obj["batch"],
    patience=JSON_Obj["patience"],  # use `patience=0` to disable EarlyStopping
    device=JSON_Obj["device"]
)
