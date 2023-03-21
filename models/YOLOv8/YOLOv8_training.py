from ultralytics import YOLO

# YOLOv5 trainin quickstart: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

# # load a pretrained model (recommended for training) and train the model
# YOLO('./weights/yolov8x.pt').train(data='BruceBeach39.yaml', epochs=3, imgsz=640)

# 1 epoch for 10 min and 8hr * 6 =48 epochs during sleep.
# continue from the last epoch. This is going to store in '/train30'
YOLO('../../runs/detect/train28/weights/last.pt').train(data='BruceBeach39.yaml', epochs=50, imgsz=640)
