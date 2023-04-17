# MIC_at_Bruce_Beach
MIC_at_Bruce_Beach

# Preparation

1. Setup venv requirements.txt: `pip install -r requirements.txt`
2. File system structure
```
./
├── data
│   └── images                            # Images for people counting
│   └── img_class_ground_truth
│       └── img_class_ground_truth.py     # Model validation
│       └── ModelPredictionsComparison.py # Model comparison
│── models
│   └── YOLOv8
│       └── train.py                      # Model Training
│   └── quickstart.py                     # A quickstart to get understand of the model outputs
│   └── SAHI_YOLO
│       └── SAHI_YOLOv8.py                # SAHI+YOLOv8 implementation
│── Prediction.py                         # People counting
│── run_application.py                    # Start FiftyONE server 
│── update_annotations_incrementally.py   # Update predictions to FiftyONE
```
3. Prepare images and put it into `./data/images/*`
4. Execute `./Prediction.py` and the results are going to be printed in stdout. Modify the output for special scenarios if needed.
