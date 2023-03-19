from models.YOLOv8 import AnnotationYOLOv8
from models.ResNet import AnnotationResNet
from models.InsightFace import AnnotationInsightFace
import sys

batch_path = None
if len(sys.argv) == 2:
    batch_path = "./data/%s/" % str(sys.argv[1])  # e.g. "./data/batch_20230318_0/"
else:
    exit(250)

AnnotationResNet(batch_path, model_id=0).foo()
AnnotationYOLOv8(batch_path, model_id=1).foo()
AnnotationInsightFace(batch_path, model_id=2).foo()
