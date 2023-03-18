from models.YOLOv8 import AnnotationYOLOv8
from models.ResNet import AnnotationResNet
from models.InsightFace import AnnotationInsightFace

batch_path = "./data/batch_20230318_2/"
AnnotationResNet(batch_path, model_id=0).foo()
AnnotationYOLOv8(batch_path, model_id=1).foo()
AnnotationInsightFace(batch_path, model_id=2).foo()
