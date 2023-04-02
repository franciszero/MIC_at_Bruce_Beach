from models.YOLOv8.YOLOv8 import AnnotationYOLOv8
from models.DertResNet50.ResNet import AnnotationResNet
from models.InsightFace.InsightFace import AnnotationInsightFace
from models.SAHI_YOLO.SAHI_YOLOv8 import AnnotationSahiYOLOv8
import sys

# AnnotationInsightFace(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
#                       images='./models/YOLOv8/BruceBeach426/test/images/',
#                       model_name='insightface_pre_BB426',
#                       model_weights='scrfd_person_2.5g.onnx').foo()
AnnotationResNet(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
                 images='./models/YOLOv8/BruceBeach426/test/images/',
                 model_name='DertResNet50_pre_BB426',
                 model_weights="facebook/detr-resnet-50").foo()
AnnotationYOLOv8(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
                 images='./models/YOLOv8/BruceBeach426/test/images/',
                 model_name='YOLOv8x_pre_BB426',
                 model_weights='./models/YOLOv8/weights/yolov8x').foo()
AnnotationSahiYOLOv8(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
                     images='./models/YOLOv8/BruceBeach426/test/images/',
                     model_name='YOLOv8x_pre_BB426_SAHI',
                     model_weights='./models/YOLOv8/weights/yolov8x').foo()
AnnotationYOLOv8(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
                 images='./models/YOLOv8/BruceBeach426/test/images/',
                 model_name='YOLOv8x_BB158_500_t16_b',
                 model_weights='./models/YOLOv8/weights/YOLOv8x_BB158_500_t18_b.pt').foo()
AnnotationYOLOv8(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
                 images='./models/YOLOv8/BruceBeach426/test/images/',
                 model_name='YOLOv8x_BB426_760_t31_b',
                 model_weights='./models/YOLOv8/weights/YOLOv8x_BB426_760_t31_b.pt').foo()
# AnnotationSahiYOLOv8(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
#                      images='./models/YOLOv8/BruceBeach426/test/images/',
#                      model_name='YOLOv8x_BB426_760_t31_b_SAHI',
#                      model_weights='./models/YOLOv8/weights/YOLOv8x_BB426_760_t31_b.pt').foo()
