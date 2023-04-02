from models.YOLOv8.YOLOv8 import AnnotationYOLOv8
# from models.DertResNet50.ResNet import AnnotationResNet
# from models.InsightFace.InsightFace import AnnotationInsightFace
# from models.SAHI_YOLO.SAHI_YOLOv8 import AnnotationSahiYOLOv8
import sys

# AnnotationInsightFace(batch_path, model_id=2).foo()
# AnnotationResNet(batch_path, model_id=0).foo()
# AnnotationYOLOv8(batch_path, model_id=1, model_weights='./models/YOLOv8/weights/yolov8x.pt').foo()
# AnnotationSahiYOLOv8(batch_path, model_id=3, model_weights='./models/YOLOv8/weights/yolov8x.pt').foo()
# AnnotationYOLOv8(batch_path, model_id=4, model_weights='./runs/detect/train18/weights/best.pt').foo()
# AnnotationYOLOv8(batch_path, model_name='YOLOv8x_BB158_1000_t24_l',
#                  model_weights='./runs/detect/train24/weights/last.pt').foo()
AnnotationYOLOv8(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
                 images='./models/YOLOv8/BruceBeach426/test/images/',
                 model_name='YOLOv8x_BB426_pre',
                 model_weights='./models/YOLOv8/weights/yolov8x').foo()
AnnotationYOLOv8(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
                 images='./models/YOLOv8/BruceBeach426/test/images/',
                 model_name='YOLOv8x_BB158_500_t16_b',
                 model_weights='./models/YOLOv8/weights/YOLOv8x_BB158_500_t16_b.pt').foo()
AnnotationYOLOv8(annotations='./models/YOLOv8/BruceBeach426/test/test.json',
                 images='./models/YOLOv8/BruceBeach426/test/images/',
                 model_name='YOLOv8x_BB426_760_t31_b',
                 model_weights='./models/YOLOv8/weights/YOLOv8x_BB426_760_t31_b.pt').foo()
# AnnotationSahiYOLOv8(batch_path, model_id=5, model_weights='./runs/detect/train18/weights/best.pt').foo()
