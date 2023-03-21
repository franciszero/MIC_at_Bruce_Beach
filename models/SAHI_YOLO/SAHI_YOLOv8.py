from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.utils.cv import read_image, visualize_object_predictions
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from PIL import Image
from fiftyone import dataset_exists, delete_dataset
from fiftyone.types import COCODetectionDataset
from ultralyticsplus import YOLO, render_result
import fiftyone as fo
from sahi.slicing import slice_image

from models.utils.Consts import MODEL_LIST, LABEL_PERSON


class AnnotationSahiYOLOv8:
    def __init__(self, workspace, model_id, model_weights='./weights/yolov8x.pt'):
        self.workspace = workspace
        if dataset_exists(MODEL_LIST[model_id]):
            self.ds = fo.load_dataset(MODEL_LIST[model_id])
        else:
            self.ds = fo.Dataset(MODEL_LIST[model_id])
            self.ds.persistent = True

        new_ds_name = MODEL_LIST[model_id] + "_1"
        if dataset_exists(new_ds_name):
            delete_dataset(new_ds_name)
        self.new_ds = fo.Dataset.from_dir(dataset_type=COCODetectionDataset,
                                          data_path=workspace + "../images/",
                                          labels_path=workspace + "/instances_default.json",
                                          name=new_ds_name)

        model_weights = './weights/yolov8x.pt'
        YOLO(model_weights)
        self.model = Yolov8DetectionModel(model_path=model_weights)

    def foo(self):
        with fo.ProgressBar() as pb:
            for sample in pb(self.new_ds.view()):
                if self.ds.values("filepath").__contains__(sample.filepath):
                    print("skip duplicated file: %s" % sample.filename)
                    continue

                image = Image.open(sample.filepath).convert('RGB')
                w, h = image.size

                results = get_sliced_prediction(
                    image,
                    detection_model=self.model,
                    slice_height=int(w/3),
                    slice_width=int(w/3),
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    verbose=0,
                ).object_prediction_list

                results_humann = list([])
                for i in range(len(results)):
                    pred = results[i]
                    if pred.category.name == 'person':
                        results_humann.append(pred)

                detections = []
                for result in results_humann:
                    [x1, y1, xw, yh] = result.bbox.to_coco_bbox()
                    rel_box = [x1 / w, y1 / h, xw / w, yh / h]
                    detections.append(
                        fo.Detection(
                            label=LABEL_PERSON,
                            bounding_box=rel_box,
                            confidence=result.score.value
                        )
                    )
                sample["predictions"] = fo.Detections(detections=detections)
                # sample.save()  # save predictions to dataset
                self.ds.add_sample(sample)
                print("load new file: %s" % sample.filename)

        # # for visualization only
        # session = fo.launch_app(self.ds)
        # session.wait()
        pass

# model_local_path = './weights/yolov8x.pt'
# YOLO(model_local_path)
# model = Yolov8DetectionModel(model_path=model_local_path)
#
# image = read_image('../../data/images/Picture2.png')
#
# result = get_sliced_prediction(
#     image,
#     detection_model=model,
#     slice_height=2048,
#     slice_width=2048,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
#     verbose=2,
# )
#
# results_human = list([])
# for i in range(len(result.object_prediction_list)):
#     pred = result.object_prediction_list[i]
#     if pred.category.name == 'person':
#         results_human.append(pred)
#
# visualization_result = visualize_object_predictions(
#     image,
#     object_prediction_list=results_human,
#     output_dir="",
#     file_name="",
# )
#
# result.object_prediction_list = results_human
# result.export_visuals(export_dir='./', file_name='123123123123')

# # Jupyter Notebook only
# from IPython.display import Image
# Image("./Picture1_visual.png")
