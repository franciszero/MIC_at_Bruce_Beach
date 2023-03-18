from PIL import Image
from fiftyone import dataset_exists, delete_dataset
from fiftyone.types import COCODetectionDataset
from ultralyticsplus import YOLO, render_result
import os
import fiftyone as fo
from torchvision.transforms import functional as func

from models.utils.Consts import MODEL_LIST, LABEL_PERSON


class AnnotationYOLOv8:
    def __init__(self, workspace, model_id):
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

        self.model = YOLO('models/weights/yolov8x')
        self.model.overrides['conf'] = 0.5  # NMS confidence threshold
        self.model.overrides['iou'] = 0.5  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 1000  # maximum number of detections per image
        pass

    def foo(self):
        with fo.ProgressBar() as pb:
            for sample in pb(self.new_ds.view()):
                if self.ds.values("filepath").__contains__(sample.filepath):
                    print("skip duplicated file: %s" % sample.filename)
                    continue

                image = Image.open(sample.filepath).convert('RGB')
                w, h = image.size

                results = self.model.predict(sample.filepath, stream=True, verbose=False)
                detections = []
                for result in results:
                    for bbox in result.boxes:
                        if int(bbox.cls.numpy()[0]) == 0:
                            [[x1, y1, x2, y2, conf, label]] = bbox.boxes.numpy()
                            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
                            detections.append(
                                fo.Detection(
                                    label=LABEL_PERSON,
                                    bounding_box=rel_box,
                                    confidence=conf
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
