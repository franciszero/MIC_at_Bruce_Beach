import numpy as np
import cv2
import insightface
from fiftyone import dataset_exists, delete_dataset
import fiftyone as fo
from fiftyone.types import COCODetectionDataset

from models.utils.Consts import MODEL_LIST, LABEL_PERSON

assert insightface.__version__ >= '0.4'


class AnnotationInsightFace:
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

        self.detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
        self.detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
        pass

    def foo(self):
        with fo.ProgressBar() as pb:
            for sample in pb(self.new_ds.view()):
                if self.ds.values("filepath").__contains__(sample.filepath):
                    print("skip duplicated file: %s" % sample.filename)
                    continue
                # image = Image.open(sample.filepath).convert('RGB')
                image = cv2.imread(sample.filepath)
                h, w, c = image.shape

                bboxes, kpss = self.detector.detect(image)
                bboxes = np.round(bboxes[:, :4]).astype(int)
                kpss = np.round(kpss).astype(int)
                kpss[:, :, 0] = np.clip(kpss[:, :, 0], 0, image.shape[1])
                kpss[:, :, 1] = np.clip(kpss[:, :, 1], 0, image.shape[0])
                vbboxes = bboxes.copy()
                vbboxes[:, 0] = kpss[:, 0, 0]
                vbboxes[:, 1] = kpss[:, 0, 1]
                vbboxes[:, 2] = kpss[:, 4, 0]
                vbboxes[:, 3] = kpss[:, 4, 1]

                detections = []
                for bbox in bboxes:
                    [x1, y1, x2, y2] = bbox
                    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
                    detections.append(
                        fo.Detection(
                            label=LABEL_PERSON,
                            bounding_box=rel_box,
                            confidence=1.0
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
