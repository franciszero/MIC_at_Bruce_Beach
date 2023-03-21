import numpy as np
import cv2
import insightface
from fiftyone import dataset_exists, delete_dataset
import fiftyone as fo
from fiftyone.types import COCODetectionDataset
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from fiftyone import ViewField as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        # self.metrics_and_pr_curve()
        pass

    def metrics_and_pr_curve(self):
        high_conf_view = self.ds.view().filter_labels("predictions", F("confidence") > 0.5, only_matches=False)
        results = high_conf_view.evaluate_detections(
            "predictions",
            gt_field="detections",
            eval_key="eval",
            compute_mAP=True,
        )
        # Get the 10 most common classes in the dataset
        counts = self.ds.count_values("detections.detections.label")
        classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]
        # Print a classification report for the top-10 classes
        results.print_report(classes=classes_top10)
        print("ap = ", results.mAP())

        plot = results.plot_pr_curves(classes=["1"])
        # plot.show()

        for sample in self.ds:
            i = None
            # running evaluation
            resFile = self.workspace + "/instances_default.json"
            cocoGt = COCO(resFile)
            cocoDt = cocoGt.loadRes(resFile)

            cocoDt = sample.get_field('predictions')
            cocoEval = COCOeval(sample.detection, sample.predictions, 'bbox')
            cocoEval.params.imgIds = sample.id
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()


        thresh_inds = results._get_iou_thresh_inds(iou_thresh=None)
        precisions = []
        has_thresholds = results.thresholds is not None
        thresholds = [] if has_thresholds else None
        for c in ["1"]:
            class_ind = results._get_class_index(c)
            precisions.append(np.mean(results.precision[thresh_inds, class_ind], axis=0))
            print(thresh_inds, class_ind)
            if has_thresholds:
                thresholds.append(np.mean(results.thresholds[thresh_inds, class_ind], axis=0))

        fig, ax = plt.figure(5, 5)
        plot = PrecisionRecallDisplay(results.precisions, results.recall)
        plot.plot(ax=ax)

# <Sample: {
#     'id': '641625b6f6e7f37e3feacd2e',
#     'media_type': 'image',
#     'filepath': '/Users/francis/Documents/Georgian College/AIDI/GitHub/Smart_Beach_at_Bruce_County/data/batch_20230318_0/images/2022-08-31_09-50-16.jpg',
#     'tags': [],
#     'metadata': <ImageMetadata: {
#         'size_bytes': None,
#         'mime_type': None,
#         'width': 4056,
#         'height': 3040,
#         'num_channels': None,
#     }>,
#     'detections': <Detections: {
#         'detections': [
#             <Detection: {
#                 'id': '641625b6f6e7f37e3feacd2a',
#                 'attributes': {},
#                 'tags': [],
#                 'label': '1',
#                 'bounding_box': [
#                     0.7469773175542406,
#                     0.5924375,
#                     0.050547337278106515,
#                     0.16629276315789474,
#                 ],
#                 'mask': None,
#                 'confidence': None,
#                 'index': None,
#                 'supercategory': '',
#                 'iscrowd': 0,
#                 'occluded': False,
#                 'rotation': 0.0,
#                 'eval': 'tp',
#                 'eval_id': '641625b6f6e7f37e3feacd2c',
#                 'eval_iou': 0.8151426731053592,
#             }>,
#             <Detection: {
#                 'id': '641625b6f6e7f37e3feacd2b',
#                 'attributes': {},
#                 'tags': [],
#                 'label': '1',
#                 'bounding_box': [
#                     0.8174679487179487,
#                     0.5824605263157895,
#                     0.040160256410256404,
#                     0.1741907894736842,
#                 ],
#                 'mask': None,
#                 'confidence': None,
#                 'index': None,
#                 'supercategory': '',
#                 'iscrowd': 0,
#                 'occluded': False,
#                 'rotation': 0.0,
#                 'eval': 'tp',
#                 'eval_id': '641625b6f6e7f37e3feacd2d',
#                 'eval_iou': 0.8254236421052641,
#             }>,
#         ],
#     }>,
#     'predictions': <Detections: {
#         'detections': [
#             <Detection: {
#                 'id': '641625b6f6e7f37e3feacd2c',
#                 'attributes': {},
#                 'tags': [],
#                 'label': '1',
#                 'bounding_box': [
#                     0.7418639053254438,
#                     0.5891447368421052,
#                     0.0589250493096647,
#                     0.175,
#                 ],
#                 'mask': None,
#                 'confidence': 1.0,
#                 'index': None,
#                 'eval_iou': 0.8151426731053592,
#                 'eval_id': '641625b6f6e7f37e3feacd2a',
#                 'eval': 'tp',
#             }>,
#             <Detection: {
#                 'id': '641625b6f6e7f37e3feacd2d',
#                 'attributes': {},
#                 'tags': [],
#                 'label': '1',
#                 'bounding_box': [
#                     0.8163214990138067,
#                     0.5809210526315789,
#                     0.04684418145956608,
#                     0.18092105263157895,
#                 ],
#                 'mask': None,
#                 'confidence': 1.0,
#                 'index': None,
#                 'eval_iou': 0.8254236421052641,
#                 'eval_id': '641625b6f6e7f37e3feacd2b',
#                 'eval': 'tp',
#             }>,
#         ],
#     }>,
#     'eval_tp': 2,
#     'eval_fp': 0,
#     'eval_fn': 0,
# }>