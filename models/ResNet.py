from PIL import Image
import torch
import fiftyone as fo
from torchvision.transforms import functional as func
from transformers import DetrImageProcessor, DetrForObjectDetection
from models.utils.Consts import MODEL_LIST
from fiftyone.core.dataset import delete_dataset, dataset_exists, list_datasets
from fiftyone.types import COCODetectionDataset


class AnnotationResNet:
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
        self.new_ds.persistent = False

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def foo(self):
        # Get class list
        classes = self.new_ds.default_classes
        # Add predictions to samples
        with fo.ProgressBar() as pb:
            for sample in pb(self.new_ds.view()):
                if self.ds.values("filepath").__contains__(sample.filepath):
                    print("skip duplicated file: %s" % sample.filename)
                    continue

                image = Image.open(sample.filepath).convert('RGB')
                image = func.to_tensor(image).to('cpu')
                c, h, w = image.shape

                inputs = self.processor(images=image, return_tensors="pt")
                outputs = self.model(**inputs)
                # convert outputs (bounding boxes and class logits) to COCO API
                target_sizes = torch.tensor([image.size()[1:]])  # tensor([[3040, 4056]])  order: x,y
                results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.65)[0]

                # Convert detections to FiftyOne format
                detections = []
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    # Convert to [top-left-x, top-left-y, width, height]
                    # in relative coordinates in [0, 1] x [0, 1]
                    x1, y1, x2, y2 = box
                    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                    item_name = self.model.config.id2label[label.item()]
                    if item_name == 'person':
                        detections.append(
                            fo.Detection(
                                label=classes[label],
                                bounding_box=rel_box,
                                confidence=score
                            )
                        )

                sample["predictions"] = fo.Detections(detections=detections)
                # sample.save()  # save predictions to dataset
                self.ds.add_sample(sample)
                print("append new file: %s" % sample.filename)

        # # for visualization only
        # session = fo.launch_app(self.ds)
        # session.wait()
        pass
