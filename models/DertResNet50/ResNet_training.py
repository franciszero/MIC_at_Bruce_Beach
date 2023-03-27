import sklearn.metrics
from PIL import Image
import torch
import fiftyone as fo
from matplotlib import pyplot as plt
from torchvision.transforms import functional as func
from transformers import DetrImageProcessor, DetrForObjectDetection
from models.utils.Consts import MODEL_LIST
from fiftyone.core.dataset import delete_dataset, dataset_exists, list_datasets
from fiftyone.types import COCODetectionDataset
from fiftyone import ViewField as F
from sklearn.metrics import PrecisionRecallDisplay

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

model.train()


# model.
