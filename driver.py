import numpy as np
import glob
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import cv2
import random
import os


class DertResnet50:
    def __init__(self, i, o):
        self.input = i
        self.output = o
        # https://huggingface.co/facebook/detr-resnet-50
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        pass

    def cnt_person(self, filename, save_file=False):
        if filename.endswith("avif"):  # or filename.endswith("png"):
            return -1
        # url = "https://apiwp.thelocal.com/wp-content/uploads/2019/07/e54a92d3593dc8d20b84feb8f2654b2b7e8c3b6983b7be057e57769980052843.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        image = Image.open(self.input + filename).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.65)[0]

        lst = list()
        id2label = self.model.config.id2label
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            item_name = id2label[label.item()]
            if item_name == 'person':
                lst.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])  # x1, y1, x2, y2
        cnt = len(lst)
        if save_file:
            self.__save_result(cnt, filename, lst)
        return cnt

    def __save_result(self, cnt, filename, lst):
        img = cv2.imread(self.input + filename)
        for [x1, y1, x2, y2] in lst:
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
        filename, file_extension = os.path.splitext(filename)
        cv2.imwrite(self.output + filename + '_' + str(cnt) + '_ResNet.' + 'jpg', img)
        pass


class Driver:
    def __init__(self, i='./images/', o='./results/'):
        self.input = i
        self.output = o
        self.resnet = DertResnet50(self.input, self.output)
        pass

    def findAllFile(self):
        for root, ds, fs in os.walk(self.input):
            for f in fs:
                yield f

    def conv2rgb(self, filename):
        img = Image.open(self.input + filename)
        return img.convert('RGB')

    def scanImages(self, overwrite=False):
        for filename in self.findAllFile():
            if filename == '.DS_Store':
                continue
            print('processing : ' + filename)
            if not overwrite:
                f1, f2 = os.path.splitext(filename)
                img_paths = glob.glob(self.output + f1 + '*.jpg')
                if len(img_paths) > 0:
                    print('Skipping.\n')
                    continue
            c1 = self.resnet.cnt_person(filename, save_file=True)
            print('Detected %2d persons with %s' % (c1, 'ResNet'))
            print("")
        pass


if __name__ == '__main__':
    foo = Driver(i='./data/test_data/', o='./data/test_results/')
    # foo = Driver(i='./data/beach_use/', o='./data/result_of_beach_use/')
    foo.scanImages(overwrite=True)
