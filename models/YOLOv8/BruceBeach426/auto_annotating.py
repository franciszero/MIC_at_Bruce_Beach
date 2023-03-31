from PIL import Image
from ultralyticsplus import YOLO, render_result
import glob
import json
import os
from collections import defaultdict

dic = defaultdict(lambda: [])
with open("image_config.txt") as my_file:
    for line in my_file:
        [n, t] = line.split('\t')
        dic[t.rsplit('\n', 1)[0]].append(n)

f = open('./instances_default.json')
j = json.load(f)
all_imgs = [img for img in j.get('images')]

for (k, files) in dic.items():
    # start from 1
    image_id = 1
    annotation_id = 1

    # one annotation for a group of images
    annotations = {"categories": [{"id": 1, "name": "1", "supercategory": ""}], "images": [], "annotations": []}

    imgs = [img for img in all_imgs if img['file_name'] in files]
    for img1 in imgs:
        img = img1.copy()
        annos = [anno for anno in j['annotations'] if anno['image_id'] == img['id']]
        for anno1 in annos:
            anno = anno1.copy()
            anno['id'] = annotation_id
            annotation_id += 1
            anno['image_id'] = image_id
            anno.pop('area', None)
            anno.pop('attributes', None)
            annotations['annotations'].append(anno)
        img_name = img['file_name']
        img['id'] = image_id
        image_id += 1
        img.pop('license', None)
        img.pop('flickr_url', None)
        img.pop('coco_url', None)
        img.pop('date_captured', None)
        annotations['images'].append(img)

    jsonStr = json.dumps(annotations)
    file = open(k + '.json', 'w')
    file.write(jsonStr)
    file.close()
    print("")
