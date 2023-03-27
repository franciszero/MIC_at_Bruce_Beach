from PIL import Image
from ultralyticsplus import YOLO, render_result
import glob
import json
import os

# def foo(fpath, already_read_lst):
#     f = open(fpath)
#     j = json.load(f)
#     ns = [jj['file_name'] for jj in j.get('images')]
#     imgs = [val for val in ns if val in set(labeled_files).difference(set(already_read_lst))]
#     f.close()
#     return imgs
#
#
# bb39 = foo('BB39.json', [])
# bb300 = foo('BB300.json', bb39)
# bb565 = foo('BB565.json', bb39 + bb300)
# dups = [val for val in bb300 if val in bb565]
# diffs = list(set(labeled_files).difference(set(bb300 + bb565 + bb39)))

# one annotation for a group of images
annotations = {"categories": [{"id": 1, "name": "1", "supercategory": ""}], "images": [], "annotations": []}

processed_imgs = []
labeled_files = [os.path.splitext(f)[0].rsplit('/', 1)[1] + ".jpg" for f in glob.glob('./val/labels/*')]
global_image_ids = {}

# start from 1
image_id = 1
annotation_id = 1
for fpath in ['./BB39.json', './BB300.json', './BB565.json']:
    f = open(fpath)
    j = json.load(f)
    ns = [img['file_name'] for img in j.get('images')]
    cur_imgs = [val for val in ns if val in set(labeled_files).difference(set(processed_imgs))]
    processed_imgs += cur_imgs
    for img in cur_imgs:
        global_image_ids[img] = image_id
        image_id += 1

    tmp_imgs = [img for img in j.get('images') if img['file_name'] in cur_imgs]
    tmp_img_ids = [img['id'] for img in tmp_imgs if img['file_name'] in cur_imgs]
    tmp_annos = [anno for anno in j.get('annotations') if anno['image_id'] in tmp_img_ids]

    for img1 in tmp_imgs:
        img = img1.copy()
        img_name = img['file_name']
        img['id'] = global_image_ids[img_name]
        img.pop('license', None)
        img.pop('flickr_url', None)
        img.pop('coco_url', None)
        img.pop('date_captured', None)
        annotations['images'].append(img)

    for anno1 in tmp_annos:
        anno = anno1.copy()
        anno['id'] = annotation_id
        annotation_id += 1
        img_id = anno['image_id']
        img_name = [img['file_name'] for img in tmp_imgs if img['id'] == img_id][0]
        anno['image_id'] = global_image_ids[img_name]
        anno.pop('area', None)
        anno.pop('attributes', None)
        annotations['annotations'].append(anno)

jsonStr = json.dumps(annotations)
file = open('annotation_BB158_val76.json', 'w')
file.write(jsonStr)
file.close()
print("")
