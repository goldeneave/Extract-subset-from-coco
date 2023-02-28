'''

After get the new set, you may want to verify whether the dataset have errors or not
So this part code will check the new set
I use detectron2 developed by fb to check, before you run the code you should install the library first
the common method is use the clone code, which need pyyaml version == 5.1

I run the code on colab, so modify the code below to install detectron2 and other requirements on your personal device

'''

# !python -m pip install pyyaml==5.1
# import sys, os, distutils.core
# !git clone 'https://github.com/facebookresearch/detectron2'
# dist = distutils.core.run_setup("./detectron2/setup.py")
# !python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
# sys.path.insert(0, os.path.abspath('./detectron2'))

# Then import necessary function
import random

# import some common libraries

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

def create_viz_dicts(json_file, img_dir):
    """create data dictionary to prepare for viz"""

    # load json file
    with open(json_file) as f:
        json_file = json.load(f)
        imgs = json_file['images']
        imgs_anns = json_file['annotations']

    # create dictionary
    dataset_dicts = []
    for idx, v in enumerate(imgs):
        record = {}
        filename = os.path.join(img_dir, v["file_name"])
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # get annotations
        annotations_list = []
        annotations = json_file['annotations'][idx]

        # match image id to annotations image id
        while v['id'] != annotations['image_id']:
            idx += 1
            annotations = json_file['annotations'][idx]

        annotations_list.append(annotations)  # add annotation

        # could be multiple annotations that match, continue looping
        while v['id'] == annotations['image_id']:
            idx += 1
            try:
                annotations = json_file['annotations'][idx]
            except IndexError:
                break
            if v['id'] == annotations['image_id']:
                annotations_list.append(annotations)

        # create objects containing all information
        objs = []
        for anno in annotations_list:
            segmentation = [x + 0.5 for sublist in anno['segmentation'] for x in sublist]
            obj = {
                "bbox": anno['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [segmentation],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs

        # append record to data dict
        dataset_dicts.append(record)

    return dataset_dicts

# in detectron2, you should register dataset before you use it
# the function have 4 parameters, which the first one is the name defined by yourself
# the third part and the last part, you should add the annotation file path and img folder path

from detectron2.data.datasets import register_coco_instances
register_coco_instances("val", {}, "./val_instance.json", "./img/val")
register_coco_instances("train", {}, "./train_instance.json", "./img/train")
MetadataCatalog.get("train").set(thing_classes=["temp"])
metadata = MetadataCatalog.get("train")

dataset_dicts = create_viz_dicts("./train_instance.json", "./img/train")
for d in random.sample(dataset_dicts, 3): # sample 3 at random
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=2)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])