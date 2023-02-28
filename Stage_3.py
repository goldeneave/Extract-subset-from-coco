'''

Actually you have finish the task to generate coco-format json file and extract corresponding img from src path to dst path
The step aimed to transfer xml file into txt file, which will generate label infos

You could also skip the step, after running the code in first stage, you already get the necessary infos to train a model

'''

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ['specific category']  # What you have added in Stage_1


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    # you should execute this part two times
    # one times for train set, the other for val set
    # the path you set here will influence the loop below
    # in_file means the path where you save your xml file
    # out_file means the path where you will save your txt file
    in_file = open('./path/to/xml/%s.xml' % (image_id))
    out_file = open('./path/to/save/txt/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        print(cls)
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# set path
# the data_path means where you save dst imgs
#
data_path = '/home/model_trainging/dataset/coco_DL/images/val2014'
img_names = os.listdir(data_path)

list_file = open('/home/model_trainging/dataset/coco_DL/class_val.txt', 'w')
for img_name in img_names:
    if not os.path.exists('/home/model_trainging/dataset/coco_DL/labels/val2014'):
        os.makedirs('/home/model_trainging/dataset/coco_DL/labels/val2014')

    list_file.write('/home/model_trainging/dataset/coco_DL/images/val2014/%s\n' % img_name)
    image_id = img_name[:-4]
    convert_annotation(image_id)

list_file.close()