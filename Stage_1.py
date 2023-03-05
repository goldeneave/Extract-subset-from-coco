'''

In the first stage you should prepare your own dataset
After that, I will try to extract specific category and then write a corresponding xml file
the code below I have tested on colab, maybe I will attach the link also

'''

from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw


# the path defined below will save the new generated file, xml
# usually we will separate the dataset into train and val and saved in 2 folders
# so dataDir should contain train and val data
save_path = "path/to/new/files"
img_dir = save_path + 'images/'
anno_dir = save_path + 'annotations/'

datasets_list = ['train', 'val']
dataDir = 'root/dir/contain/img_and_ann'

classes_names = ["specific cat"]   # There you should add the category you want extract

# define xml head
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''

def mkr(path):
    if not os.path.exists(path):
        os.makedirs(path)

def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes

def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco,dataset,filename,objs):
    # transfer image file into xml file, which means a corresponding relationship
    # for example, COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    dst_anno_dir = os.path.join(anno_dir, dataset)
    mkr(dst_anno_dir)
    anno_path=dst_anno_dir + '/' +filename[:-3]+'xml'
    img_path=dataDir + dataset+'/'+filename
    print("img_path: ", img_path)
    dst_img_dir = os.path.join(img_dir, dataset)
    mkr(dst_img_dir)
    dst_imgpath=dst_img_dir+ '/' + filename
    print("dst_imgpath: ", dst_imgpath)
    img=cv2.imread(img_path)

    #if (img.shape[2] == 1):
    #    print(filename + " not a RGB image")
    #    return

    shutil.copy(img_path, dst_imgpath)

    head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path,head, objs, tail)


def showimg(coco,dataset,img,classes,cls_id,show=True):
    global dataDir
    I=Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name=classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs

for dataset in datasets_list:
    annFile='/content/annotations/instances_attributes_val2020.json'.format(dataDir,dataset)

    #use COCO API to initial initialization annotation file
    coco = COCO(annFile)
    '''
    When the COCO object is created, the following information will be output:
    loading annotations into memory...
    Done (t=0.81s)
    creating index...
    index created!
    So far, the JSON script has been parsed and the images are associated with the corresponding annotated data.
    '''
    # get all category in dataset
    classes = id2name(coco)
    print(classes)
    #[1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    # exit()
    for cls in classes_names:
        # get ID
        cls_id=coco.getCatIds(catNms=[cls])
        img_ids=coco.getImgIds(catIds=cls_id)
        print(cls,len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs=showimg(coco, dataset, img, classes,classes_ids,show=False)
            print(objs)
            save_annotations_and_imgs(coco, dataset, filename, objs)

# after run the code, anno_dir will new a folder and contain xml file
