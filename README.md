# Extract-subset-from-coco
In the past days, I have tried many methods to extract specific data from coco-format dataset. 

Some dataset may not launched by coco officially, but any other dataset in coco-format, the dataset I used is FashionPedia, which contains many accessory imgs.
I now share some experience and related code in the repo

### The repo will attach 3 py file, or more, but generally could divided into 3 stage
So the code will named Stage_{}, each file I will try to write some annotations

## Stage_1.py
In the py file, the code will use your own dataset to extract specific category and write a corresponding xml file for each img

## Stage_2.py
In the stage, the snippet code will use the xml file generated in Stage_1 to combine a coco format json file
If you use coco official dataset(coco_2017), the image name is a value from a certain number, so you could also named the variable  *file_name*  from the minimum value, and every loop the value will +1

## Stage_3.py
If you only want get a coco format annotation, the first 2 stage is enough. In Stage_3, the code will also generate a YOLO format annotation. The txt file contain the following contents:   <*object-class*>  <*x*>  <*y*>  <*width*>  <*height*> 

where <*object-class*> is an integer representing the class of the object, <*x*> and <*y*> are the normalized coordinates of the center of the bounding box, and <*width*> and <*height*>  are the normalized dimensions of the bounding box

## verify.py
After generate a new data subset, you could also check if you make it in a correct format. For me, I use detectron2 to check its format.
Before use it, install it, I also attach code in the py file, if your device is Windows, you may refer to the official repo.
