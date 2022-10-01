import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torch
import os
import json

classes = {
    "bananapeel": 1,
    "bottle": 2,
    "can": 3,
    "lays": 4,
    "rose": 5
}

batchSize=2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

currentDir = os.getcwd()

annoDir = os.path.join(currentDir,"dataset\\data\\annotations") 
imgDir=os.path.join(currentDir,"dataset\\data\\images")

imgs=[]
for pth in os.listdir(imgDir):
    imgs.append(os.path.join(imgDir,pth))

def maskConverter(segmentation):
    mask = []

    temp_arr = []
    flag = 0
    for i in segmentation:
        temp_arr.append(i)
        flag+=1

        if flag == 2:
            mask.append(temp_arr)
            temp_arr = []
            flag = 0
    
    return  mask


def loadData():
  batch_Imgs=[]
  batch_Data=[]
  for i in range(batchSize):
    # selecting a random image from a random batch
    batch = random.choice(imgs)
    img = random.choice(os.listdir(batch))

    # import the random image
    image = cv2.imread(os.path.join(batch, img))
    image = torch.as_tensor(image)

    # getting batch no
    batch_no = batch.split("\\")[-1]
    # getting annotation.json of the batch
    annotation = "annotation_" + batch_no + ".json"

    with open(os.path.join(annoDir,annotation)) as jsonObject:
        annotation = json.load(jsonObject)
    
    for sex in annotation['images']:
        if sex['file_name'] == img:
            image_id = sex['id']

    boxes = torch.zeros([1,4], dtype=torch.float32)
    masks = []
    for sex in annotation['annotations']:
        if sex['image_id'] == image_id:
            mask = maskConverter(sex['segmentation'][0])
            # !! Convert to polygons through open cv and then make the data accordingly.
            masks.append(mask)
            x,y,w,h = sex['bbox']
            boxes[0] = torch.tensor([x,y,x+w,y+h])

    masks = torch.as_tensor(masks)
    label = annotation['info']['description']

    data = {}
    data['boxes'] = boxes
    data['masks'] = masks
    data['labels'] = torch.tensor(classes[label])
    
    batch_Imgs.append(image)
    batch_Data.append(data)  
  
    return batch_Imgs, batch_Data

#     masks = torch.as_tensor(masks, dtype=torch.uint8)
#     img = torch.as_tensor(img, dtype=torch.float32)
#     data = {}
#     data["boxes"] =  ""
#     data["labels"] =  ""   
#     data["masks"] = ""
#     batch_Imgs.append(image)
#     batch_Data.append(data)  
  
#   batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
#   batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
#   return batch_Imgs, batch_Data

#print(loadData())
