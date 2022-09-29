import cv2
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from train import EPOCHS

# loading device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# loading model 
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# updating the number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=5)

# loading the model that we trained
model.load_state_dict(torch.load(str(EPOCHS)+".torch"))
model.to(device)
model.eval()

# test image conversion - CHANGE LATER
images = cv2.imread('ENTER PATH HERE')
images = torch.as_tensor(images, dtype= torch.float32).unsqueeze(0)
images = images.swapaxes(1, 3).swapaxes(2, 3)
images = list(image.to(device) for image in images)

# make prediction
with torch.no_grad():
    pred = model(images)
    
#========================================== Output Image processing ==============================================================

im = images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)

im2 = im.copy()

print(pred) # debugging remove later!
