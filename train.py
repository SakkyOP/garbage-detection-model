import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
import torchvision.models.segmentation
import torch
from dataset import loadData
import torchvision

EPOCHS = 2000

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#load the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes= 5)

# load model on training device
model.to(device)

# set up optimization for the model training
optimizer = torch.optim.AdamW(params= model.parameters(), lr=1e-5)

# setting model to training mode
model.train()

for i in range(EPOCHS+1):
    # import the training images and image details
    images, target = loadData()
    
    # loading images and targets to our proccessing device
    images = list(image.to(device) for image in images)
    target=[{k: v.to(device) for k,v in t.items()} for t in target]
    
    # training starts here!
    optimizer.zero_grad()
    loss_dict = model(images, target) # returns all the losses in class , bounding boss and mask
    losses = sum(loss for loss in loss_dict.values()) # sum of all the losses which will be used to reset the weights of the model using backpropagation
    
    # losses are passed back to the net to update the weights using backpropagation
    losses.backward()
    optimizer.step()
    
    # every 1000 epochs save the state of the model
    if i%1000 == 0:
        torch.save(model.state_dict(), str(i)+".torch")
        print("Saved model to:", str(i)+".torch")
        
    
    
    
