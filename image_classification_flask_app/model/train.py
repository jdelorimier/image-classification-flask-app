import torch
# import jovian
import torchvision
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,ConcatDataset

from model import MResnet

## data loader
stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

test_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])

### Helper functions
def accuracy(predicted,actual):
    _, predictions = torch.max(predicted,dim=1)
    return torch.tensor(torch.sum(predictions==actual).item()/len(predictions))

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)


class ToDeviceLoader:
    def __init__(self,data,device):
        self.data = data
        self.device = device
        
    def __iter__(self):
        for batch in self.data:
            yield to_device(batch,self.device)
            
    def __len__(self):
        return len(self.data)

### Training 
@torch.no_grad()
# makes it so that gradients are not calculated when evaluating
def evaluate(model,test_dl):
    # checks progress of our model using our test/val set.
    model.eval() # sets model to evaluation mode and turns off processes from the model that are only used for training
    outputs = [model.validation_step(batch) for batch in test_dl]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    # keeps track of lr since its going to change throughout the training phase.
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs,train_dl,test_dl,model,optimizer,max_lr,weight_decay,scheduler,grad_clip=None):
    # is what we’ll use to train the model and to keeps track of our models’ progress throughout the training
    torch.cuda.empty_cache()
    
    history = []
    
    optimizer = optimizer(model.parameters(),max_lr,weight_decay=weight_decay)
    
    scheduler = scheduler(optimizer,max_lr,epochs=epochs,steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train() # opposite of model.eval(), prepares the model for training mode
        
        train_loss = []
        
        lrs = []
        
        for batch in train_dl:
            loss = model.training_step(batch)
            
            train_loss.append(loss)
            
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(),grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
            lrs.append(get_lr(optimizer))
        result = evaluate(model,test_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs
        
        model.epoch_end(epoch,result)
        history.append(result)
        
    return history

if __name__ == '__main__':
    model = MResnet(in_channels=3,num_classes=100)
    train_data = CIFAR100(download=True,root="./data",transform=train_transform)
    test_data = CIFAR100(root="./data",train=False,transform=test_transform)

    BATCH_SIZE=128
    # Num_workers generates batches in parallel. It essentially prepares the next n batches after a batch has been used.
    # Pin_memory helps speed up the transfer of data from the CPU to the GPU.
    train_dl = DataLoader(train_data,BATCH_SIZE,num_workers=4,pin_memory=True,shuffle=True)
    test_dl = DataLoader(test_data,BATCH_SIZE,num_workers=4,pin_memory=True)
    device = get_device()
    print(f"training on {device}")
    train_dl = ToDeviceLoader(train_dl,device)
    test_dl = ToDeviceLoader(test_dl,device)
    ### Run it ### 
    epochs = 25
    # optimizer = torch.optim.SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    max_lr=0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    scheduler = torch.optim.lr_scheduler.OneCycleLR

    ### train and save model
    print("starting initial evaluation")
    history = [evaluate(model,test_dl)]
    history += fit(epochs=epochs,train_dl=train_dl,test_dl=test_dl,model=model,optimizer=optimizer,max_lr=max_lr,grad_clip=grad_clip,
                  weight_decay=weight_decay,scheduler=torch.optim.lr_scheduler.OneCycleLR)

    print("evaluation finished. Saving model params")
    torch.save(model.state_dict(),"./model/net_wnb.pth")