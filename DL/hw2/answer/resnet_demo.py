import os
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pathlib, torch, tqdm


rootdir = pathlib.Path(os.path.dirname(pathlib.Path(__file__)))

#We recommend to use PIL package to read image
#All images read by PIL will be in the shape of Height*Width*3
#Please note that Height and Width of different images may be different

#load training images
train_images = []
train_labels = []
with open(rootdir/'labels'/'train_labels.txt','r') as f:
    for l in f.readlines():
        img_name, label = l.split(' ')
        train_images.append(rootdir/'train'/img_name)
        train_labels.append(int(label))

train_images = np.array(train_images)
train_labels = np.array(train_labels)

#separate 10% from the dataset as the validation set
s = int(len(train_labels)*0.9)
validate_images = train_images[s:]
validate_labels = train_labels[s:]

train_images = train_images[:s]
train_labels = train_labels[:s]
print(f'training size: {len(train_images)}')
print(f'validation size: {len(validate_images)}')

#Below is a demo of using modified Resnet18 model to solve this problem
#for more details of Resnet18 model please refer to this paper:
#He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
#https://arxiv.org/abs/1512.03385



model = models.resnet18(pretrained=True)
print(model)  ##You can check the structure of Resnet18 model


#we only need to modify the structure of model.fc
#change the out_features to 2 to make it a binary classifier
model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=2, bias=False),
                               torch.nn.Softmax(dim=1)
                              )


#Then let's make a dataloader for the Resnet18 model
#the transformation below is the special requirement of Resnet18 model
t = transforms.Compose([
        Image.open,
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#build a dataset class, this is the most commonly 
#used way to build data loader in pytorch
class vgg_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y): 
        super(vgg_dataset, self).__init__()
        self.x = [t(img_path) for img_path in x]
        self.x = torch.stack(self.x)
        self.y = y
        assert self.x.shape[0] == self.y.shape[0], "the input x and y have different size!"
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.x.shape[0]

#Then build a wrapper to make it iterable
class vgg_dataloader(torch.utils.data.DataLoader):
    
    def __init__(self, x, y, batch_size=256, shuffle=False):
        super(vgg_dataloader, self).__init__(dataset=vgg_dataset(x,y), batch_size=batch_size, shuffle=shuffle)


#Now let's train the modified Resnet18 model on our dataset
#Here are some training settings
for param in model.fc.parameters():
    if len(param.shape)>1:
        torch.nn.init.xavier_normal_(param, gain=1.0)
    else:
        torch.nn.init.zeros_(param)
    param.requires_grad = True
    
    
loss_fn = torch.nn.CrossEntropyLoss()   #by default it's the mean CrossEntropyLoss over a batch
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)   #use AdamW optimzier with starting learning rate 1e-3
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=0)   #add a scheduler to gradually decrease learning rate



max_epoch = 100
early_stopping_learning_rate = 1.0e-5

device = "cuda:0"
model = model.to(device)

train_dataloader = vgg_dataloader(train_images, train_labels)
validate_dataloader = vgg_dataloader(validate_images, validate_labels)


def validate_fn(model, validate_dataloader, device):
    l = 0
    c = 0
    for block in validate_dataloader:
        image, key = block
        image = image.to(device)
        key = key.to(device)
        
        prediction = model(image)
        loss = loss_fn(prediction,key)
        l += float(loss)
        c += 1
    
    validation_loss = l/c
    
    return validation_loss

validate_loss_curve = []
for epoch in tqdm.tqdm(range(max_epoch)):
    torch.cuda.empty_cache()
    
    #validation
    #print('validating......')
    epoch_validation_loss = validate_fn(model, validate_dataloader, device)
    validate_loss_curve.append(epoch_validation_loss)

    lr = optimizer.param_groups[0]['lr']

    if lr < early_stopping_learning_rate:
        break

    #save best model
    if scheduler.is_better(epoch_validation_loss, scheduler.best):
        torch.save(model.state_dict(), 'best.pt')

    scheduler.step(epoch_validation_loss)
    #training loop
    #print('training......')
    for block in train_dataloader:
        model.zero_grad()
        image, key = block
        image = image.to(device)
        key = key.to(device)
        
        prediction = model(image) 
        loss = loss_fn(prediction,key)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch%10==0:
        print('%d epoches finished, current validation loss: %f \n'%(epoch, epoch_validation_loss))
        
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, 'resnet_latest.pt')

