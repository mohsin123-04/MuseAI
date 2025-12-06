import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim 

# define the data transforms
transforms = transforms.Compose((
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
))

# insert the datasets
train_dataset = datasets.ImageFolder(root="../data/train", transform = transform)
test_dataset = datasets.ImageFolder(root="../data/test", transform = transform)





