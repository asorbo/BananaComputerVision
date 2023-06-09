import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import copy
import json


CLASSES = ('unripe', 'ripe', 'overripe', 'rotten')
PATH = "models/Res50_lr=0.001_batchSize=5.pth"

#The custom image dataset loader
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.IntTensor([label for label in self.img_labels.iloc[idx, 1:7]])
        label = torch.argmax(label) #hot encode the tensor 
        #label = torch.Tensor(self.img_labels.loc[img_path,:].values)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    img = torch.Tensor.cpu(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_dataloader(loader):
    # get some random training images
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    #print the tensor label
    #print(' '.join(f'{labels[j]}' for j in range(batch_size)))

    # print the human label
    images_labels = []
    for image_index in range(BATCH_SIZE):
        current_labels = []
        i = 0
        for label in torch.nn.functional.one_hot(labels[image_index].to(torch.int64)):
            if label == 1:
                current_labels.append(CLASSES[i])
            i+=1
        images_labels.append('-'.join(current_labels))
    print(" | ".join(images_labels))    

def random_predict(loader):
    dataiter = iter(loader)
    data = next(dataiter)
    images, labels = data[0].to(device), data[1].to(device)

    print('GroundTruth: ', ' '.join(f'{CLASSES[labels[j]]:5s}' for j in range(1)))

    net = load(PATH)
    outputs = net(images)
    outputs.to(device)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{CLASSES[predicted[j]]:5s}'
                                for j in range(1)))
    # print images
    imshow(torchvision.utils.make_grid(images.cpu()))

#test overall accuracy
#param averaged_minibatch_training_loss replaces the training loss if available 
def test_overall(net, dataloader, averaged_minibatch_training_loss = None):
    #do this at each epoch (then plot)
    correct = 0
    total = 0
    misclassifications = {}
    misclassified_images = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images).to(device)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            count_misclassifications(predicted, labels, misclassifications, misclassified_images, images)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(misclassifications)
    imshow(torchvision.utils.make_grid(misclassified_images))
    return {"accuracy": accuracy, "loss": (averaged_minibatch_training_loss if averaged_minibatch_training_loss else loss.item())}

def count_misclassifications(predicted, labels, misclassifications, misclassified_images, images):
    i = 0
    for prediction in predicted:
        if int(prediction) != int(labels[i]):
            index = f"True:{CLASSES[labels[i]]}_Predicted:{CLASSES[predicted[i]]}"
            if index not in misclassifications:
                misclassifications.update({index: 0})
            misclassifications[index] += 1
            if CLASSES[labels[i]] == "ripe" and CLASSES[predicted[i]] == "overripe":
                misclassified_images.append(images[i])

# The (convolutional) neural network baseline
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # slide the filter with 3 in channels, 6 out channels and stride size = 5
        self.pool = nn.MaxPool2d(2, 2)  # max pooling with a 2 by 2 filter
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 200) # fully connected layer
        self.fc2 = nn.Linear(200, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) # ReLU -> activation function that introduces the property of non-linearity to a deep learning model and solves the vanishing gradients issue. 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    
    
def load(path):
    print("Loading from ", path)
    net = Net()
    if ("CustomCNN" in path):
        net = Net().to(device)
    else:
        net = torchvision.models.resnet50(weights='DEFAULT').to(device)
    net.load_state_dict(torch.load(path))    
    print("Loaded")
    return net


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('GPU is being used' if torch.cuda.is_available() else '!!! ⚠ CPU is being used ⚠ !!!')

transform = transforms.Compose(
    [transforms.ToPILImage(),transforms.ToTensor(), #PIL -> Python Imaging Library format
     transforms.Resize(64),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   #to make model training less sensitive to the scale of features

#dataset = CustomImageDataset("/home/ago/Documents/Thesis/BananaComputerVision/photos/_classes.csv", 
#                                            "/home/ago/Documents/Thesis/BananaComputerVision/photos/",
#                                            transform=transform)
dataset = CustomImageDataset("/home/ago/Documents/Thesis/BananaComputerVision/TheDataset/classesTotal.csv", 
                                            "/home/ago/Documents/Thesis/BananaComputerVision/TheDataset/",
                                            transform=transform)
splits = torch.utils.data.random_split(dataset, [.7, .15, .15], torch.Generator().manual_seed(30))
#loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
loader = torch.utils.data.DataLoader(splits[2], batch_size=5,
                                                     shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss() 


net = load(PATH)
print(test_overall(net, loader))
#random_predict(loader)