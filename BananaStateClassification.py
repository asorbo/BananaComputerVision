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


BATCH_SIZE = 4
CLASSES = ('freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe')
EPHOCS = 2
PATH = './cifar_net.pth'


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

#The (convolutional) neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #slide the filter with 3 in channels, 6 out channels and stride size = 5
        self.pool = nn.MaxPool2d(2, 2)  #max pooling with a 2 by 2 filter
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 200) #fully connected layer
        self.fc2 = nn.Linear(200, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) # ReLU -> activation function that introduces the property of non-linearity to a deep learning model and solves the vanishing gradients issue. 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
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

    #print the human label
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

#train the net
def train():
    for epoch in range(EPHOCS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels) #use Cross entropy loss to see how far we are from the actual labels
            loss.backward() #calculate the new weights (computes the partial derivative of the output f with respect to each of the input variables)
            optimizer.step()    #apply the weights now 
            

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

def random_predict():
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    print('GroundTruth: ', ' '.join(f'{CLASSES[labels[j]]:5s}' for j in range(4)))

    net = load(PATH)
    outputs = net(images)
    outputs.to(device)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{CLASSES[predicted[j]]:5s}'
                                for j in range(4)))
    # print images
    imshow(torchvision.utils.make_grid(images))

#test overall accuracy
def test_overall_accuracy():
    #do this at each epoch (then plot)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#test predictions for each class
def test_classes_accuracy():
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def save(path):
    print("Saving to ", path)
    torch.save(net.state_dict(), path)
    print("Saved")

def load(path):
    print("Loading from ", path)
    net = Net()
    net.load_state_dict(torch.load(path))
    print("Loaded")
    return net


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('GPU is being used' if torch.cuda.is_available() else '!!! ⚠ CPU is being used ⚠ !!!')

transform = transforms.Compose(
    [transforms.ToPILImage(),transforms.ToTensor(), #PIL -> Python Imaging Library format
     transforms.Resize(64),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   #to make model training less sensitive to the scale of features

trainset = CustomImageDataset("/home/ago/Documents/Thesis/BananaComputerVision/MonoClassDataset/train/_classes.csv", 
                                   "/home/ago/Documents/Thesis/BananaComputerVision/MonoClassDataset/train/",
                                   transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = CustomImageDataset("/home/ago/Documents/Thesis/BananaComputerVision/MonoClassDataset/test/_classes.csv", 
                                   "/home/ago/Documents/Thesis/BananaComputerVision/MonoClassDataset/test/",
                                   transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)


test_dataloader(trainloader)

net = Net()
net.to(device)  #to the GPU
criterion = nn.CrossEntropyLoss()   #Cross entropy loss (to see how different the prediction is from the ground truth)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #stocastic gradient descent

train()
save(PATH)
net = load(PATH)
random_predict()
test_overall_accuracy()
test_classes_accuracy()
