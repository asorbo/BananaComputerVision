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
MAXIMUM_EPHOCS = 30  #input EPHOCHS + 1
ARCHITECHTURES = ["CustomCNN", "Res50"]
BATCH_SIZES = [5, 10, 20]
LEARNING_RATES = [0.001, 0.0005, 0.0001]

EARLY_STOPPING_TOLLERANCE = 3
EARLY_STOPPING_MINIMUM_DELTA = 0.01

PLOT_WIDTH = 15
PLOT_HEIGHT = 7
PLOT_LINEWIDTH = 0.7

BASE_PATH = "./outputs"  #The directory where to save the trained models, training plots, and training/testing data


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
        label = torch.IntTensor([label for label in self.img_labels.iloc[idx, 1:len(CLASSES) + 1]])
        label = torch.argmax(label) #hot encode the tensor 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

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

#tollerance is the amount of scores that are counted, min delta is the minimum increase required to continue
class EarlyStopping:
    def __init__(self, tolerance=EARLY_STOPPING_TOLLERANCE, min_delta=EARLY_STOPPING_MINIMUM_DELTA):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.early_stop = False
        self.scores = []

    def check(self, score):
        self.scores.append(score)
        if len(self.scores) < (self.tolerance + 1):
            return

        delta = 0
        current_score = self.scores[-(self.tolerance+1)]
        for score in self.scores[-self.tolerance:]:
            delta += score-current_score
            current_score = score
        delta /= self.tolerance
        if delta < self.min_delta:
            print(f"Improved less then {self.min_delta*100}\% in the last {self.tolerance} epochs, early stopping")
            self.early_stop = True 
    

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_dataloader(loader, batch_size):
    # get some random training images
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print the human label
    images_labels = []
    for image_index in range(batch_size):
        current_labels = []
        i = 0
        for label in torch.nn.functional.one_hot(labels[image_index].to(torch.int64)):
            if label == 1:
                current_labels.append(CLASSES[i])
            i+=1
        images_labels.append('-'.join(current_labels))
    print(" | ".join(images_labels))    

# train the net
def train(save_path):
    train_scores_history = []
    valid_scores_history = []
    early_stopping = EarlyStopping()
    best_accuracy = 0
    best_net_epoch = 0

    train_scores_history.append(test_overall(net, trainloader))
    valid_scores_history.append(test_overall(net, validloader))
    for epoch in range(MAXIMUM_EPHOCS):  # loop over the dataset multiple times
        running_loss = 0.0
        runs = 0
        loss = None
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
            runs += 1
            running_loss += loss.item()
        running_loss /= runs
        print(f"Epoch {epoch+1}/{MAXIMUM_EPHOCS}: {running_loss}")
        train_scores_history.append(test_overall(net, trainloader, averaged_minibatch_training_loss=running_loss))
        valid_scores_history.append(test_overall(net, validloader))

        latest_accuracy = valid_scores_history[-1]["accuracy"]
        if latest_accuracy > best_accuracy:
            best_accuracy = latest_accuracy
            save(save_path)
            #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.003)
            best_net_epoch = epoch

        early_stopping.check(latest_accuracy)
        if(early_stopping.early_stop):
            break
    
    print('Finished Training')
    return train_scores_history, valid_scores_history, best_net_epoch

#this saves data before plotting, should refactor
def plot(train_scores_history, valid_scores_history, best_net_epoch, test_score, path):
    dump_data.update({"train_scores_history": train_scores_history, 
                 "valid_scores_history":valid_scores_history,
                 "best_net_epoch": best_net_epoch})
    
    n_epochs = len(valid_scores_history)

    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.suptitle(name)
    plt.subplots_adjust(hspace=1)

    plt.subplot(121).set_box_aspect(1)
    plt.title("Performance")
    train_scores_accuracy_history = [score["accuracy"] for score in train_scores_history]
    valid_scores_accuracy_history = [score["accuracy"] for score in valid_scores_history]
    plt.xlim(0, n_epochs)
    plt.xticks(np.arange(0, n_epochs, 1))
    plt.axhline(y = test_score["accuracy"], linestyle="dashed", color = '#cccccc', linewidth=PLOT_LINEWIDTH)
    plt.plot(train_scores_accuracy_history, linestyle="solid", label="Train accuracy")
    plt.plot(valid_scores_accuracy_history, linestyle="dotted", label="Validation accuracy")
    plt.axvline(x = best_net_epoch, color = 'r', label = 'Early stop')
    plt.yticks(list(plt.yticks()[0]) + [test_score["accuracy"]])
    plt.plot(best_net_epoch, test_score["accuracy"], 'go', label = "Test accuracy")
    plt.plot()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.subplot(122).set_box_aspect(1)
    plt.title("Optimization")
    train_scores_loss_history = [score["loss"] for score in train_scores_history]
    valid_scores_loss_history = [score["loss"] for score in valid_scores_history]
    plt.xlim(0, n_epochs)
    plt.xticks(np.arange(0, n_epochs, 1))
    plt.axhline(y = test_score["loss"], linestyle="dashed", color = '#cccccc', linewidth=PLOT_LINEWIDTH)
    plt.plot(train_scores_loss_history, linestyle="solid", label="Train loss")
    plt.plot(valid_scores_loss_history, linestyle="dotted", label="Validation loss")
    plt.axvline(x = best_net_epoch, color = 'r', label = 'Early stop')
    plt.yticks(list(plt.yticks()[0]) + [test_score["loss"]])
    plt.plot(best_net_epoch, test_score["loss"], 'go', label = "Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (%)")
    plt.legend()
    
    
    plt.savefig(path + ".png")
    print("Plot saved as ", path + ".png")

#test overall accuracy
#param averaged_minibatch_training_loss replaces the training loss if available 
def test_overall(net, dataloader, averaged_minibatch_training_loss = None):
    #do this at each epoch (then plot)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images).to(device)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    return {"accuracy": accuracy, "loss": (averaged_minibatch_training_loss if averaged_minibatch_training_loss else loss.item())}

#test predictions for each class
def test_classes_accuracy():
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images).to(device)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1

    test_classes_accuracy = {}
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        test_classes_accuracy.update({f"{classname}": accuracy})
    return test_classes_accuracy

#Save a model to a given path
def save(path):
    print("Saving to ", path)
    torch.save(net.state_dict(), path + ".pth")

#Load a model from a given path
def load(path):
    print("Loading from ", path)
    net = Net()
    if ("CustomCNN" in path):
        net = Net().to(device)
    else:
        net = torchvision.models.resnet50(weights='DEFAULT')
        net.fc = nn.Linear(net.fc.in_features, 4)
        net.to(device)
    net.load_state_dict(torch.load(path + ".pth")) 
    print("Loaded")
    return net



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('GPU is being used' if torch.cuda.is_available() else '!!! ⚠ CPU is being used ⚠ !!!')

#Defini some preprocessing to 
transform = transforms.Compose(
    [transforms.ToPILImage(),transforms.ToTensor(), #PIL -> Python Imaging Library format
     transforms.Resize(64),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   #to make model training less sensitive to the scale of features

dataset = CustomImageDataset("/home/ago/Documents/Thesis/BananaComputerVision/TheDataset/classesTotal.csv", 
                                            "/home/ago/Documents/Thesis/BananaComputerVision/TheDataset/",
                                            transform=transform)
splits = torch.utils.data.random_split(dataset, [.7, .15, .15], torch.Generator().manual_seed(30))

start = time.time()
print(start)
dump_data = {}
for architecture in ARCHITECHTURES:
    for batch_size in BATCH_SIZES:
        for learning_rate in LEARNING_RATES:
            path = f'{BASE_PATH}/{architecture}_lr={learning_rate}_batchSize={batch_size}'
            name = f'{architecture}, lr={learning_rate}, batch size={batch_size}'
            print(f"\n\n Training: {name}")

            
            trainloader = torch.utils.data.DataLoader(splits[0], batch_size=batch_size,
                                                    shuffle=True, num_workers=2)

            validloader = torch.utils.data.DataLoader(splits[1], batch_size=batch_size,
                                                    shuffle=False, num_workers=2)
            
            testloader = torch.utils.data.DataLoader(splits[2], batch_size=batch_size,
                                                     shuffle=False, num_workers=2)

            net = Net()
            if (architecture == "Res50"):
                net = torchvision.models.resnet50(weights='DEFAULT')
                net.fc = nn.Linear(net.fc.in_features, 4).to(device)
                net.to(device)
            elif(architecture == "CustomCNN"):
                net.to(device)  #to the GPU
            criterion = nn.CrossEntropyLoss()   #Cross entropy loss (to see how different the prediction is from the ground truth)
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.003) #stocastic gradient descent weight decay implements a L2 regularization to reduce spikes in loss
            
            train_scores_history, valid_scores_history, best_net_epoch = train(path)

            load(path) #loads the latest (thus best performing) saved model
            test_score = test_overall(net, testloader)
            print(test_score)
            dump_data.update({"test_score": test_score})
            plot(train_scores_history, valid_scores_history, best_net_epoch, test_score, path)
            dump_data.update( {"test_classes_accuracy": test_classes_accuracy()})
            with open(path + "_plotdata.json", "w") as outfile:
                outfile.write(json.dumps(dump_data, indent=4))
end = time.time()
print(end - start) #total time in seconds
#random_predict()