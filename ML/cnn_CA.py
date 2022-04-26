
# Commented out IPython magic to ensure Python compatibility.
# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import SubsetRandomSampler, Subset, DataLoader, random_split, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter
import shutil
import csv
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import seaborn as sn
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report,confusion_matrix, accuracy_score
import math
from PIL import Image
from itertools import combinations_with_replacement  
import pandas as pd
import os
from collections import Counter

def imshow(subset,idx,if_save=False):
    """
    Args:
    dataset is a Subset object
    idx is the index of the data in the subset,NOT in the root dataset
    """
    img = subset.dataset[subset.indices[idx]][0].squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()  
    if if_save: 
        plt.savefig(Config.output_dir+'data_'+str(idx)+'.png')

# Configuration class
class Config():
    dataset_name = 'mydata/'
    dataset_dir = "/dataset/"+dataset_name
    output_dir = "/output/output_"+dataset_name
    image_size = 450
    batch_size = 32
    num_replicate = 1000
    
os.makedirs(Config.output_dir,exist_ok=True)

"""# Dataset"""
# Split tran and test sets by index
def split_indices(ratio):
    """
    Arg:
      ratio: the raito of trainset to the entire dataset
    """
    len_train = len(pattern_dataset)
    indices = list(range(len_train))
    split_1 = int(np.floor(ratio * len_train))
    split_2 = int(np.floor(((1-ratio)/2) * len_train))
    np.random.seed(25)
    np.random.shuffle(indices)
    train_indices = indices[:split_1]
    val_indices = indices[split_1:split_1+split_2+1]
    test_indices = indices[-split_2-1:]
    return train_indices, val_indices, test_indices

class CAdataset(Dataset):
    def __init__(self, root_dir, transform=None):

        targets_df = pd.read_csv(root_dir + 'labels.csv', header=None, squeeze = True) # path to label file
        self.targets = torch.tensor(targets_df.values) 
        self.root_dir = root_dir # dir to all final patterns
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        y = self.targets[idx]
        y = torch.tensor(y, dtype=torch.long)

        txt_name = self.root_dir+'final/final_'+str(idx+1)+'.txt'
        with open(txt_name) as f: x = f.readlines()
        x = [float(i) for i in x]
        x = torch.tensor(x)
        
        if self.transform:#
            sample = self.transform(sample)
            
        #y = F.one_hot(y, num_classes=15)
        
        return (x, y)    


"""# Model"""
class Net(nn.Module):
    def __init__(self, num_class):
        """
        Arg:
            image_size: size of input image
        """
        super().__init__()
        self.num_class = num_class
        self.fc1 = nn.Linear(450,200) 
        self.fc2 = nn.Linear(200,75)
        self.fc3 = nn.Linear(75,30)
        self.fc4 = nn.Linear(30, num_class)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))

        return x

"""# Train"""
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred = net(data)

        loss = criterion(y_pred, labels)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        total += labels.size(0)

        _, predicted = torch.max(y_pred, 1)

        correct += sum(1 for a, b in zip(predicted, labels) if a == b)

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}, accuracy: {:.4f}%'.format(epoch, train_loss / len(train_loader.dataset), 100 * correct / total))

    return train_loss/len(train_loader.dataset), correct / total


def validate():
    net.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)

            y_pred = net(data)

            # sum up batch loss
            loss = criterion(y_pred, labels)
            val_loss += loss.item()
            total += labels.size(0)

            _, predicted = torch.max(y_pred, 1)
            correct += sum(1 for a, b in zip(predicted, labels) if a == b)

    val_loss /= len(val_loader.dataset)
    print('====> Validation set loss: {:.4f}, accuracy: {:.4f} %'.format(val_loss, 100 * correct / total))

    return val_loss/len(val_loader.dataset), correct / total


def test():
    net.eval()
    test_loss= 0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            y_pred = net(data)

            # sum up batch loss
            loss = criterion(y_pred, labels)
            test_loss += loss.item()
            total += labels.size(0)

            _, predicted = torch.max(y_pred, 1)
            correct += sum(1 for a, b in zip(predicted, labels) if a == b)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}, accuracy: {:.4f} %'.format(test_loss, 100 * correct / total))

    return test_loss, correct / total


print('*****************MAKE DATASETS*****************')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                               transforms.Resize((Config.image_size,Config.image_size)),
                                ])
pattern_dataset = CAdataset(Config.dataset_dir)
print(pattern_dataset.targets)
print(Config.num_replicate)
print(print(len(pattern_dataset.targets)))
print('length:', pattern_dataset.__len__())

# Split train and test sets
ratio_list = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
train_acc_list = []
val_acc_list   = []
test_acc_list  = []

for i in ratio_list:
    # get train set
    train_indices, val_indices, test_indices = split_indices(ratio = 0.8)
    if i != 0.8: train_indices, val_indices_0, test_indices_0 = split_indices(ratio = i)
        
    train_dataset = Subset(pattern_dataset, indices=train_indices)
    val_dataset = Subset(pattern_dataset, indices=val_indices)
    test_dataset = Subset(pattern_dataset, indices=test_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=Config.batch_size, shuffle=False)
    class_list = torch.unique(pattern_dataset.targets).tolist()
    num_class = len(class_list)
    
    setattr(Config, 'num_class', num_class)
    print('Number of class:',num_class)
    print('Training dataset size:',len(train_dataset))
    print('Validation dataset size:',len(val_dataset))
    print('Test dataset size:',len(test_dataset))
    with open(Config.output_dir+'training_details_'+ str(i) +'.txt', 'w') as f:
        f.write("Number of class: %s\n" % num_class)
        f.write("Training dataset size: %s\n" % len(train_dataset))
        f.write("Validation dataset size: %s\n" % len(val_dataset))
        f.write("Test dataset size: %s\n" % len(test_dataset))
            
    # Make model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(Config.num_class)
    net.to(device)

    
    print('*****************SETUP TRAINING*****************')
    # Training setup
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(),lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    epoch = 200
    
    # Train 
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []
    
    train_accuracy_history = []
    val_accuracy_history = []
    test_accuracy_history = []
    
    print('*****************TRAIN**********************************')

    for epoch in range(epoch):
        train_loss, train_accuracy = train(epoch)
        val_loss, val_accuracy = validate()
        test_loss, test_accuracy = test()
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        test_loss_history.append(test_loss)
        
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)
        test_accuracy_history.append(test_accuracy)

        scheduler.step()

    # plot training history
    fig,axs = plt.subplots(2,3,figsize=(10,8))
    axs[0,0].plot(train_loss_history)
    axs[0,0].set_title('Train loss',fontsize =20)
    axs[0,1].plot(val_loss_history)
    axs[0,1].set_title('Validation loss',fontsize =20)
    axs[0,2].plot(test_loss_history)
    axs[0,2].set_title('Test loss',fontsize =20)
    axs[1,0].plot(train_accuracy_history)
    axs[1,0].set_title('Train accuracy',fontsize =20)
    axs[1,1].plot(val_accuracy_history)
    axs[1,1].set_title('Validation accuracy',fontsize =20)
    axs[1,2].plot(test_accuracy_history)
    axs[1,2].set_title('Test accuracy',fontsize =20)
    plt.show
    plt.savefig(Config.output_dir+'train_history_'+ str(i) +'.png')

    with open(Config.output_dir+'train_loss_'+ str(i) +'.txt', 'w') as f:
        for item in train_loss_history: f.write("%s\n" % item)
    with open(Config.output_dir+'val_loss_'+ str(i) +'.txt', 'w') as f:
        for item in val_loss_history: f.write("%s\n" % item)
    with open(Config.output_dir+'test_loss_'+ str(i) +'.txt', 'w') as f:
        for item in test_loss_history: f.write("%s\n" % item)
   
    with open(Config.output_dir+'train_accuracy_'+ str(i) +'.txt', 'w') as f:
        for item in train_accuracy_history: f.write("%s\n" % item)
    with open(Config.output_dir+'val_accuracy_'+ str(i) +'.txt', 'w') as f:
        for item in val_accuracy_history: f.write("%s\n" % item)
    with open(Config.output_dir+'test_accuracy_'+ str(i) +'.txt', 'w') as f:
        for item in test_accuracy_history: f.write("%s\n" % item)
    
    train_acc_list.append(train_accuracy_history[-1])
    val_acc_list.append(val_accuracy_history[-1])
    test_acc_list.append(test_accuracy_history[-1])
    
    """# Save model """
    path_net = Config.output_dir + 'net_' + str(i) + '.pt'
    torch.save(net, path_net)

    # load the entire models
    #net = torch.load(path_net)
    net.eval() # Set to evaluation mode for inference

    """# Analysis"""
    pred_test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
    for i, (data,labels) in enumerate(pred_test_loader, 0): 
        data = data.to(device)
        labels = labels.to(device)
        y_pred = net(data)
        _, predicted = torch.max(y_pred, 1)

    predicted = predicted.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    accuracy = accuracy_score(predicted, labels)
    precision = precision_score(predicted, labels, average='macro')
    recall = recall_score(predicted, labels, average="macro")
    with open(Config.output_dir+'results_' + str(i) + '.txt','w') as f:
        f.write('accuracy:{}\nprecision:{}\nrecall:{}'.format(accuracy,precision,recall))

    cm = confusion_matrix(labels, predicted)
    cm = cm / cm.astype(np.float).sum(axis=1)*100
    cm = np.round(cm,1)
    plt.figure(figsize=(20, 20))    
    df_cm = pd.DataFrame(cm,index = class_list,columns = class_list)
    sn.heatmap(cm, annot=True, cmap = 'Blues',xticklabels=class_list,yticklabels=class_list)
    plt.savefig(Config.output_dir+'cm' + str(i) + '.png')

# write results summary    
with open(Config.output_dir+'summary_train_accuracy_'+ str(i) +'.txt', 'w') as f:
    for item in train_acc_list: f.write("%s\n" % item)
with open(Config.output_dir+'summary_val_accuracy_'+ str(i) +'.txt', 'w') as f:
    for item in val_acc_list: f.write("%s\n" % item)
with open(Config.output_dir+'summary_test_accuracy_'+ str(i) +'.txt', 'w') as f:
    for item in test_acc_list: f.write("%s\n" % item)
plt.figure(figsize=(10,10))
xlabel_list = [int(Config.num_replicate * i) for i in ratio_list]
plt.plot(xlabel_list, train_acc_list, label='train accuracy')
plt.plot(xlabel_list, val_acc_list, label='validation accuracy')
plt.plot(xlabel_list, test_acc_list, label='test accuracy')
plt.xlabel('ratio')
plt.ylabel('accuracy')
plt.show()
plt.savefig(Config.output_dir + 'accuracy_trend_' + str(i) + '.png')
