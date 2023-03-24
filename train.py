import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import random
import torchvision.transforms as transforms
import albumentations
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

epochs = 40
batches = 30
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

actions = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'me', 'sorry', 'thankyou', 'you'])

def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True 
s=42
seed_everything(s)

xtrain = 'data_all/xtrain.npy'
ytrain = 'data_all/ytrain.npy'
xval = 'data_all/xval.npy'
yval = 'data_all/yval.npy'
xtest = 'data_all/xtest.npy'
ytest = 'data_all/ytest.npy'

# class ISLR(nn.Module):
#     def __init__(self):
#         super(ISLR, self).__init__()
#         self.lstm1 = nn.LSTM(258, 64, num_layers=1, batch_first=True, bidirectional=False)
#         self.lstm2 = nn.LSTM(64, 128, num_layers=1, batch_first=True, bidirectional=False)
#         self.lstm3 = nn.LSTM(128, 64, num_layers=1, batch_first=True, bidirectional=False)
#         self.fc1 = nn.Linear(64, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.out = nn.Linear(32, actions.shape[0])
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         out, _ = self.lstm1(x)
#         out, _ = self.lstm2(out)
#         out, _ = self.lstm3(out)
#         out = self.fc1(out[:, -1, :])
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.out(out)
#         out = self.softmax(out)
        
#         return out

# class ISLR(nn.Module):
#     def __init__(self, num_layers, output_size):
#         super(ISLR, self).__init__()
#         self.lstm1 = nn.LSTM(input_size=(batches, 30, 258), hidden_size=(batches, 30,64), num_layers=num_layers, batch_first=True, dropout=0.2)
#         self.lstm2 = nn.LSTM(input_size=(30,64), hidden_size=(30,128), num_layers=num_layers, batch_first=True, dropout=0.2)
#         self.lstm3 = nn.LSTM(input_size=(30,128), hidden_size=(30,64), num_layers=num_layers, batch_first=True, dropout=0.2)
#         self.fc1 = nn.Linear((30,64), 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, output_size)
        
#     def forward(self, x):
#         x = self.lstm1(x)
#         x = self.lstm2(x)
#         x = self.lstm3(x)
#         x = x[:, -1, :]
#         x = nn.ReLU(self.fc1(x))
#         x = nn.ReLU(self.fc2(x))
#         x = nn.Softmax(self.fc3(x))
#         return x

class ISLR(nn.Module):
    def __init__(self, num_layers, output_size):
        super(ISLR, self).__init__()
        self.lstm1 = nn.LSTM(input_size=258, hidden_size=64, num_layers=num_layers)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=num_layers)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=num_layers)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x.reshape(-1, 64)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
# class ASLData(Dataset):
#     def __init__(self, path, labels, transform=None):
#         self.X = path
#         self.y = labels
#         self.transform = transform
#         self.aug = albumentations.Compose([
#             albumentations.Resize(64, 64, always_apply=True),
#         ])
#     def __len__(self):
#         return len(self.X)
#     def __getitem__(self, idx):
#         image = cv2.imread(self.X[idx])
#         image = self.aug(image=np.array(image))['image']
#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)
#         label = self.y[idx]
#         return torch.tensor(image.astype(int), dtype=torch.float), torch.tensor(label.astype(int), dtype=torch.long)

class ISLRDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = np.load(data_file)
        self.labels = np.load(label_file)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        return torch.tensor(data).float(), torch.tensor(label).long()

    
train_data = ISLRDataset(xtrain, ytrain)
val_data = ISLRDataset(xval, yval)
test_data = ISLRDataset(xtest, ytest)

 
# dataloaders
trainloader = DataLoader(train_data, batch_size=batches, shuffle=True, drop_last=True)
valloader = DataLoader(val_data, batch_size=batches, shuffle=False, drop_last=True)
testloader = DataLoader(test_data, batch_size=batches, shuffle=False)

model = ISLR(num_layers=2, output_size=actions.shape[0]).to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def train(model, dataloader, optimizer, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.train()

    print('Training...')
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input = data[0].to(device)
        target= data[1].to(device)

        input = input.view(-1, 30, 258)

        optimizer.zero_grad()
        predictions = model(input)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(predictions, 1)
        epoch_loss += loss.item()
        epoch_acc += (preds == target).sum().item()

    train_loss = epoch_loss / len(dataloader.dataset)
    train_acc = epoch_acc / len(dataloader.dataset)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}")
    return train_loss, train_acc


def validate(model, dataloader, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.eval()

    print('Validating...')
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            input = data[0].to(device)
            target= data[1].to(device)

            input = input.view(-1, 30, 258)

            predictions = model(input)
            loss = criterion(predictions, target)

            _, preds = torch.max(predictions, 1)
            epoch_loss += loss.item()
            epoch_acc += (preds == target).sum().item()

        val_loss = epoch_loss / len(dataloader.dataset)
        val_acc = epoch_acc / len(dataloader.dataset)

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}")
        return val_loss, val_acc


train_loss = []
train_acc = []
val_loss = []
val_acc = []

start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}")

    train_epoch_loss, train_epoch_acc = train(model, trainloader, optimizer, criterion, device)
    train_loss.append(train_epoch_loss)
    train_acc.append(train_epoch_acc)

    val_epoch_loss, val_epoch_acc = validate(model, valloader, criterion, device)
    val_loss.append(val_epoch_loss)
    val_acc.append(val_epoch_acc)

end = time.time()

torch.save(model.state_dict(), 'model.pth')

plt.figure(figsize=(10, 8))
plt.plot(train_loss, label='Training Loss', color='red')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(train_acc, label='Training Accuracy', color='green')
plt.plot(val_acc, label='Validation Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
