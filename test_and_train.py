
import os
import h5py
import numpy as np
import hdf5storage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim

class ModelNetDataset(data.Dataset):
    def __init__(self, npy_file,test_dataset=False,transform=None):

        self.np_array_data = np.load(npy_file)
        self.test_bool = test_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.np_array_data['targets'])

    def __getitem__(self, idx):
        x_n = self.np_array_data['features'][idx,:,:,:,:]
        class_label = self.np_array_data['targets'][idx]
        sample = {'feature': x_n, 'class_label': class_label}
        if self.transform:
            sample['feature'] = self.transform(sample['feature'])
            sample['class_label'] = self.transform(sample['class_label'])
        return sample
    
def train(model, device, train_loader, optimizer, epoch,losslist,loss):
    model.train()
    b_idx = 0
    for x in train_loader:
        b_idx+=1
        
        
        x_feat, label = x['feature'].to(device), x['class_label'].type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        prediction = model(x_feat)

        loss_eval = loss(prediction,label)            
        loss_eval.backward()
        optimizer.step()
        
        if b_idx%6 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, b_idx * x_feat.shape[0], len(train_loader.dataset),
                100. * b_idx / len(train_loader), loss_eval.item()))
    losslist.append(loss_eval.item())

def test(model,device, test_loader,epoch,losslist):
            
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x in test_loader:
            x_feat, TrueLabel = x['feature'].to(device), x['class_label'].to(device)
            prediction = model(x_feat)
            _, predicted = torch.max(prediction, 1)
            total += TrueLabel.size(0)
            correct += (predicted == TrueLabel).sum().item()

    print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))
    losslist.append(correct/total)

reg_decay = 0.001
BatchSize = 10
num_epochs = 1
lr = 0.001
printout = 20
device = torch.device('cuda:1')
kwargs = {'num_workers': 1, 'pin_memory': True}
snapshot = 10
snapshot_dir = 'snapshots'

try:
    os.mkdir(snapshot_dir)
except:
    pass


# Instantiate a dataset loader

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = ModelNetDataset('../Generative-and-Discriminative-Voxel-Modeling/datasets/modelnet40_rot_train.npz',False)
test_dataset = ModelNetDataset('../Generative-and-Discriminative-Voxel-Modeling/datasets/modelnet40_rot_test.npz',True)

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=BatchSize, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=BatchSize, shuffle=True, **kwargs)

# Instantiate the network
model_net = VoxceptionNet(n_classes=40).to(device)
print('model loaded')
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_net.parameters(), lr=lr, weight_decay=reg_decay)
#annealing startegy is different in paper
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5)

# Whether to save a snapshot
save = False

global_loss = []
partial_loss = []
print("training")
for epoch in range(1, num_epochs + 1):
    scheduler.step()
    train(model_net, device, train_loader, optimizer, epoch,partial_loss,loss)
    test(model_net, device, test_loader, epoch,global_loss)
