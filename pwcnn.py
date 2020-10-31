import torch
import torch.nn as nn
import torch.nn.functional as F


class PWCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = (3,3))
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = (3,3))
        self.conv3 = nn.Conv2d(in_channels = 64,out_channels = 128, kernel_size = (3,3))
        self.pool = nn.MaxPool2d(stride = (2,2),kernel_size =(2,2))
        self.fc1 = nn.Linear(128*4*4,64)
        self.fc2 = nn.Linear(64,2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x,dim=1)

def check_accur(X,y,model):
    with torch.no_grad():
        output = model(X.reshape(211,1,50,50))
        _,pred = output.max(1)

        accur = torch.round(torch.sum(torch.Tensor(pred.float()) == y) / float(len(y)) * 100)

        print(f"Epoch:{i+1},Accuracy:{accur}")
        return accur
