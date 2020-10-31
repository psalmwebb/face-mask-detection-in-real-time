


import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import tqdm
import os


# In[5]:


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


# In[3]:


glob_path = "C:\\Users\\user\\Documents\\LEARNAI\\face_mask"

os.chdir(glob_path)
train_test_data = []

with_mask = 0
without_mask = 0

target = {'with_face_mask':1,'without_face_mask':0}

for i in target:
    path = os.path.join(glob_path,i)
    try:
        for j in os.listdir(path):
            img = cv2.imread(os.path.join(path,j),0)
            img = cv2.resize(img,(50,50))
            pix = np.asarray(img)
            train_test_data.append([pix,target[i]])
            if i == 'with_face_mask':
                with_mask+=1
            elif i == 'without_face_mask':
                without_mask+=1
    except Exception as E:
        print(E)


print(with_mask)
print(without_mask)


# In[4]:


os.chdir("C:\\Users\\user\\Documents\\LEARNAI")

print(len(train_test_data))

np.random.shuffle(train_test_data)

np.save('maskVSnoMask.npy',train_test_data)


# In[3]:


data = np.load('maskVSnoMask.npy',allow_pickle=True)

X = torch.Tensor([i[0] for i in data])

X=X.view(X.shape[0],1,50,50)

y = torch.Tensor([i[1] for i in data])


# In[4]:


# num = 357
# plt.imshow(X[num].view(50,50))
# print(y[num])
with_mask = 0
without_mask = 0

for i in y:
    if i == 0:
        without_mask+=1
    elif i == 1:
        with_mask+=1
print(with_mask,without_mask)


# In[5]:


val_size = int(len(X) * 0.0553)

X_train = X[:len(X) - val_size] / 255.0
y_train = y[:len(y) - val_size]

print(X_train.shape)
print(y_train.shape)

X_test = X[len(X) - val_size :] / 255.0
y_test = y[len(y) - val_size :]

print(X_test.shape)
print(y_test.shape)


# In[1]:



net = PWCNN()

optimizer = optim.Adam(net.parameters(),lr=0.001)


criterion = nn.CrossEntropyLoss()

prev_accur = 0

Epoch = 25

BATCH_SIZE = 226

for i in range(Epoch):
    cf = 0
    for j in tqdm.tqdm(range(0,len(X_train),BATCH_SIZE)):
        output = net(X_train[j:j+BATCH_SIZE])
        optimizer.zero_grad()
        loss = criterion(output,y_train.long()[j:j+BATCH_SIZE])
        loss.backward()
        cf+=loss
        optimizer.step()
    cf/=len(X_train)
    print(f"Epoch : {i+1},Loss : {loss}")

    c_accur=check_accur(X_test,y_test,net)
    if i > 9:
        if c_accur > prev_accur:
            checkpoints = {
                'cnn':net.state_dict(),
                'optimizer':optimizer.state_dict()
            }
            torch.save(checkpoints,'maskVSNoMask_checkpoints.pth')
            prev_accur = c_accur
            print("Saved")


