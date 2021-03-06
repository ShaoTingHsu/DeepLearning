# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:33:32 2020

@author: USER
"""
#Q1
import os
os.getcwd()
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.metrics import confusion_matrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#input data
data = pd.read_csv('covid_19.csv')
data = data.drop(index = [0,1])
data.index = data.iloc[:,0]
data = data.drop(columns = ['Unnamed: 0','Lat', 'Long'])
country_name = data.index
data = data.values
data = data.astype(np.float)
# data = data.astype(float)
#calculate delta of patient for covid-19
#delta_data = data[:,range(1,data.shape[1])] - data[:,range(data.shape[1]-1)]
# delta_data = pd.DataFrame(delta_data)
#correlation matrix
correlation_matrix = np.corrcoef(data)
#correlation
plt.matshow(correlation_matrix)
plt.colorbar()
plt.show()

#pick up the country which have high correlation with US 
#find country in C(based on US)
correlation_for_US = correlation_matrix[np.where(country_name=='US')[0],:]
country_name_with_high_corr_withUS = country_name[np.where(correlation_for_US>0.95)[1]]
OG_data = data[np.where(correlation_for_US>0.95)[1],:]

#train_data, train_label = get_subsequence_and_label(OG_data, 5)

#correlation matrix
correlation_matrix = np.corrcoef(OG_data)
#correlation
plt.matshow(correlation_matrix)
plt.colorbar()
plt.show()

def get_subsequence_and_label(OG_data, L):
    #number of subsequence is len(data)-L+1
    #index of the last day of i-th subsequence is L+i (i is start with 0)
    subsequence = []#np.zeros([(OG_data.shape[1]-L+1)*(OG_data.shape[1]-L+1),L])
    label = []#np.zeros([(OG_data.shape[1]-L+1)*(OG_data.shape[1]-L+1),1])
    for i in range(OG_data.shape[0]):# number of country
        for j in range((OG_data.shape[1]-L)):
            subsequence.append(OG_data[i,range(j,L+j)])#subsequence[i*(OG_data.shape[1]-L+1)+j,:] = OG_data[i,range(j,L+j)]
            label.append(np.where((OG_data[i,(L+j)]-OG_data[i,(L+j-1)])>0,1,0))#label[i*(OG_data.shape[1]-L+1)+j,:] = np.where(OG_data[i,(L+j)]>0,1,0)
        #subsequence[i*(OG_data.shape[1]-L+1)+OG_data.shape[1]-L,:] = OG_data[i,range(OG_data.shape[1]-L,L+OG_data.shape[1]-L)]
        #label[i*(OG_data.shape[1]-L)+OG_data.shape[1]-L,:] = 0
    subsequence = np.stack(subsequence)
    subsequence = torch.from_numpy(subsequence.astype(np.float32)[:, :,np.newaxis])
    label = np.stack(label)
    label = torch.from_numpy(label.astype(np.int32))
    return [subsequence, label]

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim): #1,100,1,2
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim # Hidden dimensions
        self.layer_dim = layer_dim # Number of hidden layers
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')#(batch_dim, seq_dim, input_dim)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :]) 
        return out
    
class LSTMRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim): #1,100,1,2
        super(LSTMRNN, self).__init__()
        self.hidden_dim = hidden_dim # Hidden dimensions
        self.layer_dim = layer_dim # Number of hidden layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)#(batch_dim, seq_dim, input_dim)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()
        out, (hn,cn) = self.rnn(x, (h0.detach(),c0.detach()))
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
#Shuffle
shuffle_ind = np.arange(OG_data.shape[0])
np.random.shuffle(shuffle_ind)
OG_data = OG_data[shuffle_ind]
#build train data and test data
train_data = OG_data[range(100),:]
test_data = OG_data[100:133,:]

#切資料
L = 10
train_dataset_tim = get_subsequence_and_label(train_data, L)
test_dataset_tim = get_subsequence_and_label(test_data, L)
#定義batch size, epoch
batch_size = 77
n_iters = 2000000
num_epochs = n_iters / ((train_dataset_tim[0].shape[0]) / batch_size)
num_epochs = int(num_epochs)
#創建model
model = LSTMRNN(input_dim = 1, hidden_dim = 20, layer_dim = 1, output_dim = 2).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

print((train_dataset_tim[0].shape[0]))
print(num_epochs)

# Number of steps to unroll
np.random.seed(100)
iter = 0
train_loss = []
train_acc = []
test_acc = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0,len(train_dataset_tim),batch_size):
        model.train()
        inputs = train_dataset_tim[0][i:(i+batch_size),:,:]
        inputs = Variable(inputs)
        inputs = inputs.to(device)
        labels = train_dataset_tim[1][i:(i+batch_size)]
        labels = Variable(labels.long())
        labels = labels.to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()        
        iter += 1
        if(epoch%100==0):
            inputs = test_dataset_tim[0]
            inputs = Variable(inputs)
            inputs = inputs.to(device)
            output_test = model(inputs)
            pred_y_test = torch.max(output_test.cpu(), 1)[1].data.numpy()
            inputs = train_dataset_tim[0]
            inputs = Variable(inputs)
            inputs = inputs.to(device)
            output_train = model(inputs)
            pred_y_train = torch.max(output_train.cpu(), 1)[1].data.numpy()
            train_loss.append(loss.item())
            train_acc.append(np.sum(np.diag(confusion_matrix(pred_y_train, train_dataset_tim[1])))/len(pred_y_train))
            test_acc.append(np.sum(np.diag(confusion_matrix(pred_y_test, test_dataset_tim[1])))/len(pred_y_test))
        if(epoch==1):
            print(confusion_matrix(pred_y_train, train_dataset_tim[1]))
            print(confusion_matrix(pred_y_test, test_dataset_tim[1]))
        if(epoch==num_epochs-1):
            print(confusion_matrix(pred_y_train, train_dataset_tim[1]))
            print(confusion_matrix(pred_y_test, test_dataset_tim[1]))
            
len(train_acc)
print(num_epochs)

#plot
ind = np.arange(0, 214, 1)
loss = train_acc
plt.plot(ind, loss)
plt.title('train_acc')
plt.show()
ind = np.arange(0, 214, 1)
loss = test_acc
plt.plot(ind, loss)
plt.title('test_acc')
plt.show()
ind = np.arange(0, 214, 1)
loss = train_loss
plt.plot(ind, loss)
plt.title('train_loss')
plt.show()

len(train_acc)

#把之前所有高相關國家的最後L天抓出來，用來預測未來患者會不會上升
L = 5
temp_OG_data = data[np.where(correlation_for_US>0.95)[1],:]
temp_subsequence = []
temp_label = []
for i in range(temp_OG_data.shape[0]):# number of country
    temp_subsequence.append(temp_OG_data[i,range(temp_OG_data.shape[1]-L,temp_OG_data.shape[1])])
temp_subsequence = np.stack(temp_subsequence)
temp_subsequence = torch.from_numpy(temp_subsequence.astype(np.float32)[:, :,np.newaxis])
#做一版所有高相關國家都在裡面的model(LSTM)
train_dataset_tim = get_subsequence_and_label(temp_OG_data, L)
#定義batch size, epoch
batch_size = 77
n_iters = 2000000
num_epochs = n_iters / ((train_dataset_tim[0].shape[0]) / batch_size)
num_epochs = int(num_epochs)
#創建model
model = LSTMRNN(input_dim = 1, hidden_dim = 20, layer_dim = 1, output_dim = 2).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Number of steps to unroll
np.random.seed(100)
iter = 0
train_loss = []
train_acc = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0,len(train_dataset_tim),batch_size):
        model.train()   
        inputs = train_dataset_tim[0][i:(i+batch_size),:,:]
        inputs = Variable(inputs)
        inputs = inputs.to(device)
        labels = train_dataset_tim[1][i:(i+batch_size)]
        labels = Variable(labels.long())
        labels = labels.to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs.float()) 
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()        
        iter += 1
        if(epoch%100==0):
            inputs = train_dataset_tim[0]
            inputs = Variable(inputs)
            inputs = inputs.to(device)
            output_train = model(inputs)
            pred_y_train = torch.max(output_train.cpu(), 1)[1].data.numpy()
            train_loss.append(loss.item())
            train_acc.append(np.sum(np.diag(confusion_matrix(pred_y_train, train_dataset_tim[1])))/len(pred_y_train))
#預測
inputs = temp_subsequence
inputs = Variable(inputs)
inputs = inputs.to(device)
output_test = model(inputs)
pred_y_test = torch.max(output_test.cpu(), 1)[1].data.numpy()
#手算softmax
prob = np.exp(output_test.cpu().data.numpy()[:,1])/(np.exp(output_test.cpu().data.numpy()[:,1])+np.exp(output_test.cpu().data.numpy()[:,0]))

len(temp_subsequence)

#畫畫
import pycountry
from more_itertools import locate

# 將國家轉換為2個音文字母的代碼
input_countries = country_name_with_high_corr_withUS.tolist()
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_2
    
country_name = [countries.get(country, 'Unknown code') for country in input_countries] 
ind_unknown = list(locate(codes, lambda x: x == 'Unknown code'))
unknown_name = ['bo','mm','cd','cg','ci','mk','la','md','ru','sy','tz','us','ps']
#原來Burma是緬甸
for i in range(len(ind_unknown)):
    country_name[ind_unknown[i]] = unknown_name[i]
    
pred_1 = country_code[np.where(pred_y_test==1)].tolist()
pred_0 = country_code[np.where(pred_y_test==0)].tolist()
prob_1 = prob[np.where(pred_y_test==1)]
prob_0 = prob[np.where(pred_y_test==0)]

# create dictionary that can show the prob.belongs to label
dic_1 = {pred_1[i]:prob_1[i] for i in range(len(pred_1)) }
dic_0 = {pred_0[i]:prob_0[i] for i in range(len(pred_0)) }
from pygal_maps_world.maps import World
worldmap_chart = World()
worldmap_chart.title = 'Covid_19'
worldmap_chart.add('Ascending', dic_1)
worldmap_chart.add('Descending', dic_0)

#第二題
import os
os.chdir('/home/jeff/mount')
print(os.getcwd())
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg #show image  with '.png'
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

route_img = 'Homework2/anime-faces/data'
img = list()
for filename in os.listdir(route_img):
    if(filename=='.ipynb_checkpoints'): continue
    temp_img = cv2.imread(os.path.join(route_img,filename))
    temp_img = cv2.resize(temp_img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    #這步其實可以不用, 就是RGB在opencv跟plt.imshow所排的順序不同, 想把它們喬成一樣而已。 
    b,g,r = cv2.split(temp_img)  
    temp_img = cv2.merge([r,g,b]) 
    img.append(temp_img)
img = np.array(img)
img = img/255
#把channel 往前移
img = np.moveaxis(img, 3, 1)
img_torch = torch.Tensor(img)

#試畫一張圖
fig = plt.figure(figsize=(10,10)) 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.01, wspace=0.01)
image1 = np.moveaxis(img, 1, 3)[0]
#image1 = np.moveaxis(real_sample1[0],0,2)
ax = fig.add_subplot(1, 11, 1, xticks=[], yticks=[])
ax.imshow(image1)
ax.axis('off') 
print(img[0].shape)

image_size = 3*28*28
num_epochs = 100
batch_size = 256
num_batch=img_torch.shape[0]/batch_size+1
num_batch = int(num_batch)
learning_rate = 0.0005
h_dim = 20
z_dim = 20

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size = image_size, h_dim = h_dim, z_dim = z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, z_dim) # mean
        self.fc4 = nn.Linear(h_dim, z_dim) # variance
        self.fc5 = nn.Linear(z_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, image_size)
        
    # step of encode
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2), self.fc4(h2)
    
    # step of generating code with noise
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    #  step of decode
    def decode(self, z):
        h = torch.relu(self.fc5(z))
        return torch.sigmoid(self.fc6(h))
    
    #  forward process- encode -> decode
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
    
#construct model
model = VAE().to(device)
#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_tensor = img_torch
sample_dir = 'temp_img'
#os.makedirs(sample_dir)

train_loss = []
for epoch in range(num_epochs):
    index=torch.randperm(len(train_tensor))
    train_tensor = train_tensor[index]
    
    for i in range(num_batch):
        x = train_tensor[i:(i+batch_size),:,:]
        x = x.cuda().view(-1, image_size)
        x_reconst, mu, log_var = model(x)
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average = False)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # backpropogation and optimization
        loss = reconst_loss + 0*kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
    # 利用训练的模型进行测试
    with torch.no_grad():
        # 随机生成的图像
        z = torch.randn(batch_size, z_dim).cuda()
        out = model.decode(z).view(-1, 3, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))
        # 重构的图像
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 3, 28, 28), out.view(-1, 3, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

# plot learning curve of ELBO
ind = np.arange(0, num_epochs, 1)
loss = train_loss
plt.plot(ind, loss)
plt.title('train_acc')
plt.title('Learning Curve for VAE')
plt.show()


