# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:04:04 2020

@author: pokemon
"""

# =============================================================================
# 第一大題
# =============================================================================
#packages----
import numpy as np
import pandas as pd
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#input data----
data_train = np.load('D:/USER/Documents/Cloud/DeepLearning/Dataset_HW1/problem1-DNN/train.npz')
data_test = np.load('D:/USER/Documents/Cloud/DeepLearning/Dataset_HW1/problem1-DNN/test.npz')
#data preprocess
x_train = data_train['image']
x_train_answer = data_train['label']
x_train_answer = pd.get_dummies(x_train_answer).to_numpy()
x_test = data_test['image']
x_test_answer = data_test['label']
x_test_answer = pd.get_dummies(x_test_answer).to_numpy()
#把矩陣攤成向量並標準化
x_train = x_train.reshape((-1,x_train.shape[1]*x_train.shape[2]))/225
x_test = x_test.reshape((-1,x_test.shape[1]*x_test.shape[2]))/225
#set.seed----
np.random.seed(100)
#functions----
#forward----
def softmax(x):
    z_u = np.zeros([10,x.shape[0]])
    for i in range(x.shape[0]):
        z_u[:,i] = np.exp(x[i,:])/np.sum(np.exp(x[i,:]))
    return(z_u)
def ReLU(x):
    x = np.where(x>0,x,0)
    return(x)
def y_hat(x_n,Theta):
    z_1 = x_n.dot(Theta['w_1'])+Theta['b_1']
    a_1 = ReLU(z_1)
    z_2 = a_1.dot(Theta['w_2'])+Theta['b_2']
    z_u = softmax(z_2)
    outcome = [z_u,z_1,a_1,z_2]
    outcome = dict(zip(['z_u','z_1','a_1','z_2'], outcome))
    return(outcome)
#backward----
def partial_loss_z2(z_u,y):
    return((z_u-y).T)
def partial_z2_a1(Theta):
    return(Theta['w_2'].transpose())
def partial_a1_z1(a_1):
    temp = np.where(a_1>0,1,0)
    return(temp)
def partial_loss_w2(z_u,y,a_1):
    temp1 = partial_loss_z2(z_u,y)
    temp2 = a_1.reshape((len(a_1),1)).dot(temp1.reshape((1,len(temp1))))
    return(temp2)
def partial_loss_b2(z_u,y):
    temp1 = partial_loss_z2(z_u,y)
    return(temp1)    
def partial_loss_w1(z_u,y,Theta,x_n,a_1):
    temp1 = partial_loss_z2(z_u,y)
    temp2 = temp1.dot(partial_z2_a1(Theta))
    temp3 = temp2*partial_a1_z1(a_1)
    temp4 = x_n.reshape((len(x_n),1)).dot(temp3.reshape((1,len(temp3))))
    return(temp4)
def partial_loss_b1(z_u,y,Theta,a_1):
    temp1 = partial_loss_z2(z_u,y)
    temp2 = temp1.dot(partial_z2_a1(Theta))
    temp3 = temp2*partial_a1_z1(a_1)
    return(temp3)
def backward(forward_outcome, Theta,data,answer):
    delta_w2 = 0
    delta_b2 = 0
    delta_w1 = 0
    delta_b1 = 0
    for i in range(data.shape[0]):
        delta_w2 = delta_w2 + partial_loss_w2(forward_outcome['z_u'][:,i], answer[i,:], forward_outcome['a_1'][i,:])
        delta_b2 = delta_b2 + partial_loss_b2(forward_outcome['z_u'][:,i], answer[i,:])
        delta_w1 = delta_w1 + partial_loss_w1(forward_outcome['z_u'][:,i], answer[i,:], Theta, data[i,:],forward_outcome['a_1'][i,:])
        delta_b1 = delta_b1 + partial_loss_b1(forward_outcome['z_u'][:,i], answer[i,:], Theta,forward_outcome['a_1'][i,:]) 
    return delta_w2, delta_b2, delta_w1, delta_b1
#acc or loss function
def cross_entropy(y,z_u):
    temp = 0
    for i in range(y.shape[0]):
        temp = temp + sum(y[i,:]*np.log(z_u[:,i],))
    temp = temp*(-1)
    return(temp)
def acc(y,z_u):
    z_u = np.argmax(z_u, axis=0)
    y = np.argmax(y, axis=1)
    temp1 = confusion_matrix(y, z_u)
    temp2 = np.sum(np.diag(temp1))
    return(temp2/y.shape[0])
def train_model(x_train,x_train_answer,x_test,x_test_answer,num_epoch,num_node,learning_rate,Theta):
    #等等用來儲存acc, loss
    train_acc_rate = np.zeros((1,num_epoch))
    test_acc_rate = np.zeros((1,num_epoch))
    train_loss = np.zeros((1,num_epoch))
    test_loss = np.zeros((1,num_epoch))
    for epoch in range(num_epoch):
        tStart_epoch = time.time()
        #Shuffle
        shuffle_ind = np.arange(12000)
        np.random.shuffle(shuffle_ind)
        x_train = x_train[shuffle_ind,:]
        x_train_answer = x_train_answer[shuffle_ind,:]
        #一個epoch的樣子
        for batch_ind in range(100):
            #tStart = time.time()
            #minibatch
            x_train_batch = x_train[range((batch_ind*120),((batch_ind+1)*120),1),:]
            x_train_answer_batch = x_train_answer[range((batch_ind*120),((batch_ind+1)*120),1),:]
            #計算z_1, z_u
            forward_outcome = y_hat(x_train_batch, Theta)
            delta_w2, delta_b2, delta_w1, delta_b1 = backward(forward_outcome, Theta, x_train_batch, x_train_answer_batch)
            Theta['w_2'] = Theta['w_2'] - Theta['learning_rate']*delta_w2/120
            Theta['b_2'] = Theta['b_2'] - Theta['learning_rate']*delta_b2/120
            Theta['w_1'] = Theta['w_1'] - Theta['learning_rate']*delta_w1/120
            Theta['b_1'] = Theta['b_1'] - Theta['learning_rate']*delta_b1/120
            forward_outcome = y_hat(x_train_batch, Theta)
            #print(cross_entropy(x_train_answer_batch, forward_outcome['z_u']))
            #print(acc(x_train_answer_batch, forward_outcome['z_u']))
            #tEnd = time.time()
            #print ('batch',batch_ind,':一個batch的時間是 ',(tEnd - tStart))
            print ('epoch',epoch,', batch: ',batch_ind)
        tEnd_epoch = time.time()
        print('epoch',epoch,':一個epoch的時間是 ',(tEnd_epoch - tStart_epoch))
        forward_outcome_train_total = y_hat(x_train, Theta)
        forward_outcome_test_total = y_hat(x_test, Theta)
        train_acc_rate[:,epoch] = acc(x_train_answer, forward_outcome_train_total['z_u'])
        test_acc_rate[:,epoch] = acc(x_test_answer, forward_outcome_test_total['z_u'])
        train_loss[:,epoch] = cross_entropy(x_train_answer, forward_outcome_train_total['z_u'])/x_train_answer.shape[0]
        test_loss[:,epoch]  = cross_entropy(x_test_answer, forward_outcome_test_total['z_u'])/x_test_answer.shape[0]
        print('Total cross entropy/#data for train data is: ',train_loss[:,epoch])
        print('Total cross entropy/#data for test data is: ',test_loss[:,epoch])
        print('Total accuracy for train data is: ', train_acc_rate[:,epoch])
        print('Total accuracy test data is: ',test_acc_rate[:,epoch])
        if num_node == 2:
            if epoch == 0 or epoch == 199:
                plot_answer = np.argmax(x_train_answer, axis=1)
                plot_data = pd.DataFrame([forward_outcome_train_total['z_1'][:,0],forward_outcome_train_total['z_1'][:,1],plot_answer]).T
                plot_data.columns = ['x','y','ground_true']
                sns.lmplot(data=plot_data, x='x', y='y', hue='ground_true', 
                           fit_reg=False, legend=True, legend_out=True)
    return forward_outcome_train_total, forward_outcome_test_total, train_acc_rate, test_acc_rate, train_loss, test_loss


#Problem1 and 2
np.random.seed(100)
num_epoch = 500
num_node = 60
#initial parameters----
w_1 = np.zeros((784,num_node))
w_2 = np.zeros((num_node,10))
# w_1 = np.random.normal(0,0.01,784*num_node).reshape((784,num_node))
# w_2 = np.random.normal(0,0.01,num_node*10).reshape((num_node,10))
b_1 = np.random.normal(0,0.01,1*num_node).reshape((1,num_node))
b_2 = np.random.normal(0,0.01,1*10).reshape((1,10))
learning_rate = 0.05
Theta = [w_1,w_2,b_1,b_2,learning_rate]
Theta = dict(zip(['w_1','w_2','b_1','b_2','learning_rate'], Theta))
outcome_problem1 = train_model(x_train,x_train_answer,x_test,x_test_answer,num_epoch,num_node,learning_rate,Theta)
#plot
ind = np.arange(0, 500, 1)
loss = outcome_problem1[4][0]
plt.plot(ind, loss)
plt.title('train_loss')
plt.show()
ind = np.arange(0, 500, 1)
loss = 1-outcome_problem1[2][0]
plt.plot(ind, loss)
plt.title('train_error_rate')
plt.show()
ind = np.arange(0, 500, 1)
loss = 1-outcome_problem1[3][0]
plt.plot(ind, loss)
plt.title('test_error_rate')
plt.show()
#confusion matrix
z_u = np.argmax(outcome_problem1[1]['z_u'], axis=0)
y = np.argmax(x_test_answer, axis=1)
cm = confusion_matrix(y, z_u)

#problem3
np.random.seed(100)
num_epoch = 200
num_node = 2
#initial parameters----
# w_1 = np.zeros((784,num_node))
# w_2 = np.zeros((num_node,10))
w_1 = np.random.normal(0,0.01,784*num_node).reshape((784,num_node))
w_2 = np.random.normal(0,0.01,num_node*10).reshape((num_node,10))
b_1 = np.random.normal(0,0.01,1*num_node).reshape((1,num_node))
b_2 = np.random.normal(0,0.01,1*10).reshape((1,10))
learning_rate = 0.05
Theta = [w_1,w_2,b_1,b_2,learning_rate]
Theta = dict(zip(['w_1','w_2','b_1','b_2','learning_rate'], Theta))
outcome_problem3 = train_model(x_train,x_train_answer,x_test,x_test_answer,num_epoch,num_node,learning_rate,Theta)
#confusion matrix
z_u = np.argmax(outcome_problem3[1]['z_u'], axis=0)
y = np.argmax(x_test_answer, axis=1)
cm = confusion_matrix(y, z_u)

# =============================================================================
# 第二大題
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.optim import Adam
# from torchsummary import summary

route_picture = 'D:/User/Documents/Cloud/DeepLearning/Dataset_HW1/problem2-CNN/images/'
route_location_train = 'D:/User/Documents/Cloud/DeepLearning/Dataset_HW1/problem2-CNN/train.csv'
route_location_test = 'D:/User/Documents/Cloud/DeepLearning/Dataset_HW1/problem2-CNN/test.csv'

# route_picture = 'D:/document/Cloud/DeepLearning/Dataset_HW1/problem2-CNN/images/'
# route_location_train = 'D:/document/Cloud/DeepLearning/Dataset_HW1/problem2-CNN/train.csv'
# route_location_test = 'D:/document/Cloud/DeepLearning/Dataset_HW1/problem2-CNN/test.csv'
# route_location = 'D:/User/Documents/Cloud/DeepLearning/Dataset_HW1/problem2-CNN/train.csv'
def acc(y,z_u):
    z_u = np.argmax(z_u, axis=0)
    y = np.argmax(y, axis=1)
    temp1 = confusion_matrix(y, z_u)
    temp2 = np.sum(np.diag(temp1))
    return(temp2/y.shape[0])
def get_data(route_location, route_picture):
    #先抓取train data: 圖片中口罩位置
    data_location = pd.read_csv(route_location)
    #把所有圖片的名字抓出來, 不要有重複
    file_name = data_location['filename']
    file_name = pd.unique(file_name)
    #抓取所有口罩圖片
    tStart = time.time()
    data_pictures = list()
    for i in file_name:
        data_pictures.append(cv2.imread(route_picture+i))
    tEnd = time.time()
    print('處理圖片的時間為: ',(tEnd - tStart))
    data_pictures = dict(zip(file_name, data_pictures))
    #抓有口罩的你各位出來
    data_pictures_subset = list()
    data_pictures_subset_label = list()
    for i in file_name:
        temp_data_ind = np.array(np.where(np.array(data_location['filename']) == i))
        temp_data_location = [data_location.iloc[k,:] for k in temp_data_ind][0]
        for j in range(temp_data_location.shape[0]):
            temp_xmin = np.array(temp_data_location['xmin'])[j]
            temp_xmax = np.array(temp_data_location['xmax'])[j]
            temp_ymin = np.array(temp_data_location['ymin'])[j]
            temp_ymax = np.array(temp_data_location['ymax'])[j]
            temp_pictures_subset = data_pictures[i][temp_ymin:temp_ymax, temp_xmin:temp_xmax]
            #resize圖片到同樣值
            temp_pictures_subset = cv2.resize(temp_pictures_subset, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            #這步其實可以不用, 就是RGB在opencv跟plt.imshow所排的順序不同, 想把它們喬成一樣而已。 
            # b,g,r = cv2.split(temp_pictures_subset)  
            # temp_pictures_subset = cv2.merge([r,g,b])  
            #把切出來且resize過的的圖片存起來
            data_pictures_subset.append(temp_pictures_subset)
            data_pictures_subset_label.append(np.array(temp_data_location['label'])[j])
    #開始轉成tensor
    data_pictures_subset = np.stack(data_pictures_subset)
    #把channel 往前移
    data_pictures_subset = np.moveaxis(data_pictures_subset, 3, 1)
    # 將numpy中的array轉成tensor
    data_pictures_subset = torch.from_numpy(data_pictures_subset)
    #這是問同學的label轉tensor方法
    data_pictures_subset_label = np.array(data_pictures_subset_label)
    label_name=['bad','none','good']
    for i in range(len(label_name)):
        data_pictures_subset_label[data_pictures_subset_label==label_name[i]]=i
    data_pictures_subset_label = data_pictures_subset_label.astype(np.int32)
    data_pictures_subset_label = torch.from_numpy(data_pictures_subset_label)
    data = [data_pictures_subset,data_pictures_subset_label]
    data = dict(zip(['picture','label'], data))
    return(data)
def plot(train_model_output):
    #train_loss
    ind = np.arange(0,train_model_output[2].shape[1], 1)
    loss = train_model_output[2][0]
    plt.plot(ind, loss)
    plt.title('train_loss')
    plt.show()
    
    #train_error_rate
    ind = np.arange(0,train_model_output[0].shape[1], 1)
    loss = train_model_output[0][0]
    plt.plot(ind, loss)
    plt.title('train_acc_rate')
    plt.show()
    
    #test_error_rate
    ind = np.arange(0,train_model_output[1].shape[1], 1)
    loss = train_model_output[1][0]
    plt.plot(ind, loss)
    plt.title('test_acc_rate')
    plt.show()
    print(train_model_output[3])
def train_model(train_data,test_data,LR,cnn,loss_func,num_epoch,batch_size,seed,lr_dym = False):
    #train_data: data used in train data
    #test_data: data used in evaluate the performace of model
    #LR: learning rate
    #cnn: model
    #loss_func: loss function between model and data
    #num_epoch: number of epoch we want to do
    #batch_size: batch size of each iterate
    #seed: set seed for random
    np.random.seed(seed)
    optimizer = Adam(cnn.parameters(), lr=LR)
    train_acc_rate = np.zeros((1,num_epoch))
    test_acc_rate = np.zeros((1,num_epoch))
    train_loss = np.zeros((1,num_epoch))
    for epoch in range(num_epoch):
        tStart_epoch = time.time()
        #Shuffle
        shuffle_ind = np.arange(len(train_data['picture']))
        np.random.shuffle(shuffle_ind)
        train_data['picture'] = train_data['picture'][shuffle_ind]
        train_data['label'] = train_data['label'][shuffle_ind]
        if(lr_dym):
            optimizer.param_groups[0]['lr'] = LR * (0.1 ** (epoch // 30))
        #一個epoch的樣子
        for batch_ind in range(int(len(train_data['picture'])/batch_size)):
            #minibatch
            x_train_batch = train_data['picture'][range((batch_ind*batch_size),((batch_ind+1)*batch_size),1)]
            x_train_answer_batch = train_data['label'][range((batch_ind*batch_size),((batch_ind+1)*batch_size),1)]
            output = cnn(x_train_batch)[0]
            loss = loss_func(output, x_train_answer_batch.long())
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            print('現在是epoch: ',epoch, 'batch: ',batch_ind)
        output_train = cnn(train_data['picture'])[0]
        pred_y_train = torch.max(output_train, 1)[1].data.numpy()
        accuracy_train = float((pred_y_train == train_data['label'].data.numpy()).astype(int).sum()) / float(train_data['label'].size(0))
        output_test = cnn(test_data['picture'])[0]
        pred_y_test = torch.max(output_test, 1)[1].data.numpy()
        accuracy_test = float((pred_y_test == test_data['label'].data.numpy()).astype(int).sum()) / float(test_data['label'].size(0))
        train_acc_rate[:,epoch] = accuracy_train
        test_acc_rate[:,epoch] = accuracy_test
        train_loss[:,epoch] = loss.item()
        print('Total accuracy for train data is: ', accuracy_train)
        print('Total accuracy test data is: ',accuracy_test)
        tEnd_epoch = time.time()
        print('epoch',epoch,':一個epoch的時間是 ',(tEnd_epoch - tStart_epoch))
    cm = confusion_matrix(test_data['label'].data.numpy(), pred_y_test)
    return train_acc_rate, test_acc_rate, train_loss, cm
def train_model_resample(train_data,test_data,LR,cnn,loss_func,num_epoch,batch_size,seed,num_resample,lr_dym = False):
    np.random.seed(seed)
    ind_bad = np.where(np.array(train_data['label']) == 0)[0]
    ind_none = np.where(np.array(train_data['label']) == 1)[0]
    ind_good = np.where(np.array(train_data['label']) == 2)[0]
    resample_size = np.min([len(ind_bad),len(ind_none),len(ind_good)])
    total_epoch = 0
    for resample in range(num_resample): 
        np.random.shuffle(ind_bad)
        np.random.shuffle(ind_none)
        np.random.shuffle(ind_good)
        train_data_resample = list()
        train_data_resample.append(torch.cat([train_data['picture'][ind_bad[range(resample_size)]],train_data['picture'][ind_none[range(resample_size)]],train_data['picture'][ind_good[range(resample_size)]]]))
        train_data_resample.append(torch.cat([train_data['label'][ind_bad[range(resample_size)]],train_data['label'][ind_none[range(resample_size)]],train_data['label'][ind_good[range(resample_size)]]]))
        train_data_resample = dict(zip(['picture','label'], train_data_resample))
        optimizer = Adam(cnn.parameters(), lr=LR)
        train_acc_rate = np.zeros((1,num_epoch*num_resample))
        test_acc_rate = np.zeros((1,num_epoch*num_resample))
        train_loss = np.zeros((1,num_epoch*num_resample))
        for epoch in range(num_epoch):
            tStart_epoch = time.time()
            #Shuffle
            shuffle_ind = np.arange(len(train_data_resample['picture']))
            np.random.shuffle(shuffle_ind)
            train_data_resample['picture'] = train_data_resample['picture'][shuffle_ind]
            train_data_resample['label'] = train_data_resample['label'][shuffle_ind]
            if(lr_dym):
                optimizer.param_groups[0]['lr'] = LR * (0.1 ** (epoch // 30))
            #一個epoch的樣子
            for batch_ind in range(int(len(train_data_resample['picture'])/batch_size)):
                #minibatch
                x_train_batch = train_data_resample['picture'][range((batch_ind*batch_size),((batch_ind+1)*batch_size),1)]
                x_train_answer_batch = train_data_resample['label'][range((batch_ind*batch_size),((batch_ind+1)*batch_size),1)]
                output = cnn(x_train_batch)[0]
                loss = loss_func(output, x_train_answer_batch.long())
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients
                print('現在是resample: ',resample,'epoch',epoch, 'batch: ',batch_ind)
            output_train = cnn(train_data_resample['picture'])[0]
            pred_y_train = torch.max(output_train, 1)[1].data.numpy()
            accuracy_train = float((pred_y_train == train_data_resample['label'].data.numpy()).astype(int).sum()) / float(train_data_resample['label'].size(0))
            output_test = cnn(test_data['picture'])[0]
            pred_y_test = torch.max(output_test, 1)[1].data.numpy()
            accuracy_test = float((pred_y_test == test_data['label'].data.numpy()).astype(int).sum()) / float(test_data['label'].size(0))
            train_acc_rate[:,total_epoch] = accuracy_train
            test_acc_rate[:,total_epoch] = accuracy_test
            train_loss[:,total_epoch] = loss.item()
            print('Total accuracy for train data is: ', accuracy_train)
            print('Total accuracy test data is: ',accuracy_test)
            tEnd_epoch = time.time()
            print('total_epoch',total_epoch,':一個epoch的時間是 ',(tEnd_epoch - tStart_epoch))
            total_epoch += 1
    cm = confusion_matrix(test_data['label'].data.numpy(), pred_y_test)
    return train_acc_rate, test_acc_rate, train_loss, cm
            

#problem2
train_data = get_data(route_location_train,route_picture)
test_data = get_data(route_location_test,route_picture)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=4),    
        )
        self.out = nn.Linear(16*7*7, 3)   
    def forward(self, x):
        x = self.conv1(x.float())
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
LR = 0.0005              # learning rate
cnn = CNN()
loss_func = nn.CrossEntropyLoss()
num_epoch = 100
batch_size = 49 #沒有為什麼, 就是剛好整除看起來很舒服 xD
seed = 777 
problem_out_problem2 = train_model(train_data,test_data,LR,cnn,loss_func,num_epoch,batch_size,seed,lr_dym = True)

plot(problem_out_problem2)

#problem3: change weight fo cross entropy
class CNN_problem3_1(nn.Module):
    def __init__(self):
        super(CNN_problem3_1, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=4),    
        )
        self.out = nn.Linear(16*7*7, 3)   
    def forward(self, x):
        x = self.conv1(x.float())
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
LR = 0.0005              # learning rate
cnn = CNN_problem3_1()
loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor([4.9,27.4,1]))
num_epoch = 100
batch_size = 49 #沒有為什麼, 就是剛好整除看起來很舒服 xD
seed = 777    
problem_out_problem3 = train_model(train_data,test_data,LR,cnn,loss_func,num_epoch,batch_size,seed,lr_dym = True)
plot(problem_out_problem3)

#problem3: resample data
class CNN_problem3_2(nn.Module):
    def __init__(self):
        super(CNN_problem3_2, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=4),    
        )
        self.out = nn.Linear(16*7*7, 3)   
    def forward(self, x):
        x = self.conv1(x.float())
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
LR = 0.0005              # learning rate
cnn = CNN_problem3_2()
loss_func = nn.CrossEntropyLoss()
num_epoch = 100
batch_size = 78 #沒有為什麼, 就是剛好整除看起來很舒服 xD
seed = 777 
num_resample = 50
problem_out_problem3_2 = train_model_resample(train_data,test_data,LR,cnn,loss_func,num_epoch,batch_size,seed,num_resample,lr_dym = True)
plot(problem_out_problem3_2)