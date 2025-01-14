#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import importlib
import fncs
import matplotlib.pyplot as plt
import random
import pandas as pd

importlib.reload(fncs)


# Helper Function

# In[54]:


import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# This function produces a summary of performance metrics including a confusion matrix
def summaryPerf(yTrain,yTrainHat,y,yHat):
    # Plotting confusion matrix for the non-training set:
    cm = metrics.confusion_matrix(y,yHat,normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=
                                  ['Walk Hard','Down Stairs','Up Stairs','Walk Soft'])
    disp.plot()

    # Displaying metrics for training and non-training sets
    print('Training:  Acc = {:4.3f}'.format(metrics.accuracy_score(yTrain,yTrainHat)))
    print('Non-Train: Acc = {:4.3f}'.format(metrics.accuracy_score(y,yHat)))
    print('Training:  BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(yTrain,yTrainHat)))
    print('Non-Train: BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(y,yHat)))


# ## Load and visualize distribution of classes to pick validation trials

# In[55]:


X_trial_path = "./GaitData/data/train/"

#trainIDs = list(set(np.array(range(29))+1))
trainIDs=[1]

def load_and_viz_data(path,trainIDs):
    
    # Matplot lib stuff
    fig, axs = plt.subplots(5, 6, figsize=(2*18, 2*15))
    axs = axs.flatten()
   
    Xts = []
    Xvs = []
    yts = []
    yvs = []
    for i, idx in enumerate(trainIDs):
        Xt, Xv, yt, yv = fncs.loadTrial(path, idx)
        Xts.append(Xt)
        Xvs.append(Xv)
        yts.append(yt)
        yvs.append(yv)

        unique_classes, class_counts = np.unique(yv, return_counts=True)
        
        axs[i].bar(unique_classes, class_counts, color='skyblue')
        axs[i].set_title(f'Trial {idx} Class Distribution')
        axs[i].set_xlabel('Class Label')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xticks(unique_classes)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Want to visualize the distribution of training trials
    for i in range(len(trainIDs), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
    return Xts, Xvs, yts, yvs

Xts, Xvs, yts, yvs = load_and_viz_data(X_trial_path,trainIDs)


# ## Choosing Validation Trials [1, 8, 9, 20, 27] to get most heterogenous trials

# ## Defining Window Extraction Function (reusing some code from the previous notebook)

# In[57]:


def extractWindow(xt,xv,winSz,timeStart,timeEnd,timeStep):
    tList = []
    windowList = []
    # Specifying the initial window for extracting features
    t0 = timeStart
    t1 = t0+winSz

    while(t1<=timeEnd):
        # Using the middle time of the window as a reference time
        tList.append((t0+t1)/2)

        # Extracting xWindow
        xWin = xv[(xt>=t0)*(xt<=t1),:]
        windowList.append(xWin)
        # Updating the window by shifting it by the step size
        t0 = t0+timeStep
        t1 = t0+winSz

    tList = np.array(tList)
    windowList = np.array(windowList)

    return tList, windowList


# In[58]:


# Extract label takes mode over window range. This should be fine.
timeStep = 1
winSz = 3
time_start = 0
time_end = 60

# Temporary to test windowing function

Xt_train = Xts[0] 
Xv_train = Xvs[0]
yt_train = yts[0]
yv_train = yvs[0]

time_list_temp, window_list_temp = extractWindow(Xt_train, Xv_train, winSz, time_start, time_end, timeStep)


# ### Dims: 1D array of all mean window times

# In[59]:


time_list_temp, time_list_temp.shape 


# ### Dims: Number of windows x samples per window x number of channels

# In[60]:


window_list_temp.shape


# #### Checking the extract label for y's

# In[61]:


yt_temps, yv_temps = fncs.extractLabel(yt_train, yv_train, winSz, time_start, time_end, timeStep)


# In[62]:


yt_temps.shape, yv_temps.shape


# In[63]:


time_list_temp


# In[64]:


yt_temps


# Looks good. Now we extract the windows from each trial and save to training and validation sets.

# ### Raw data to windowed data

# Specify hyperparameters window size and step size

# In[65]:


def loadWindows(dataFolder,winSz,timeStep,idList):
    for k,id in enumerate(idList):
        # Loading the raw data
        xt, xv, yt, yv = fncs.loadTrial(dataFolder,id=id)

        # Extracting the time window for which we have values for the measurements and the response
        timeStart = np.max((np.min(xt),np.min(yt)))
        timeEnd = np.min((np.max(xt),np.max(yt)))

        # Extracting the features
        x_times, x_windows = extractWindow(xt,xv,winSz,timeStart,timeEnd,timeStep)
        y_times, lab = fncs.extractLabel(yt,yv,winSz,timeStart,timeEnd,timeStep)

        # Storing the features
        if(k==0):
            window_list = x_windows
            labList = lab
            x_times_list = x_times
            y_times_list = y_times
        else:
            window_list = np.concatenate((window_list,x_windows),axis=0)
            labList = np.concatenate((labList,lab),axis=0)
            x_times_list = np.concatenate((x_times_list,x_times),axis=0)
            y_times_list = np.concatenate((y_times_list,y_times),axis=0)

    return window_list, labList, x_times_list, y_times_list


# In[66]:


window_size = 3
time_step = 1

val_idxs = [1, 8, 9, 20, 27]
train_idxs = [i for i in range(1,30) if i not in val_idxs]

Xw_train, yw_train, xt_train, _ = loadWindows(X_trial_path,window_size,time_step,train_idxs)
Xw_val, yw_val, xt_val, _ = loadWindows(X_trial_path,window_size,time_step,val_idxs)

    


# In[67]:


Xw_train.shape


# ## Preprocessing

# In[68]:


import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


# Define weights which to weight cross-entropy function based on class imbalances

# In[69]:


# Weights for the loss function scaled by class
class_distribution = np.unique(yw_train, return_counts=True)[1]
scale_factor = 2
weights = (scale_factor*(1/class_distribution)/np.linalg.norm(1/class_distribution)).tolist()


# In[70]:


weights


# ## Trying synthetic data

# In[71]:


Xw_train.shape


# In[72]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

# fit predictor and target
temp_shape = Xw_train.shape
Xw_train = Xw_train.reshape(Xw_train.shape[0],-1)

Xw_train, yw_train = ros.fit_resample(Xw_train, yw_train)

Xw_train = Xw_train.reshape([-1,120,6])


# In[73]:


import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import normalize


def create_dataloader(X, y, batch_size, device, shuffle=False):
    X = torch.from_numpy(X).float()
    
    # Normalizing the data along LAST "channel" dimension (actual data)
    X = normalize(X, dim = 2)

    # Reshaping the data to be in the format (batch, channels, data)
    X = X.permute(0, 2, 1)
    
    X = X.to(device)
    y = torch.from_numpy(y).long()
    y = y.to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

batch_size = 200
train_dataloader, train_dataset = create_dataloader(Xw_train, yw_train, batch_size, device, shuffle=True)
val_dataloader, val_dataset = create_dataloader(Xw_val, yw_val, batch_size, device, shuffle=False)


# ## Simple CNN

# In[77]:


simple_cnn = nn.Sequential(
    nn.Conv1d(6, 12, kernel_size=3, padding="same"),
    nn.ReLU(),
    nn.Conv1d(12, 24, kernel_size=3, padding="same"),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2, stride=2),
    nn.Dropout(p=0.5) ,

    nn.Flatten(),
    nn.Linear(24*60, 200),
    nn.ReLU(),
    nn.Linear(200, 4),
)


# In[78]:


simple_cnn =  simple_cnn.to(device)

# criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.75,1,1,1]).to(device))
optimizer = optim.SGD(simple_cnn.parameters(), lr=0.001, momentum=0.999, weight_decay=0.01)

epochs = 100
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = simple_cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


# In[ ]:


y_train_hat = simple_cnn(train_dataset.tensors[0].to(device)).argmax(1).cpu().numpy()
y_val_hat = simple_cnn(val_dataset.tensors[0].to(device)).argmax(1).cpu().numpy()
summaryPerf(yw_train,y_train_hat,yw_val,y_val_hat)


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def cm(y, y_hat):
    cm = confusion_matrix(y, y_hat, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walk Hard','Down Stairs','Up Stairs','Walk Soft'])
    disp.plot()
cm(yw_train, y_train_hat)


# ## Hyperparameter Tuning

# In[ ]:


import optuna

class CNN_simple(nn.Module): 
    def __init__(self, dropout_p=0.5, kernel1=3, kernel2=3, layer1 = 12,layer2 = 24, hidden_fc = 200):
        super(CNN_simple, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(6, layer1, kernel_size=kernel1, padding="same"),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(layer1, layer2, kernel_size=kernel2, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_p), 

            nn.Flatten(),
            nn.Linear(layer2*60, hidden_fc),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(hidden_fc, 4),
        )

    def forward(self, x):
        return self.model(x)

def objective(trial, epochs=90):
    # Hyperparameters
    dropout_p = trial.suggest_float('dropout_p', 0, 0.5)
    kernel1 = trial.suggest_int('kernel1', 3, 5)
    kernel2 = trial.suggest_int('kernel2', 3, 5)
    
    layer2 = trial.suggest_int('layer2', 24, 48)

    weight_decays = trial.suggest_loguniform('weight_decays', 1e-5, 1e-2)

    hidden_fc = trial.suggest_int('hidden_fc', 100, 500)

    weight1 = trial.suggest_float('weight1', 1, 2)

    # For criterion
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    temp_model = CNN_simple(dropout_p=dropout_p, kernel1=kernel1, kernel2=kernel2, layer1=12,
                            layer2=layer2, hidden_fc=hidden_fc)
    temp_model.to(device)
    
    optimizer_temp = optim.Adam(temp_model.parameters(), lr=learning_rate, weight_decay=weight_decays)
    criterion_temp = nn.CrossEntropyLoss(weight=torch.tensor([weight1, 1, 1, 1]).to(device))

    for epoch in range(epochs):
        temp_model.train()
        for inputs, labels in train_dataloader:
            optimizer_temp.zero_grad()
            outputs = temp_model(inputs)
            loss = criterion_temp(outputs, labels)
            loss.backward()
            optimizer_temp.step()

        # validation
        temp_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in val_dataloader:
                outputs = temp_model(inputs)
                loss = criterion_temp(outputs, labels)
                val_loss += loss.item()
        trial.report(val_loss, epoch)

    return val_loss        


# In[52]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

# Retrieve best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)


# In[ ]:





# In[75]:





# In[ ]:




