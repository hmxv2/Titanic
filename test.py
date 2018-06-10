
# coding: utf-8

# In[34]:

import csv
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import math

print('imported!')
class Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Net,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,out_dim)
        self.relu=nn.ReLU()#ReLU()#Sigmoid()
        self.softmax=nn.Softmax()
        self.dropout=nn.Dropout(0.45)
    def forward(self,x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        x=self.relu(x)
        x=self.layer3(x)
        x=self.relu(x)

        return x


# In[35]:

file_handle=open('test_data.x','rb')
test_data=pickle.load(file_handle)
file_handle.close()
row_len=len(test_data)
col_len=len(test_data[0])
print(row_len)
#load model
model=torch.load('./bettermodels/epoch160.0loss0.38089102506637573accuracy_rate0.8598130841121495.pkl')
#test
model.eval()
predict_out=model(Variable(torch.from_numpy(test_data)).float())
max_value,max_idx=torch.max(predict_out,1)
#print(max_idx)
predict=max_idx.data.numpy()
print(predict)
print(len(predict))


# In[36]:

with open("result.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerow(["PassengerId","Survived"])
    #写入多行用writerows
    for idx in range(row_len):
        writer.writerow([str((idx+892)),str(predict[idx])])


# In[37]:

sum(predict)/len(predict)


# In[ ]:



