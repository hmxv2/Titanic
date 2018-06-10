
# coding: utf-8

# In[212]:


import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import pickle
#import xgboost as xgb
print('imported!')


# In[213]:

#import data for training
path=os.getcwd()+'/train.csv'#/sample_submission.csv'
all_data=pd.read_csv(path)


# In[214]:

#all_data.columns
#all_data.index
#all_data.iloc[1]
data_train=all_data
data_train.info()


# In[215]:

all_data=np.array(all_data)
#shuffle
#np.random.shuffle(all_data)
row_num=len(all_data)
#snapshot
print(all_data[0])
print(all_data[1])
print(all_data[2])
print(row_num)


# In[216]:

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图 
plt.title(u"survived or not") # 标题
plt.ylabel(u"people number")  

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"people number")
plt.title(u"class of passengers")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"age")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title(u"correlation of age and survival")


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")# plots an axis lable
plt.ylabel(u"density") 
plt.title(u"passengers age of three class")
plt.legend((u'1st cabin', u'2nd cabin',u'3rd cabin'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"number of passengers from every embarked")
plt.ylabel(u"people number")
plt.show()


# In[217]:

#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'suvived':Survived_1, u'unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"survival data of three class")
plt.xlabel(u"passenger class") 
plt.ylabel("people number") 
plt.show()


# In[218]:

#看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'male':Survived_m, u'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"survival data of different gender")
plt.xlabel(u"gender") 
plt.ylabel(u"people number")
plt.show()


# In[219]:

def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
#data_train = set_Cabin_type(data_train)


# In[220]:

data_train


# In[221]:

#deal with train data
#set gender to be number
'''
for ii in range(890):
    if data_train.iloc[ii][4]=='female':
        data_train.iloc[ii][4]=0
    elif data_train.iloc[ii][4]=='male':
        data_train.iloc[ii][4]=1
        '''
data_train['Gender']=data_train['Sex'].astype('category')
data_train['Gender'].cat.categories=[0,1]
data_train['Port']=data_train['Embarked'].astype('category')
data_train['Port'].cat.categories=[0,1,2]


# In[222]:

#sample some valuable data
data_train_new=data_train[['Survived','Age','Fare','Parch','SibSp','Pclass','Gender','Port']]
#print(data_train_new)
data_train_np=np.array(data_train_new)
data_train_new.info()


# In[223]:

data_train_np


# In[224]:

row_len=len(data_train_np)
col_len=len(data_train_np[0])
tmp=np.zeros((row_len,col_len))
for ii in range(row_len):
    for jj in range(col_len):
        if math.isnan(data_train_np[ii][jj]):
            print('is null',ii,jj)
            tmp[ii][jj]=-1
        else:
            tmp[ii][jj]=float(data_train_np[ii][jj])
print(tmp[61][7])
#deal with age, 
#someone who 's age<12 is a teenager.
for ii in range(row_len):
    if tmp[ii][1]<16:
        tmp[ii][1]=1
    else:
        tmp[ii][1]=0
#normalize the ticket fare
max_=max(tmp[:,2])
min_=min(tmp[:,2])
tmp[:,2]=(tmp[:,2]-min_)/(max_-min_)
print(sum(sum(tmp)))
save_file=open('train_data.x','wb')
pickle.dump(tmp,save_file,True)
save_file.close()


# In[225]:

tmp


# In[226]:

#deal with test data
data_test = pd.read_csv('test.csv')
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
data_test['Gender']=data_test['Sex'].astype('category')
data_test['Gender'].cat.categories=[0,1]
data_test['Port']=data_test['Embarked'].astype('category')
data_test['Port'].cat.categories=[0,1,2]
#sample some valuable data
data_test_new=data_test[['Age','Fare','Parch','SibSp','Pclass','Gender','Port']]
data_test_np=np.array(data_test_new)
data_test_new.info()

#save
row_len=len(data_test_np)
col_len=len(data_test_np[0])
tmp1=np.zeros((row_len,col_len))
for ii in range(row_len):
    for jj in range(col_len):
        if math.isnan(data_test_np[ii][jj]):
            print('is null',ii,jj)
            tmp1[ii][jj]=-1
        else:
            tmp1[ii][jj]=float(data_test_np[ii][jj])
#deal with age, 
#someone who 's age<16 is a teenager.
for ii in range(row_len):
    if tmp1[ii][0]<16:
        tmp1[ii][0]=1
    else:
        tmp1[ii][0]=0
#normalize the ticket fare
tmp1[:,1]=(tmp1[:,1]-min_)/(max_-min_)
print(sum(sum(tmp1)))
save_file=open('test_data.x','wb')
pickle.dump(tmp1,save_file,True)
save_file.close()


# In[ ]:



