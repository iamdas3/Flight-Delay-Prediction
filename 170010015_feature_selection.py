#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


dataset = pd.read_csv("FlightDelays.csv")
dataset.head()


# In[3]:


dataset.dropna()
print(dataset.shape)
print(list(dataset.columns))


# In[4]:


flight_status = dataset["Flight Status"]


# In[5]:


for i in range(len(flight_status.values)):
    if (flight_status.values[i] == 'ontime'):
        flight_status.values[i] = 1
    else : 
        flight_status.values[i] = 0


# In[6]:


dataset['Flight Status'].value_counts()


# In[7]:


count_delayed = len(dataset[dataset['Flight Status']==0])
count_ontime = len(dataset[dataset['Flight Status']==1])
pct_of_delayed = count_delayed/(count_delayed+count_ontime)
print("percentage of Delayed Flights is", pct_of_delayed*100)
pct_of_ontime = count_ontime/(count_ontime+count_delayed)
print("percentage of Ontime Flights is", pct_of_ontime*100)


# In[8]:


cat_vars = ["CARRIER", "DEST", "DISTANCE", "ORIGIN", "Weather", "DAY_WEEK"]
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(dataset[var], prefix=var)
    data1=dataset.join(cat_list)
    dataset=data1


# In[9]:


cat_vars = ["CARRIER", "DEST", "DISTANCE", "ORIGIN", "Weather", "DAY_WEEK"]
data_vars=dataset.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[10]:


data_final=dataset[to_keep]
data_final.columns.values


# In[11]:


time = dataset["DEP_TIME"]


# In[12]:


for i in range(len(time)):
    if (time.values[i]>0 and time.values[i]<60):
        time.values[i] = 0
    elif (time.values[i]>59 and time.values[i]<160):
        time.values[i] = 1
    elif (time.values[i]>159 and time.values[i]<260):
        time.values[i] = 2
    elif (time.values[i]>259 and time.values[i]<360):
        time.values[i] = 3
    elif (time.values[i]>359 and time.values[i]<460):
        time.values[i] = 4
    elif (time.values[i]>459 and time.values[i]<560):
        time.values[i] = 5
    elif (time.values[i]>559 and time.values[i]<660):
        time.values[i] = 6
    elif (time.values[i]>659 and time.values[i]<760):
        time.values[i] = 7
    elif (time.values[i]>759 and time.values[i]<860):
        time.values[i] = 8
    elif (time.values[i]>859 and time.values[i]<960):
        time.values[i] = 9
    elif (time.values[i]>959 and time.values[i]<1060):
        time.values[i] = 10
    elif (time.values[i]>1059 and time.values[i]<1160):
        time.values[i] = 11
    elif (time.values[i]>1159 and time.values[i]<1260):
        time.values[i] = 12
    elif (time.values[i]>1259 and time.values[i]<1360):
        time.values[i] = 13
    elif (time.values[i]>1359 and time.values[i]<1460):
        time.values[i] = 14
    elif (time.values[i]>1459 and time.values[i]<1560):
        time.values[i] = 15
    elif (time.values[i]>1559 and time.values[i]<1660):
        time.values[i] = 16
    elif (time.values[i]>1659 and time.values[i]<1760):
        time.values[i] = 17
    elif (time.values[i]>1759 and time.values[i]<1860):
        time.values[i] = 18
    elif (time.values[i]>1859 and time.values[i]<1960):
        time.values[i] = 19
    elif (time.values[i]>1959 and time.values[i]<2060):
        time.values[i] = 20
    elif (time.values[i]>2059 and time.values[i]<2160):
        time.values[i] = 21
    elif (time.values[i]>2159 and time.values[i]<2260):
        time.values[i] = 22
    elif (time.values[i]>2259 and time.values[i]<2360):
        time.values[i] = 23


# In[13]:


a = list(time.values)
onehot_encoded = list()
for i in range(len(a)):
    letter = [0 for _ in range(24)]
    letter[a[i]] = 1
    onehot_encoded.append(letter)


# In[14]:


df = pd.DataFrame(onehot_encoded, columns = ['DEPT_0', 'DEPT_1', 'DEPT_2', 'DEPT_3', 'DEPT_4', 'DEPT_5', 'DEPT_6', 'DEPT_7', 'DEPT_8', 'DEPT_9', 'DEPT_10', 'DEPT_11', 'DEPT_12', 'DEPT_13', 'DEPT_14', 'DEPT_15', 'DEPT_16', 'DEPT_17', 'DEPT_18', 'DEPT_19', 'DEPT_20', 'DEPT_21', 'DEPT_22', 'DEPT_23'])


# In[15]:


pd.concat([data_final, df], axis = 1)


# In[16]:


data_final.columns.values


# In[17]:


data_final.drop(["CRS_DEP_TIME", "DEP_TIME", "FL_DATE", "FL_NUM","DAY_OF_MONTH", "TAIL_NUM"], axis = 1, inplace = True)
data_final.columns.values


# In[18]:


X = data_final.loc[:, data_final.columns != 'Flight Status']
y = data_final.loc[:, data_final.columns == 'Flight Status']
X


# In[19]:


y = y.astype('int')


# In[ ]:





# In[ ]:





# In[20]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X, y.values.ravel())
print(rfe.support_)
#print(rfe.ranking_)
print(data_final.columns.values)


# In[21]:


X.drop(['CARRIER_DL', 'DEST_EWR', 'DEST_JFK', 'DEST_LGA', 'DISTANCE_199', 'DISTANCE_228', 'ORIGIN_BWI', 'ORIGIN_IAD', 'DAY_WEEK_2', 'DAY_WEEK_3', 'DAY_WEEK_5'], axis = 1, inplace = True)
X.columns.values


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# In[23]:


X_train


# In[24]:


y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[25]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train.values.ravel())
predictions = logmodel.predict(X_test)


# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[ ]:




