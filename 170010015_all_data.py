#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
plt.rc("font", size = 14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style = "white")
sns.set(style = "whitegrid", color_codes = True)


# In[2]:


dataset = pd.read_csv("FlightDelays.csv", header = 0)


# In[3]:


dataset.head()


# In[4]:


dataset.dropna()
print(dataset.shape)
print(list(dataset.columns))


# In[5]:


day_of_week = dataset["DAY_WEEK"]
dept_time = dataset["DEP_TIME"]
origin = dataset["ORIGIN"]
dest = dataset["DEST"]
carrier = dataset["CARRIER"]
Weather = dataset["Weather"]
flight_status = dataset["Flight Status"]
flight_status[53]


# In[6]:


for i in range(len(flight_status.values)):
    if (flight_status.values[i] == 'ontime'):
        flight_status.values[i] = 1
    else : 
        flight_status.values[i] = 0


# In[7]:


flight_status[53]


# In[8]:


dataset['Flight Status'].value_counts()


# In[9]:


sns.countplot(x = 'Flight Status', data = dataset, palette = 'hls')
plt.show()


# In[10]:


count_delayed = len(dataset[dataset['Flight Status']==0])
count_ontime = len(dataset[dataset['Flight Status']==1])
pct_of_delayed = count_delayed/(count_delayed+count_ontime)
print("percentage of Delayed Flights is", pct_of_delayed*100)
pct_of_ontime = count_ontime/(count_ontime+count_delayed)
print("percentage of Ontime Flights is", pct_of_ontime*100)


# In[11]:


dataset.groupby('Flight Status').mean()


# In[12]:


dataset.groupby('ORIGIN').mean()


# In[13]:


dataset.groupby("Weather").mean()


# In[14]:


dataset.groupby("CARRIER").mean()


# In[15]:


dataset.groupby("DEP_TIME").mean()


# In[16]:


dataset.groupby("DEST").mean()


# In[17]:


dataset.groupby("DAY_WEEK").mean()


# In[18]:


pd.crosstab(dataset.CARRIER, dataset['Flight Status']).plot(kind='bar')
plt.title("Flight Status for Carriers")
plt.xlabel("Carrier")
plt.ylabel("Frequency of Delay")
plt.legend()


# In[19]:


table=pd.crosstab(dataset.ORIGIN,dataset["Flight Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of ORIGIN vs Flight Status')
plt.xlabel("ORIGIN")
plt.ylabel('FLight Status')


# In[20]:


table=pd.crosstab(dataset.DEST,dataset["Flight Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of DESTINATION vs Flight Status')
plt.xlabel("DEST")
plt.ylabel('FLight Status')


# In[21]:


table=pd.crosstab(dataset.Weather,dataset["Flight Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Weather vs Flight Status')
plt.xlabel("Weather")
plt.ylabel('FLight Status')


# In[22]:


pd.crosstab(dataset.DAY_WEEK, dataset['Flight Status']).plot(kind='bar')
plt.title("Flight Status for Carriers")
plt.xlabel("Day of Week")
plt.ylabel("Frequency of Delay")
plt.legend()


# In[23]:


dataset.DEP_TIME.hist()
plt.title("Histogram of Departure Time")
plt.xlabel("Departure Time")
plt.ylabel("Frequency")


# In[24]:


dataset.info()


# In[25]:


carrier = pd.get_dummies(dataset["CARRIER"])
dest = pd.get_dummies(dataset["DEST"])
distance = pd.get_dummies(dataset["DISTANCE"])
origin = pd.get_dummies(dataset["ORIGIN"])
weather = pd.get_dummies(dataset["Weather"])
day = pd.get_dummies(dataset["DAY_WEEK"])


# In[26]:


time = dataset["DEP_TIME"]


# In[27]:


dataset.drop(["CRS_DEP_TIME", "CARRIER", "DEP_TIME", "DEST", "DISTANCE", "FL_DATE", "FL_NUM", "ORIGIN", "Weather", "DAY_WEEK", "DAY_OF_MONTH", "TAIL_NUM"], axis = 1, inplace = True)
dataset.head()


# In[28]:


dataset = pd.concat([carrier, dest,distance,origin,weather,day, dataset["Flight Status"]], axis =1 )


# In[29]:


dataset["Flight Status"]


# In[30]:


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


# In[31]:


import numpy as np
x = np.array(time.values)
np.unique(x)


# In[32]:


a = list(time.values)
onehot_encoded = list()
for i in range(len(a)):
    letter = [0 for _ in range(24)]
    letter[a[i]] = 1
    onehot_encoded.append(letter)


# In[33]:


df = pd.DataFrame(onehot_encoded, columns = ['DEPT_0', 'DEPT_1', 'DEPT_2', 'DEPT_3', 'DEPT_4', 'DEPT_5', 'DEPT_6', 'DEPT_7', 'DEPT_8', 'DEPT_9', 'DEPT_10', 'DEPT_11', 'DEPT_12', 'DEPT_13', 'DEPT_14', 'DEPT_15', 'DEPT_16', 'DEPT_17', 'DEPT_18', 'DEPT_19', 'DEPT_20', 'DEPT_21', 'DEPT_22', 'DEPT_23'])


# In[34]:


pd.concat([dataset, df], axis = 1)


# In[35]:


dataset


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop('Flight Status',axis=1),dataset['Flight Status'], test_size=0.40,random_state=101)


# In[37]:


Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[38]:


print(Y_train)
print(Y_test)


# In[39]:


logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))


# In[ ]:




