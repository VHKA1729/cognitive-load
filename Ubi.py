#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd
import numpy as np 
import statistics as st
from scipy.stats import skew 
from scipy.stats import kurtosis


# In[184]:


from scipy import signal
from matplotlib import pyplot as plt


# In[185]:


gsr=pd.read_csv('train/gsr_train.csv',header=None)
rr=pd.read_csv('train/rr_train.csv',header=None)
hr=pd.read_csv('train/hr_train.csv',header=None)
temp=pd.read_csv('train/temp_train.csv',header=None)
label=pd.read_csv('train/labels_train.csv',header=None)


# In[186]:


gsr_cs=np.cumsum(gsr,axis=1)[29]
hr_cs=np.cumsum(hr,axis=1)[29]
rr_cs=np.cumsum(rr,axis=1)[29]


# In[187]:


#Normalizing 
from sklearn.preprocessing import Normalizer
gsr.iloc[:,:] = Normalizer(norm='l1').fit_transform(gsr)
rr.iloc[:,:] = Normalizer(norm='l1').fit_transform(rr)
hr.iloc[:,:] = Normalizer(norm='l1').fit_transform(hr)
temp.iloc[:,:] = Normalizer(norm='l1').fit_transform(temp)

#(gsr)


# In[188]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier


# Power Spectrum Calculation

# In[189]:


gsr_ps=[]
for i in range(0,632):    
    freqs, psd = signal.welch(gsr.iloc[i])
    gsr_ps.append(np.mean(psd))
rr_ps=[]
for i in range(0,632):    
    freqs, psd = signal.welch(rr.iloc[i])
    rr_ps.append(np.mean(psd))
hr_ps=[]
for i in range(0,632):    
    freqs, psd = signal.welch(hr.iloc[i])
    hr_ps.append(np.mean(psd))


# In[190]:


gsr_mean=(gsr.mean(axis=1))
gsr_mean2_gsr=(gsr.mean(axis=1))**2
gsr_var=(gsr.var(axis=1))
gsr_med=gsr.median(axis=1)
gsr_kurt=gsr.kurt(axis=1)
 #gsr_ps
hr_mean=(hr.mean(axis=1))
hr_mean2=(hr.mean(axis=1))**2
hr_var=(hr.var(axis=1))
hr_med=hr.median(axis=1)
hr_kurt=hr.kurt(axis=1)
#hr_ps

rr_mean=(rr.mean(axis=1))
rr_mean2=(rr.mean(axis=1))**2
rr_var=(rr.var(axis=1))
rr_med=rr.median(axis=1)
rr_kurt=rr.kurt(axis=1)

#rr_ps


# In[200]:


df=pd.DataFrame(list(zip(gsr_mean,gsr_var,gsr_med,gsr_kurt,gsr_cs,gsr_ps,hr_mean,hr_var,hr_med,hr_kurt,hr_cs,hr_ps,rr_mean,rr_var,rr_med,rr_kurt,rr_cs,rr_ps,label[0])))


# In[201]:


df.columns=['gsr_mean','gsr_var','gsr_med','gsr_kurt','gsr_cs','gsr_ps','hr_mean','hr_var','hr_med','hr_kurt','hr_cs','hr_ps','rr_mean','rr_var','rr_med','rr_kurt','rr_cs','rr_ps','label']


# In[202]:

