#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import plot_confusion_matrix,accuracy_score,classification_report

from warnings import filterwarnings 
filterwarnings('ignore')


# In[2]:


df = pd.read_csv('weather_classification_data.csv')


# In[3]:


df


# # Understanding the dataset

# In[4]:


df.info()


# In[5]:


df.describe().transpose()


# In[6]:


df.isnull().sum()


# #### Hence no data cleaning required since there are no missing entries

# # Exploratory Data Analysis (EDA)

# In[7]:


df.head(5)


# ### Going through the categorical variables: Cloud cover, Season, Location, Weather type

# In[8]:


df['Cloud Cover'].value_counts()


# In[9]:


df['Season'].value_counts()


# In[10]:


df['Location'].value_counts()


# In[11]:


# Sorting out the categorical coloumns and the numerical coloumns
numeric_cols =df.select_dtypes(include=['number']).columns
categorical_cols =df.select_dtypes(include=['object']).columns


# In[12]:


numeric_cols


# In[13]:


categorical_cols


# In[14]:


# Countplot for all the categorical Factors

fig,axes = plt.subplots(nrows=4,ncols=1,figsize=(8,6))

for i,column in enumerate(categorical_cols):
    sns.countplot(y=df[column],ax=axes[i])
    axes[i].set_title(f'Count of {column}')
    
plt.tight_layout()


# ### Going through the numerical columns

# In[15]:


# Distribution for all the numerical coloumns

fig,axes = plt.subplots(nrows=len(numeric_cols),ncols=1,figsize=(8,12))

for i, column in enumerate(numeric_cols):
    sns.histplot(df[column],ax=axes[i])
    axes[i].set_title(f'Distribution of {column}')

plt.tight_layout()


# ### Some more visualizations

# In[16]:


plt.plot(figsize=(6,6))
sns.barplot(data = df, x='Location',y='Temperature',hue='Weather Type')
plt.legend(bbox_to_anchor=(1.3,1),loc='upper right')
plt.title('Temperatures across different location type')


# In[17]:


plt.plot(figsize=(6,6))
sns.barplot(data = df, x='Season',y='Temperature',hue='Weather Type')
plt.legend(bbox_to_anchor=(1.3,1),loc='upper right')
plt.title('Temperature across different seasons')


# In[18]:


plt.plot(figsize=(6,6))
sns.barplot(data = df, x='Cloud Cover',y='Temperature',hue='Weather Type')
plt.legend(bbox_to_anchor=(1.3,1),loc='upper right')
plt.title('Level of temperatures across different type of cloud cover')


# In[19]:


fig,axes = plt.subplots(nrows=len(numeric_cols),ncols=1,figsize=(10,10))

for i,row in enumerate(numeric_cols):
    sns.barplot(x=df['Weather Type'],y=df[row],ax=axes[i])
    axes[i].set_title(f'{row} relative to different Weather Types')

plt.tight_layout()


# ### Let us see how changes in UV Index affects the average temperature

# In[20]:


UT = df.groupby('UV Index')['Temperature'].mean()
UT.plot(kind='line',marker='o',color='red')
plt.grid(True)
plt.xlabel('UV Index')
plt.ylabel('Average Temperature')
plt.title('Affect on average temperature due to UV Index')


# ### Let us see how changes in Humidity levels affect the average visibility

# In[21]:


VH = df.groupby('Humidity')['Visibility (km)'].mean()
VH.plot(kind='line',color='green')
plt.title('Affect on Visibility due to Humidity',fontsize=10)
plt.grid(True)
plt.xlabel('Humidity',fontsize=8)
plt.ylabel('Average Average Visibility',fontsize=8)


# In[ ]:





# In[ ]:





# ### Let us see how changes in Precipitation levels affect the average Atmospheric Pressure

# In[22]:


PA=df.groupby('Precipitation (%)')['Atmospheric Pressure'].mean()
PA.plot(kind='line',figsize=(8,6),lw=1)
plt.grid(True)
plt.xlabel('Precipitation')
plt.ylabel('Atmospheric Pressure')
plt.title('Affect on Average Atmospheric Pressure due to changes in Precipitation level')


# ### Visualizations of correlation between factors

# In[23]:


plt.plot(figsize=(8,8),dpi=200)
sns.heatmap(data=df.corr(),annot = True)
plt.title('Correlation between different numerical factors',fontsize=20,pad=20)


# In[24]:


temp_corr = df.corr()['Temperature'].sort_values()
temp_corr.plot(kind='bar',color='orange')
plt.title('Correlation between Temperature and different numerical factors')
plt.ylabel('Correlation Coeffecient')


# # Data Processing

# In[25]:


X=df.drop('Weather Type',axis=1)


# In[26]:


X = pd.get_dummies(X,drop_first=True)


# In[27]:


y=df['Weather Type']


# ### We splitted the Target Variable from the rest of the independent variables and now we split them into trainning and testing sets

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ###  We scaled the entire X set because Logistic Regression is sensitive to the scale of the data and it enhances the performance of the machine learning algorithm

# In[29]:


scaler = StandardScaler()


# In[30]:


scaled_X_train = scaler.fit_transform(X_train)


# In[31]:


scaled_X_test = scaler.transform(X_test)


# # 1. LOGISTIC REGRESSION

# ### Setting up our Logistic Regression Model with certain parameters

# In[32]:


log_model = LogisticRegression(solver='saga',multi_class = 'ovr',max_iter = 500)


# ### Finding the best hyperparameters with the help of Grid Search Cross Validation

# In[33]:


penalty = ['l1','l2','elasticnet']


# In[34]:


l1_ratio = np.linspace(0,1,5)


# In[35]:


C = np.logspace(0,10,5)


# In[36]:


param_grid = {'penalty':penalty,
             'l1_ratio':l1_ratio,
             'C':C}


# In[37]:


grid_model = GridSearchCV(log_model,param_grid = param_grid)


# ### Fitting our model and finding the best hyperparameters

# In[38]:


grid_model.fit(scaled_X_train,y_train)


# ### These are our best hyperparameters 

# In[39]:


grid_model.best_params_


# ### Prediciting the Target labels based on our model

# In[40]:


y_grid_pred = grid_model.predict(scaled_X_test)


# In[41]:


accuracy_score(y_test,y_grid_pred)


# In[42]:


print(classification_report(y_test,y_grid_pred))


# In[43]:


plot_confusion_matrix(grid_model,scaled_X_test,y_test)


# ## So we get an accuracy score of 85.4% meaning 85 out of 100 times we would be correct in predicting the correct outcome

# # 2. DECISION TREE LEARNING

# In[44]:


decision_tree_model = DecisionTreeClassifier()


# In[45]:


decision_tree_model.fit(X_train,y_train)


# In[46]:


decision_tree_pred= decision_tree_model.predict(X_test)


# ### Finding out the most important feature

# In[47]:


imp_feats = pd.DataFrame(index=X.columns,data=decision_tree_model.feature_importances_,columns=['Feature Importances']).sort_values('Feature Importances')


# In[48]:


plt.figure(figsize=(6,6))
sns.barplot(data=imp_feats,x=imp_feats.index,y='Feature Importances')
plt.xticks(rotation=90);


# ### Temperature is the most important feature in this model

# In[49]:


print(classification_report(y_test,decision_tree_pred))


# In[50]:


accuracy_score(y_test,decision_tree_pred).round(2)


# In[51]:


plot_confusion_matrix(decision_tree_model,X_test,y_test)


# ## So just with default parameters we are getting an accuracy score of 91 %

# # 3. RANDOM FOREST CLASSIFIER

# In[52]:


rfc_model = RandomForestClassifier()


# In[53]:


rfc_model.fit(X_train,y_train)


# In[54]:


rfc_preds = rfc_model.predict(X_test)


# In[55]:


print(classification_report(y_test,rfc_preds))


# In[56]:


accuracy_score(y_test,rfc_preds).round(2)


# ## With default parameters we are having 92 percent accuracy, let us perform Gridsearch over other parameters to check if there is any scope for improvement

# ### Performing Grid Search for the best hyper parameters

# In[57]:


n_estimators = [64,100,128,200]
max_features = [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]


# In[58]:


param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap,
             'oob_score':oob_score}


# In[59]:


rfc_model=RandomForestClassifier()


# In[ ]:





# In[60]:


grid_rfc_model = GridSearchCV(rfc_model,param_grid)


# In[61]:


grid_rfc_model.fit(X_train,y_train)


# In[62]:


grid_rfc_model.best_params_


# In[63]:


y_grid_rfc_pred = grid_rfc_model.predict(X_test)


# In[ ]:





# In[64]:


accuracy_score(y_test,y_grid_rfc_pred).round(2)


# In[65]:


plot_confusion_matrix(grid_rfc_model,X_test,y_test)


# ## With Random Forest Classification we are having 92 % accuracy

# # 4. SUPPORT VECTOR CLASSIFIER

# ### Performing Grid Search Cross Validation for the best hyper parameters

# In[66]:


svm_model = SVC()
param_grid = {'C':[0.000001,0.01,0.1,1],
              'kernel':['linear','rbf']}
svm_grid=GridSearchCV(svm_model,param_grid)
svm_grid.fit(scaled_X_train,y_train)


# In[67]:


svm_grid.best_params_


# In[68]:


svm_y_preds = svm_grid.predict(scaled_X_test)


# In[77]:


accuracy_score(y_test,svm_y_preds).round(2)


# In[78]:


print(classification_report(y_test,svm_y_preds))


# In[79]:


plot_confusion_matrix(svm_grid,scaled_X_test,y_test)


# ## Let us test our model for one random instance

# In[70]:


X_test[:1]


# In[71]:


testing_set = [[13.0,71,9.0,41.0,1019.47,1,8.0,0,0,1,0,1,0,1,0]]
testing_set


# ###  Given these set of information our models are predicting it to be a cloudy weather type

# In[72]:


grid_rfc_model.predict(testing_set)


# In[80]:


grid_model.predict(testing_set)


# In[82]:


decision_tree_model.predict(testing_set)


# In[74]:


df.iloc[12182]


# ### !! From our dataset we can see it is indeed cloudy 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




