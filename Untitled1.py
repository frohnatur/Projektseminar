#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sqlite3

from sqlite3 import Error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[6]:


df=pandas.read_csv(r"C:\Users\Yanina\Desktop\ws20\imputed_data.csv")


# In[7]:


df.describe()
df.groupby(['immobilienart']).describe()


# In[8]:


df_red=df.drop(columns=['breitengrad','einwohner','laengengrad','aufzug','barrierefrei','energietyp','energie_effizienzklasse','gaeste_wc','heizung','immobilienzustand','terrasse_balkon','unterkellert','vermietet','Unnamed: 0'])


# In[9]:


df_red
df_red.info()


# In[10]:


df_nan=df_red.dropna()
df_nan


# In[11]:


df_nan.info()


# In[15]:


data = df_nan # load data set
X = df_nan[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche','wohnflaeche']]  # values converts it into a numpy array
Y = df_nan['angebotspreis']  # -1 means that calculate the dimension of rows, but have 1 column
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X_train, y_train)  # perform linear regression
Y_pred = linear_regressor.predict(X_test)  # 


# In[16]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[17]:


df_nan


# In[200]:





# In[ ]:





# In[106]:


data = df_nan
X = df_nan[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche']]  # values converts it into a numpy array
Y = df_nan['angebotspreis']  # -1 means that calculate the dimension of rows, but have 1 column
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
Y_pred=rf.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))


# In[107]:


dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[21]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[111]:


X = df_nan[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche']]  # values converts it into a numpy array
Y = df_nan['angebotspreis']  # -1 means that calculate the dimension of rows, but have 1 column
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.2, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 20)
xg_reg.fit(X_train, y_train)
Y_pred=xg_reg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))


# In[112]:


dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[23]:


for col in ['immobilienart']:
    df_red[col] = df_red[col].astype('category')
    print(df_nan.dtypes)


# In[144]:


df_nan=df_red.dropna()
print(df_nan.dtypes)


# In[145]:


df_nan_hot = pd.get_dummies(data=df_nan,columns=['baujahr'])


# In[146]:


df_nan_hot


# In[27]:


X = df_nan_hot[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche',
                'immobilienart_Apartment', 'immobilienart_Bauernhaus','immobilienart_Bungalow',
                'immobilienart_Dachgeschosswohnung','immobilienart_Doppelhaushälfte','immobilienart_Einfamilienhaus',
                'immobilienart_Erdgeschosswohnung','immobilienart_Etagenwohnung','immobilienart_Herrenhaus','immobilienart_Maisonette',
                'immobilienart_Mehrfamilienhaus','immobilienart_Penthouse', 'immobilienart_Reiheneckhaus', 'immobilienart_Reihenendhaus',
               'immobilienart_Reihenmittelhaus', 'immobilienart_Schloss', 'immobilienart_Sonstige','immobilienart_Sonstiges',
                'immobilienart_Stadthaus','immobilienart_Unbekannt','immobilienart_Villa','immobilienart_Wohnung','immobilienart_Zweifamilienhaus']]  # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']  # -1 means that calculate the dimension of rows, but have 1 column
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 20)
xg_reg.fit(X_train, y_train)
Y_pred=xg_reg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[28]:


X = df_nan_hot[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche',
                'immobilienart_Apartment', 'immobilienart_Bauernhaus','immobilienart_Bungalow',
                'immobilienart_Dachgeschosswohnung','immobilienart_Doppelhaushälfte','immobilienart_Einfamilienhaus',
                'immobilienart_Erdgeschosswohnung','immobilienart_Etagenwohnung','immobilienart_Herrenhaus','immobilienart_Maisonette',
                'immobilienart_Mehrfamilienhaus','immobilienart_Penthouse', 'immobilienart_Reiheneckhaus', 'immobilienart_Reihenendhaus',
               'immobilienart_Reihenmittelhaus', 'immobilienart_Schloss', 'immobilienart_Sonstige','immobilienart_Sonstiges',
                'immobilienart_Stadthaus','immobilienart_Unbekannt','immobilienart_Villa','immobilienart_Wohnung','immobilienart_Zweifamilienhaus']]  # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
Y_pred=rf.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[29]:


X = df_nan_hot[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche',
                'immobilienart_Apartment', 'immobilienart_Bauernhaus','immobilienart_Bungalow',
                'immobilienart_Dachgeschosswohnung','immobilienart_Doppelhaushälfte','immobilienart_Einfamilienhaus',
                'immobilienart_Erdgeschosswohnung','immobilienart_Etagenwohnung','immobilienart_Herrenhaus','immobilienart_Maisonette',
                'immobilienart_Mehrfamilienhaus','immobilienart_Penthouse', 'immobilienart_Reiheneckhaus', 'immobilienart_Reihenendhaus',
               'immobilienart_Reihenmittelhaus', 'immobilienart_Schloss', 'immobilienart_Sonstige','immobilienart_Sonstiges',
                'immobilienart_Stadthaus','immobilienart_Unbekannt','immobilienart_Villa','immobilienart_Wohnung','immobilienart_Zweifamilienhaus']]  # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
linear_regressor=LinearRegression()
linear_regressor.fit(X_train, y_train)
Y_pred=linear_regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[60]:


df_nan_hot


# In[321]:


pip install tqdm


# In[325]:


from tqdm.notebook import tqdm, trange
import time    # to be used in loop iterations


# In[332]:


pip install --upgrade category_encoders


# In[120]:


df_nan_hot = pd.get_dummies(data=df_nan,columns=['immobilienart', 'plz'])


# In[149]:


df_nan_hot.head()


# In[96]:


X = df_nan_hot.drop(columns=["angebotspreis"]).values # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
linear_regressor=LinearRegression()
linear_regressor.fit(X_train, y_train)
Y_pred=linear_regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[97]:


X = df_nan_hot.drop(columns=["angebotspreis"]).values # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
Y_pred=rf.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[98]:


X = df_nan_hot.drop(columns=["angebotspreis"]).values # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 20)
xg_reg.fit(X_train, y_train)
Y_pred=xg_reg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[150]:


means = df_nan_hot.groupby('plz')['angebotspreis'].mean()
means


# In[151]:


df_nan_hot['plz']=df_nan_hot['plz'].map(means)


# In[152]:


df_nan_hot


# In[79]:


X = df_nan_hot[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche',
                'immobilienart_Apartment', 'immobilienart_Bauernhaus','immobilienart_Bungalow',
                'immobilienart_Dachgeschosswohnung','immobilienart_Doppelhaushälfte','immobilienart_Einfamilienhaus',
                'immobilienart_Erdgeschosswohnung','immobilienart_Etagenwohnung','immobilienart_Herrenhaus','immobilienart_Maisonette',
                'immobilienart_Mehrfamilienhaus','immobilienart_Penthouse', 'immobilienart_Reiheneckhaus', 'immobilienart_Reihenendhaus',
               'immobilienart_Reihenmittelhaus', 'immobilienart_Schloss', 'immobilienart_Sonstige','immobilienart_Sonstiges',
                'immobilienart_Stadthaus','immobilienart_Unbekannt','immobilienart_Villa','immobilienart_Wohnung','immobilienart_Zweifamilienhaus','plz']]  # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
linear_regressor=LinearRegression()
linear_regressor.fit(X_train, y_train)
Y_pred=linear_regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[81]:


X = df_nan_hot[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche',
                'immobilienart_Apartment', 'immobilienart_Bauernhaus','immobilienart_Bungalow',
                'immobilienart_Dachgeschosswohnung','immobilienart_Doppelhaushälfte','immobilienart_Einfamilienhaus',
                'immobilienart_Erdgeschosswohnung','immobilienart_Etagenwohnung','immobilienart_Herrenhaus','immobilienart_Maisonette',
                'immobilienart_Mehrfamilienhaus','immobilienart_Penthouse', 'immobilienart_Reiheneckhaus', 'immobilienart_Reihenendhaus',
               'immobilienart_Reihenmittelhaus', 'immobilienart_Schloss', 'immobilienart_Sonstige','immobilienart_Sonstiges',
                'immobilienart_Stadthaus','immobilienart_Unbekannt','immobilienart_Villa','immobilienart_Wohnung','immobilienart_Zweifamilienhaus','plz']]  # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
Y_pred=rf.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[82]:


X = df_nan_hot[['wohnflaeche','anzahl_badezimmer','anzahl_parkplatz','anzahl_zimmer','grundstuecksflaeche',
                'immobilienart_Apartment', 'immobilienart_Bauernhaus','immobilienart_Bungalow',
                'immobilienart_Dachgeschosswohnung','immobilienart_Doppelhaushälfte','immobilienart_Einfamilienhaus',
                'immobilienart_Erdgeschosswohnung','immobilienart_Etagenwohnung','immobilienart_Herrenhaus','immobilienart_Maisonette',
                'immobilienart_Mehrfamilienhaus','immobilienart_Penthouse', 'immobilienart_Reiheneckhaus', 'immobilienart_Reihenendhaus',
               'immobilienart_Reihenmittelhaus', 'immobilienart_Schloss', 'immobilienart_Sonstige','immobilienart_Sonstiges',
                'immobilienart_Stadthaus','immobilienart_Unbekannt','immobilienart_Villa','immobilienart_Wohnung','immobilienart_Zweifamilienhaus','plz']]  # values converts it into a numpy array
Y = df_nan_hot['angebotspreis']  # -1 means that calculate the dimension of rows, but have 1 column
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 20)
xg_reg.fit(X_train, y_train)
Y_pred=xg_reg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
dflinreg = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
dflinreg


# In[125]:


df_nan_hot.corr()


# In[153]:


import seaborn as sn
import matplotlib.pyplot as plt
df_nan_hot


# In[154]:


df = pd.DataFrame(df_nan_hot,columns=['angebotspreis','anzahl_badezimmer','anzahl_parkplatz','grundstuecksflaeche','wohnflaeche','plz','immobilienart'])


# In[155]:


corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
plt.show()

