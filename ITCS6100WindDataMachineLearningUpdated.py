#!/usr/bin/env python
# coding: utf-8

# # ITCS 6100 - Group 12 project
# ## NREL Wind Integration National Dataset
# ### Data pulled from wtk-us.h5 file from the NREL dataset
# ### located: arn:aws:s3:::nrel-pds-hsds/nrel/wtk-us.h5

# In[1]:


# At Terminal: Install HSDS from the HDF group to access the .h5 file
# pip install --user h5pyd
# pip install pyproj
get_ipython().run_line_magic('pip', 'install --user h5pyd')
get_ipython().run_line_magic('pip', 'install pyproj')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import h5pyd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyproj import Proj
import dateutil


# ## Save the .h5 data file so it can be viewed via Python

# In[3]:


# Open the wind data "file"
# server endpoint, username, password is found via a config file
f = h5pyd.File("/nrel/wtk-us.h5", 'r', endpoint='https://developer.nrel.gov/api/hsds', bucket="nrel-pds-hsds", api_key='CG74xbnQohlDJZdN2hn0QBAZnNbKn78SMyFZWahg')


# In[4]:


print(f.keys())
list(f.attrs)


# ## List of Datapoints / Features

# In[5]:


# list datapoints]
list(f)


# ## Exploring the data
# ### Saving an object with just 1 feature of the dataset (windspeed_100m)

# In[6]:


dset = f['windspeed_100m']


# In[7]:


dt = f["datetime"]
dt = pd.DataFrame({"datetime": dt[:]},index=range(0,dt.shape[0]))
dt['datetime'] = dt['datetime'].apply(dateutil.parser.parse)


# ### Creating an index to see dataset from just the year 2007

# In[8]:


# Index for 2007
twoThousandSeven = dt.loc[(dt.datetime >= '2007-01-01') & (dt.datetime < '2008-01-01')].index
twoThousandSeven


# ### Selecting a location
# **In order to traverse the data by location, we need to convert location latitude and longitude into X/Y indices** <br>**Using https://www.findlatitudeandlongitude.com/ I chose Charlotte and Raleigh NC as example locations**
# <br>**The function "indicesForCoord" does just that**

# In[9]:


# This function finds the nearest x/y indices for a given lat/lon.
# Rather than fetching the entire coordinates database, this
# uses the Proj4 library to find a nearby point and then converts to x/y indices
def indicesForCoord(f, lat_index, lon_index):
    dset_coords = f['coordinates']
    projstring = """+proj=lcc +lat_1=30 +lat_2=60 
                    +lat_0=38.47240422490422 +lon_0=-96.0 
                    +x_0=0 +y_0=0 +ellps=sphere 
                    +units=m +no_defs """
    projectLcc = Proj(projstring)
    origin_ll = reversed(dset_coords[0][0])  # Grab origin directly from database
    origin = projectLcc(*origin_ll)
    
    coords = (lon_index,lat_index)
    coords = projectLcc(*coords)
    delta = np.subtract(coords, origin)
    ij = [int(round(x/2000)) for x in delta]
    return tuple(reversed(ij))

CharlotteNC = (35.34, -80.83)
CharlotteNC_idx = indicesForCoord( f, CharlotteNC[0], CharlotteNC[1] )

print("y,x indices for Charlotte NC: \t\t {}".format(CharlotteNC_idx))
print("Coordinates of Charlotte NC: \t {}".format(CharlotteNC))
print("Coordinates of nearest point: \t {}".format(f["coordinates"][CharlotteNC_idx[0]][CharlotteNC_idx[1]]))


# In[10]:


# The 2007 timeseries data for a point in Charlotte NC
get_ipython().run_line_magic('time', 'tseriesSeven = dset[min(twoThousandSeven):max(twoThousandSeven)+1, CharlotteNC_idx[0], CharlotteNC_idx[1]]')


# ## Windspeed at 100m/s in Charlotte NC - over the year 2007
# ### We squeeze the data down to just 1 year and 1 location, so analysis and modeling runs faster

# In[11]:


plt.plot(dt.iloc[twoThousandSeven,].datetime, tseriesSeven)
plt.ylabel("Windspeed at 100m (m/s)")
plt.title("Charlotte NC Windspeed in 2007")


# In[12]:


# The timeseries data for all 7 years for a point in Raleigh NC
RaleighNC_idx = indicesForCoord(f, 35.78, -78.64)
get_ipython().run_line_magic('time', 'tseries = dset[:,RaleighNC_idx[0],RaleighNC_idx[1]]')


# ## Windspeed at 100m/s in Raleigh NC - over the full dataset of time
# ### We squeeze the data down to just 1 location but keep all 7 years (2007 to 2013)

# In[13]:


plt.plot(dt.datetime, tseries)
plt.ylabel("Windspeed at 100m (m/s)")
plt.title("Raliegh NC Windspeed 2007-2013")


# ## Saving as a DataFrame
# ### One way we could analyze the data without it being overwhelmingly large is feature reduction:
# ### Here I've reduced down to 1 feature of each type (Temperature, Windspeed, Pressure and Precipitation Rate)

# In[14]:


# save 4 features in a dataframe for all dates of Charlotte NC
get_ipython().run_line_magic('time', 'df = pd.DataFrame({"temperature_100m": f[\'temperature_100m\'][:, CharlotteNC_idx[0], CharlotteNC_idx[1]],                          "windspeed_100m": f[\'windspeed_100m\'][:, CharlotteNC_idx[0], CharlotteNC_idx[1]],                          "pressure_100m": f[\'pressure_100m\'][:, CharlotteNC_idx[0], CharlotteNC_idx[1]],                          "precipitationrate_0m": f[\'precipitationrate_0m\'][:, CharlotteNC_idx[0], CharlotteNC_idx[1]]},                          index=map(dateutil.parser.parse, f["datetime"][:]))')
df.index.name = 'datetime'
df.head()


# ### Example scatter plot of the DataFrame (Charlotte NC): 
# ### Analyzing relationship between Precipitation Rate and Temperature

# In[15]:


df.plot(x='temperature_100m', y='precipitationrate_0m', kind='scatter')


# ### Example bar chart of the DataFrame (Charlotte NC): 
# ### Analyzing the precipitation rate changing over time

# In[16]:


df.plot(y='precipitationrate_0m', use_index=True)


# In[17]:


df.to_csv('CharlotteNC_Wind.csv')


# ### Import classes for Machine Learning

# In[64]:


from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model


# In[19]:


df.head()


# In[74]:


# Decision Tree Regression -> windspeed_100m
X = df.iloc[:,[0,2,3]]
Y = df["windspeed_100m"]


# In[75]:


# splitting into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


# In[76]:


treeRegressionModel = DecisionTreeRegressor(random_state=0)
treeRegressionModel.fit(X_train, Y_train)
treeRegressionPredictions = treeRegressionModel.predict(X_test)

print(treeRegressionPredictions)


# In[77]:


from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.tree import plot_tree, export_graphviz
import matplotlib.pyplot as plt


# In[78]:


DTRScore = mean_absolute_error(Y_test, treeRegressionPredictions)
print("mean_absolute_error:", DTRScore)

DTRScore = r2_score(Y_test, treeRegressionPredictions)
print("r2_score:", DTRScore)

DTRScore = mean_squared_error(Y_test, treeRegressionPredictions)
print("mean_squared_error:", DTRScore)


# In[57]:


fig,ax = plt.subplots(figsize=(12,8))
plot_tree(treeRegressionModel,label="all", max_depth= 3, ax=ax,fontsize=8, filled=True, rounded=True)
plt.tight_layout()
plt.show()


# In[70]:


def plot_targets_against_preds(y, t_test):

    plt.plot(t_test, y, '.')


    plt.plot([-25,25], [-25, 25], 'r--')

    plt.xlim([-25, 25])
    plt.ylim([-25, 25])

    plt.xlabel("target")
    plt.ylabel("predicted")   


# In[71]:


model = LinearRegression()
model.fit(X_train, Y_train)

pred = model.predict(X_test)

score = model.score(X_test, Y_test)
print("The score: ", score)

MAE = mean_absolute_error(Y_test, pred)
print('MAE: %.3f' % MAE) 

r2 = r2_score(Y_test, pred)
print('r2: %.3f' % r2) 

MSE = mean_squared_error(Y_test, pred)
print('MSE: %.3f' % MSE) 

plot_targets_against_preds(pred, Y_test)
plt.title("LinearRegression")


# In[81]:


model = linear_model.ElasticNet(alpha=0.05, l1_ratio=1)
model.fit(X_train, Y_train)

pred = model.predict(X_test)

score = r.score(X_test, Y_test)
print("The score: ", score)

MAE = mean_absolute_error(Y_test, pred)
print('MAE: %.3f' % MAE) 

r2 = r2_score(Y_test, pred)
print('r2: %.3f' % r2) 

MSE = mean_squared_error(Y_test, pred)
print('MSE: %.3f' % MSE) 

plot_targets_against_preds(pred, Y_test)
plt.title("Elastic")


# In[83]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=3, random_state=0)

regr.fit(X_train, Y_train)

pred = regr.predict(X_test)

score = regr.score(X_test, Y_test)
print("The score: ", score)

MAE = mean_absolute_error(Y_test, pred)
print('MAE: %.3f' % MAE) 

r2 = r2_score(Y_test, pred)
print('r2: %.3f' % r2) 

mse = mean_squared_error(Y_test, pred)
print('MSE: %.3f' % mse) 

plot_targets_against_preds(pred, Y_test)
plt.title("RandomForestRegressor")


# In[38]:


# Linear regression windspeed vs pressure
A = df.iloc[:, 1].values.reshape(-1, 1)
B = df.iloc[:, 2].values.reshape(-1, 1)

linearRegressionModel = LinearRegression()
linearRegressionModel.fit(A,B)
BPrediction = linearRegressionModel.predict(A)
print(BPrediction)
plt.scatter(A,B)
plt.plot(A, BPrediction, color='red')
plt.show()


# In[39]:


# split above data and test it with new data/dates?
V = np.array(df["windspeed_100m"]).reshape(-1,1)
W = np.array(df["temperature_100m"]).reshape(-1,1)

V_train, V_test, W_train, W_test = train_test_split(V,W, test_size = 0.2)

LRModel = LinearRegression()
LRModel.fit(V_train, W_train)
print(LRModel.score(V_test, W_test))


# In[40]:


# predictions with the above
WPrediction = LRModel.predict(V_test)

plt.scatter(V_test, W_test)
plt.plot(V_test, WPrediction, color='red')


# In[ ]:




