#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
pd.__version__ # Version of Pandas


# In[9]:


spreadsheet = pd.read_csv('/Users/MyWor/Downloads/raw.githubusercontent.com_alexeygrigorev_datasets_master_housing.csv')


# In[10]:


print(spreadsheet) # Number of columns in the dataset


# In[11]:


np.where(pd.isnull(spreadsheet)) 


# In[21]:


empty_entries = spreadsheet.isna() # Are there any columns with missing data?
print(empty_entries)


# In[22]:


len(spreadsheet.columns) - len(spreadsheet.dropna(axis=1,how='all').columns)


# In[13]:


spreadsheet.nunique() # Number of unique values in the 'ocean_proximity' column


# In[23]:


print(spreadsheet.describe()) 


# In[29]:


spreadsheet['total_bedrooms'] = spreadsheet.fillna(spreadsheet['total_bedrooms'].mean())


# In[ ]:


# Additionals 

# Calculate the average of total_bedrooms column in the dataset.
# Use the fillna method to fill the missing values in total_bedrooms with the mean value from the previous step.
# Now, calculate the average of total_bedrooms again.
# Has it changed?

# Select all the options located on islands.
# Select only columns housing_median_age, total_rooms, total_bedrooms.
# Get the underlying NumPy array. Let's call it X.
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# Compute the inverse of XTX.
# Create an array y with values [950, 1300, 800, 1000, 1300].
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# What's the value of the last element of w?

