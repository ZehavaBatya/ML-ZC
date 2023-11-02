#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Features
# For the rest of the assignment, you'll need to use only these columns:

# Make,
# Model,
# Year,
# Engine HP,
# Engine Cylinders,
# Transmission Type,
# Vehicle Style,
# highway MPG,
# city mpg,
# MSRP
# Data preparation
# Select only the features from above and transform their names using the next line:
# data.columns = data.columns.str.replace(' ', '_').str.lower()
# Fill in the missing values of the selected features with 0.
# Rename MSRP variable to price.


# In[2]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("classification-car-price.csv")


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:


# Select only them and fill in the missing values with 0.


# In[8]:


# Question 1
# What is the most frequent observation (mode) for the column transmission_type?

# AUTOMATIC
# MANUAL
# AUTOMATED_MANUAL
# DIRECT_DRIVE


# In[9]:


df.describe(include=["O"])


# In[10]:


df['Transmission Type'].value_counts()


# In[11]:


# Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.

# What are the two features that have the biggest correlation in this dataset?

# engine_hp and year
# engine_hp and engine_cylinders
# highway_mpg and engine_cylinders
# highway_mpg and city_mpg


# In[14]:


d_num = df.copy()
d_num = df.drop(['Make', 'Model', 'Transmission Type', 'Vehicle Style'], axis=1)
d_num.describe()


# In[13]:


d_num.corr()


# In[ ]:


plt.figure(figsize=(9, 6))
sns.heatmap(d_num.corr(), cmap="summer", annot=True, fmt='.3f')
plt.title('Heatmap showinng correlations between numerical data')
plt.show();


# In[ ]:


d_num.corr().unstack().sort_values().sort_values(ascending = False)


# In[ ]:


# Make price binary
# Now we need to turn the price variable from numeric into a binary format.
# Let's create a variable above_average which is 1 if the price is above its mean value and 0 otherwise.

# Split the data
# Split your data in train/val/test sets with 60%/20%/20% distribution.
# Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.
# Make sure that the target value (above_average) is not in your dataframe.


# In[ ]:


df['price'].mean()

data_class = data.copy()
mean = data_class['price'].mean()

data_class['above_average'] = np.where(data_class['price']>=mean,1,0)

data_class = data_class.drop(['price'], axis=1)

data_class


# In[ ]:


# Question 3
# Calculate the mutual information score between above_average and other categorical variables in our dataset. Use the training set only.
# Round the scores to 2 decimals using round(score, 2).
# Which of these variables has the lowest mutual information score?

# make
# model
# transmission_type
# vehicle_style


# In[ ]:


# Split the data

# Split your data in train/val/test sets, with 60%/20%/20% distribution
# Use Scikit-Learn for that (the train_test_split function) and set the seed to 42
# Make sure that the target value (price) is not in your dataframe


# In[ ]:


# SEED = 42


# df_full_train, df_test = train_test_split(data_class, test_size=0.2, random_state=SEED)
# df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED)

# assert len(data_class) == (len(df_train) + len(df_val) + len(df_test))

# len(df_train), len(df_val), len(df_test)

# df_train = df_train.reset_index(drop=True)
# df_val = df_val.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)

# y_train = df_train.above_average.values
# y_val = df_val.above_average.values
# y_test = df_test.above_average.values


# In[ ]:


# Question 3
# Calculate the mutual information score between above_average and other categorical variables in our dataset. Use the training set only.
# Round the scores to 2 decimals using round(score, 2).
# Which of these variables has the lowest mutual information score?

# make
# model
# transmission_type
# vehicle_style


# In[ ]:


# def mutual_info(series):
#     return mutual_info_score(series, df_train.above_average)


# In[ ]:


# cat = ['make', 'model', 'transmission_type', 'vehicle_style']


# In[ ]:


# df_mi = df_train[cat].apply(calculate_mi)
# df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
# df_mi

# Answer: Transmission Type has the lowest score.


# In[ ]:


# Question 4
# Now let's train a logistic regression.
# Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
# Fit the model on the training dataset.
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
# model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
# Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
# What accuracy did you get?

# 0.60
# 0.72
# 0.84
# 0.95


# In[ ]:


# dv = DictVectorizer(sparse=False)
# train_dict = df_train.to_dict(orient='records')
# X_train = dv.fit_transform(train_dict)


# In[ ]:


# model = LogisticRegression(solver='liblinear', max_iter=1000, C=10, random_state=SEED)
# model.fit(X_train, y_train)


# In[ ]:


# val_dict = df_val.to_dict(orient='records')
# X_val = dv.transform(val_dict)

# y_pred = model.predict(X_val)


# In[ ]:


# accuracy = np.round(accuracy_score(y_val, y_pred),2)
# print(f'Accuracy = {accuracy}')


# In[ ]:


# Question 5
# Let's find the least useful feature using the feature elimination technique.
# Train a model with all these features (using the same parameters as in Q4).
# Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
# For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
# Which of following feature has the smallest difference?

# year
# engine_hp
# transmission_type
# city_mpg
# Note: the difference doesn't have to be positive


# In[ ]:


# features = df_train.columns.to_list()
# features


# In[ ]:


# original_score = accuracy
# scores = pd.DataFrame(columns=['eliminated_feature', 'accuracy', 'difference'])
# for feature in features:
#     subset = features.copy()
#     subset.remove(feature)
    
#     dv = DictVectorizer(sparse=False)
#     train_dict = df_train[subset].to_dict(orient='records')
#     X_train = dv.fit_transform(train_dict)

#     model = LogisticRegression(solver='liblinear', max_iter=1000, C=10, random_state=SEED)
#     model.fit(X_train, y_train)
    
#     val_dict = df_val[subset].to_dict(orient='records')
#     X_val = dv.transform(val_dict)
    
#     y_pred = model.predict(X_val)
#     score = accuracy_score(y_val, y_pred)
    
#     scores.loc[len(scores)] = [feature, score, original_score - score]


# In[16]:


# scores


# In[ ]:


# min_diff = scores.difference.min()
# scores[scores.difference == min_diff]


# In[ ]:


# Question 6
# For this question, we'll see how to use a linear regression model from Scikit-Learn.
# We'll need to use the original column price. Apply the logarithmic transformation to this column.
# Fit the Ridge regression model on the training data with a solver 'sag'. Set the seed to 42.
# This model also has a parameter alpha. Let's try the following values: [0, 0.01, 0.1, 1, 10].
# Round your RMSE scores to 3 decimal digits.
# Which of these alphas leads to the best RMSE on the validation set?

# 0
# 0.01
# 0.1
# 1
# 10
# Note: If there are multiple options, select the smallest alpha.


# In[ ]:


# data['price'] = np.log1p(data['price'])


# In[ ]:


# df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=SEED)
# df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEE


# In[ ]:


# df_train = df_train.reset_index(drop=True)
# df_val = df_val.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)


# In[ ]:


# y_train = df_train.price.values
# y_val = df_val.price.values
# y_test = df_test.price.values


# In[ ]:


# df_train = df_train.drop('price', axis=1)
# df_val = df_val.drop('price', axis=1)
# df_test = df_test.drop('price', axis=1)

# assert 'price' not in df_train.columns
# assert 'price' not in df_val.columns
# assert 'price' not in df_test.columns


# In[ ]:


# y_train.shape, y_val.shape


# In[ ]:


# dv = DictVectorizer(sparse=False)
# train_dict = df_train.to_dict(orient='records')
# X_train = dv.fit_transform(train_dict)

# val_dict = df_val.to_dict(orient='records')
# X_val = dv.transform(val_dict)


# In[ ]:


# scores = {}
# for alpha in [0, 0.01, 0.1, 1, 10]:
#     model = Ridge(alpha=alpha, solver='sag', random_state=SEED)
#     model.fit(X_train, y_train)
    
#     y_pred = model.predict(X_val)
    
#     score = mean_squared_error(y_val, y_pred, squared=False)
#     scores[alpha] = round(score, 3)
#     print(f'alpha = {alpha}:\t RMSE = {score}')


# In[ ]:


# scores


# In[ ]:


# print(f'The smallest `alpha` is {min(scores, key=scores.get)}.')

