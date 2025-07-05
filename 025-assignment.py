cd your-repo
python your_script.py

#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>2.5. Predicting Apartment Prices in Mexico City 🇲🇽</strong></font>

# In[1]:


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# In this assignment, you'll decide which libraries you need to complete the tasks. You can import them in the cell below. 👇

# In[2]:


# Import libraries here
import warnings
from glob import glob

import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
import plotly.express as px

warnings.simplefilter(action="ignore", category=FutureWarning)


# # Prepare Data

# ## Import

# **Task 2.5.1**
# 
# <div class="alert alert-block alert-info">
# <b>Tip:</b> Don't try to satisfy all the criteria in the first version of your <code>wrangle</code> function. Instead, work iteratively. Start with the first criteria, test it out with one of the Mexico CSV files in the <code>data/</code> directory, and submit it to the grader for feedback. Then add the next criteria.</div>

# In[3]:


# Build your `wrangle` function
def wrangle (filepath):
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Mexico City", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 100_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    #Split lat-lon column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand = True).astype(float)
    df.drop(columns = "lat-lon", inplace = True)

    #Split lat-lon column
    df["borough"] = df["place_with_parent_names"].str.split("|", expand = True)[1]
    df.drop(columns = "place_with_parent_names", inplace = True)

    #Drop columns with more than 50% null values
    df.drop(columns = ["surface_total_in_m2", "price_usd_per_m2", "rooms",
                      "floor", "expenses"], inplace = True)

    #Drop low- and high- cardinality categorical variables
    df.drop(columns= ["operation", "property_type", "currency", "properati_url"], inplace=True)

    # Drop leaky columns
    df.drop(columns=[
        'price',
        'price_aprox_local_currency',
        'price_per_m2'
    ],inplace = True)
    
    return df


# In[4]:


# Use this cell to test your wrangle function on the file `mexico-city-real-estate-1.csv`
frame1 = wrangle("data/mexico-city-real-estate-1.csv")
print(frame1.info())
frame1.head()


# In[5]:


frame1.isnull().sum()/len(frame1)


# In[6]:


frame1.select_dtypes("object").nunique()


# In[7]:


sorted(frame1.columns)


# In[8]:


# Check for multicollinearlity
corr = frame1.select_dtypes("number").drop(columns = "price_aprox_usd").corr()
sns.heatmap(corr)


# **Task 2.5.2** 

# In[9]:


# Extract data files
files = glob("data/mexico-city-real-estate-*.csv")
files.sort()
files


# **Task 2.5.3**

# In[10]:


# Create datafromes
frames = [wrangle(file) for file in files]
frames[1]


# In[11]:


# Combine the dataframes in frames into a single dataframe
df = pd.concat(frames, ignore_index= True)
print(df.info())
df.head()


# ## Explore

# <div class="alert alert-info" role="alert">
#   <strong>Slight Code Change</strong>
# 
# In the following task, you'll notice a small change in how plots are created compared to what you saw in the lessons.
# While the lessons use the global matplotlib method like <code>plt.plot(...)</code>, in this task, you are expected to use the object-oriented (OOP) API instead.
# This means creating your plots using <code>fig, ax = plt.subplots()</code> and then calling plotting methods on the <code>ax</code> object, such as <code>ax.plot(...)</code>, <code>ax.hist(...)</code>, or <code>ax.scatter(...)</code>.
# 
# If you're using pandas’ or seaborn’s built-in plotting methods (like <code>df.plot()</code> or <code>sns.lineplot()</code>), make sure to pass the <code>ax=ax</code> argument so that the plot is rendered on the correct axes.
# 
# This approach is considered best practice and will be used consistently across all graded tasks that involve matplotlib.
# </div>
# 

# **Task 2.5.4**

# In[12]:


fig, ax = plt.subplots() 

# Plot the histogram on the axes object
ax.hist(df["price_aprox_usd"]) 

# Label axes using the axes 
ax.set_xlabel("Price [$]")
ax.set_ylabel("Count")


# Add title 
ax.set_title("Distribution of Apartment Prices")


# **Task 2.5.5**

# In[13]:


fig, ax = plt.subplots() 

# Create the scatter plot on the axes object
ax.scatter(x = df["surface_covered_in_m2"], y = df["price_aprox_usd"]) 

# Label axes 
ax.set_xlabel("Area [sq meters]")
ax.set_ylabel("Price [USD]")

#  Add title 
ax.set_title("Mexico City: Price vs. Area")


# Do you see a relationship between price and area in the data? How is this similar to or different from the Buenos Aires dataset?<span style='color: transparent; font-size:1%'>WQU WorldQuant University Applied Data Science Lab QQQQ</span>

# **Task 2.5.6** **(UNGRADED)** Create a Mapbox scatter plot that shows the location of the apartments in your dataset and represent their price using color. 
# 
# What areas of the city seem to have higher real estate prices?

# In[14]:


# Plot Mapbox location and price

fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    width=600,
    height=600,
    color= "price_aprox_usd",
    hover_data = ["price_aprox_usd"],
)

fig.update_layout(mapbox_style="open-street-map")

fig.show()


# ## Split

# **Task 2.5.7**

# In[15]:


# Split data into feature matrix `X_train` and target vector `y_train`.
target = "price_aprox_usd"
features = ["surface_covered_in_m2", "lat", "lon", "borough"]
y_train = df[target]
X_train = df[features]


# In[16]:


X_train.shape


# In[17]:


y_train.shape


# # Build Model

# ## Baseline

# **Task 2.5.8**

# In[18]:


y_mean = y_train.mean()
y_pred_baseline = [y_mean]*len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)


# ## Iterate

# **Task 2.5.9**

# In[19]:


# Build Model
model = make_pipeline(
    OneHotEncoder(use_cat_names= True),
    SimpleImputer(),
    Ridge()
)

# Fit model
model.fit(X_train, y_train)


# ## Evaluate

# **Task 2.5.10**

# <div class="alert alert-block alert-info">
# <b>Tip:</b> Make sure the <code>X_train</code> you used to train your model has the same column order as <code>X_test</code>. Otherwise, it may hurt your model's performance.
# </div>

# In[20]:


X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()


# **Task 2.5.11**

# In[21]:


y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()


# # Communicate Results

# **Task 2.5.12** 

# In[22]:


# Extract the coefficients
coefficients = model.named_steps["ridge"].coef_
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index=features)
feat_imp = pd.Series(feat_imp.values, index=feat_imp.index).reindex(feat_imp.abs().sort_values().index)
feat_imp


# **Task 2.5.13**

# In[23]:


# Plot the 10 most influential coefficients
fig, ax = plt.subplots()

# Create the horizontal bar plot on the axes object
feat_imp.sort_values(key = abs).tail(10).plot(kind = "barh", ax = ax)

#feat_imp...plot(..., ax=ax)

#  Label axes 
ax.set_xlabel("Importance [USD]") 
ax.set_ylabel("Feature")

# Add title 
ax.set_title("Feature Importances for Apartment Price")


# ---
# Copyright 2024 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
