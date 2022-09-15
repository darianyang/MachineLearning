"""
Lab 1: EDA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import seaborn as sns
#import sklearn

# Load data, prepare for analysis, process if necessary
# Analyze types of data

# load data
data = pd.read_csv("lab1_dataset.csv")
# analyze data types
#print(data.info())
#print(data.columns)

# detect columns with nan vals
# for col in data.columns:
#     if data[col].isnull().values.any():
#         print(col)

#print(data["MW2"])

# TODO:
# Find and process missing and erroneous features
#print(data.isnull().values.any())
# Find outliers
# Find highly correlated variables (if any).
# Find if the target variable is correlated with any features.
# Use PCA to plot data in 2D and color code by the target property. Do you see any patterns?


#plt.plot(data["experimental_proprty"])
plt.plot(data["MW2"])

# Need to find highly correlated variables
# I should go through each column and plot it against exp prop
# go through linear regression and find best R^2 values
# should do this for the target exp prop and for all others? (any correlated variables)
#plt.scatter(data["experimental_proprty"], data["nF"])
plt.show()