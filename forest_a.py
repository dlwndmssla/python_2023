import pandas as pd 

data = pd.read_excel("forest.xlsx")

print(data.head())
print(data.tail())

shape = data.shape
print(shape)
data_type = type(data)
print(data_type)

print(data.columns)

print(data.describe())

print("결측치")

print(data.isnull().sum())

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

print('interpolate')

data_df = pd.DataFrame(data = data)

data_df = data_df.iloc[:,1:]

data_fl = data_df.astype('float')

print(data_fl.info())

data_final = data_fl.interpolate(method='linear')

print(data_final.describe())

print("결측치")

print(data_final.isnull().sum())

print("finished!!!!!!!!!!!!!!!!!!!")

print(data_final.head())

data_final_1 = pd.DataFrame(data_final)

#data_final_1.to_excel('C:/Users/user/first/data_final.xlsx')