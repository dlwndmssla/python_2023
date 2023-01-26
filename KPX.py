import numpy
import pandas as pd
data = pd.read_csv("KPX_data.csv")
data = pd.DataFrame(data)

print(data.head())
print(data.tail())

print(data.isnull().sum())


data1 = data.iloc[1,1:]
data2 = data.iloc[2,1:]
data_24 = pd.concat([data1,data2] , axis= 0 )

print(data_24.info())
print(data_24.shape)
print(data_24.tail())

for i in range(2, 1826):
    data2 = data.iloc[i,1:]
    data_25 = pd.concat([data1,data2] , axis= 0 )
    
print(data_25.info())
print(data_25.tail)
