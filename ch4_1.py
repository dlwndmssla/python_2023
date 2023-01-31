import numpy as np
import pandas as pd


########1번문제
data = pd.read_csv("boston.csv")

print(data)

print(data.shape)

print(data.info())

x = data.iloc[:,13]
print(x)

y = data.sort_values(by = 'MEDV', ascending = True)

final = y.iloc[:10,13]

print(final.info())

print(final)

#####2번문제
print(data.isnull().sum())

data2 = data["RM"]

print(data2)

print("################")
print(data2.shape)
print(data2.info())
print(data2.describe())
rm_mean = 6.285102

data_fill = data2.fillna(rm_mean)

print(data_fill)

print(data_fill.describe())

print(data_fill.isnull().sum())

fill_std = data_fill.std()

print(fill_std )

data_delete = data2.dropna()

delete_std = data_delete.std()

print(delete_std)

print(abs(fill_std - delete_std))

#########3번

data_3 = data["ZN"]
print(data_3.describe())

data_3_std = data_3.std()
data_3_mean = data_3.mean()


max = 1.5*data_3_std+data_3_mean
min = data_3_mean - 1.5*data_3_std

max_sum = data_3[ data_3[:] > max].sum()

min_sum = data_3[ data_3[:]  < min].sum()

data3 = max_sum + min_sum

print(data_3[ data_3[:] > max])



print(max_sum)
print(min_sum)

print(data3)

#########4번 
