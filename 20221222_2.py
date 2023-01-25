from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import numpy as np 
array1 = np.zeros((2,3))
print(array1)

array2 = np.ones((2,3))
print(array2)

array3 = np.full((2,3),10)
print(array3)

array4 = np.array(range(20))
print(array4)

array5 = np.array(range(20)).reshape(4,5)
print(array5)

import pandas as pd
import numpy as np

data = {
    '나이' : [20,23,49,38,32,29,25,30,32,np.nan],
    '성별' : ['남','여','남','여','여','여','여','여','남','남']
}

data = pd.DataFrame(data)

print(data)

print(type(data))

print(data['나이'])

data['나이'].sum()

print(data['나이'].sum())

print(max(data['나이']))

print(data['나이'].quantile(0.25))

data1 = data["나이"]/7
print(data1)

print(round(data1))
print(round(data1, 3))
print(round(data1, -1))

print(data.shape)
print(data['나이'].shape)

count = data['성별'].value_counts()

print(count)

uni = data['성별'].unique()
print(uni)

str = data.info()

describe = data.describe()


print('-------------------')
print(describe)

trans = data.transpose()

print(trans)

print(trans.T)
print('11111111111')
print(data.loc[3])
print('22222222')
print(data.loc[3:7])
print('3333333')
print(data.loc[3:7,'성별'])

print(data['성별']=='남')

print(data.loc[data['성별']=='남'])

print(data.iloc[3])
print(data.iloc[4:8])

from xgboost import XGBClassifier