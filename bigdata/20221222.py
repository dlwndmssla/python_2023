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
    '나이' : [20,23,49,38,32,29,25,30,32,26],
    '성별' : ['남','여','남','여','여','여','여','여','남', np.nan]
}

data = pd.DataFrame(data)

print(data)

print(type(data))

print(data['나이'])

data['나이'].sum()

print(data['나이'].sum())