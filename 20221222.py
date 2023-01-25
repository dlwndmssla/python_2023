import pandas as pd  # noqa
list_var = (['홍',21,'남'],['이',23,'여'],['임',36,'남'],['권',33,'여'])
dataframe_var = pd.DataFrame(list_var)
print(dataframe_var)
print(type(dataframe_var))

dataframe_var.index = ['1번','2번','3번','4번']
print(dataframe_var)

dataframe_var.columns = ['이름','나이','성별']

print(dataframe_var)

print(dataframe_var['나이'])

print(dataframe_var[['나이','성별']])



