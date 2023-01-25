import pandas as pd
data = pd.read_csv('mtcars.csv')
print(data.head())

shape = data.shape
print(shape)
data_type = type(data)
print(data_type)

print(data.columns)

print(data.describe())

print(data['am'].unique())

print(data['gear'].unique())

info = data.info()
print(info)

cor = data.corr()
print(cor)

X = data.drop(columns = 'mpg')
Y = data['mpg']

print(X.head())

X= X.iloc[:,1:]
print(X)

print(X.isnull().sum())

x_cyl_mean = X['cyl'].mean()

X['cyl'] = X['cyl'].fillna(x_cyl_mean)

x_qsec_median = X['qsec'].quantile(0.5)
X['qsec'] = X['qsec'].fillna(x_qsec_median)

print(x_qsec_median)

print(X.isnull().sum())

