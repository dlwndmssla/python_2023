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

def interpolated(df):
    temp = []
    for i in df['ID'].unique():
        temp.append(df[df['ID'] == i].interpolate())
    new_df = pd.concat(temp, axis= 0)
    return new_df
    
print("def")

new_df = interpolated(data)

#Scipy 와 Matplotlib 라이브러리 사용 
from scipy import interpolate
import matplotlib.pyplot as plt 

#그림2의 데이터 정의 ( x가 개월물, y가 스왑포인트)
x = [30, 60, 90, 180, 360]
y = [-130, -240, -355, -940, -2370]

#SciPy의 interploate 모듈의 interp1d 함수를 사용해서 선형보간 함수를 생성 
linear_func = interpolate.interp1d(x, y, kind='linear')

#선형보간 함수에 x를 넣어서 선형보간된 값인 y_linear를 구함
y_linear = linear_func(x)

#maplotlib 를 이용해서 x 값, y 값 으로 점 "o" 를 찍어주고
#x 값 y_linear 값 으로 "-" 표시로 점 사이를 채워줌
plt.plot(x,y,"o", x, y_linear, "-")

#maplotlib 를 이용해서 그래프를 보여줌
plt.show()