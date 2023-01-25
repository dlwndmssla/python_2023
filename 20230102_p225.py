import pandas as pd 
data = pd.read_csv('mtcars.csv')

print(data.head()) #데이터 변수.head(위에 몇줄인지, 기본값 5줄)
print(data.tail()) #데이터 변수.tail(밑에서 몇줄인지, 기본값 5줄)

print(data.shape) #데이터 변수의 모양을 알려줌(행,열) -> 뒤에 괄호 없음
print(type(data)) #변수의 데이터 타입을 알려줌 ex. dataframe -> 애 혼자 형태 이상

print(data.columns) #데이터의 열들을 알려줌,쉽게말해 종속변수, 독립변수

print(data.describe()) #기초통계량구하기 전체적인 분위값을 알려준다.

print(data['hp'].describe()) #특정 변수에 대해  기초통계량을 구할때는 대괄호를 사용한다.

print(data['am'].unique()) #am칼럼에서 데이터값 확인

print(data.info()) #요약정보, 범주형변수, 연속형 변수 예상

print(data.corr()) #상관관계를 알려줌

##여기부터는 데이터 분석이라고 볼수 있지 않을까(?)

X = data.drop(columns = 'mpg') #독립변수
Y = data['mpg'] #종속변수

print(X.head())

print(Y.head())

##데이터 관찰, 가공, 전처리 ----독립변수를 물고 뜯고 맛보기이ㅣㅣ
##빅분기는 시각화를 안다룬다고 한다. 나는 시각화가 좋은데 왜,,,? 시각화 예쁘자나,,,
## 1. 데이터 준비 2. 데이터 관찰 가공 3. 데이터 분리 4.공부 평가 5. 결과 출력 저장

X = X.iloc[:,1:] #전체행과 2열부터 끝열까지 포함한다.
print(X.head()) #이 책은 왜 변수 이름을 X로 했을까? 진짜 대문자라서 더럽게 불편하다

#결측치 여부 확인하기
print(X.isnull().head())

print(X.isnull().sum()) #사실 윗줄은 별 쓸모없고 이 줄이 중요하다 #cyl 결측값2개 qsec1개다

##결측값은 평균값, 중위값을 바꾸거나 삭제할수 있는데 삭제를 하면 시험때 개판나므로 하지 말자

X_cyl_mean = X['cyl'].mean() #평균값 대치

print(X_cyl_mean) #7.6

X['cyl'] = X['cyl'].fillna(X_cyl_mean)

print(X['cyl'])

X_qsec_median = X['qsec'].median()

X['qsec'] = X['qsec'].fillna(X_qsec_median)

print(X.isnull().sum()) ##결측값 다 채웠다. 하하

#잘못된값을 올바르게 바꾸기 

print(X['gear'].unique())

print(X['gear'].replace('*3','3').replace('*5','3'))

X['gear'] = X['gear'].replace('*3','3').replace('*5','3')

print(X['gear'].unique())


###----------------------------20230103
X_describe = X.describe()
print(X_describe)

X_iqr = X_describe.loc['75%'] - X_describe.loc['25%'] #IQR값 확인하기
print(X_iqr)


print(X_describe.loc['75%'] + 1.5*X_iqr)  ## 각 열의 3사분위수+1.5iqr 확인하기

print(X_describe.loc['max']) #x변수의 최댓값 확인

print(X.loc[X['cyl']> 14] )  ##cyl 열 값이 14보다 초과하는 값 찾기

X.loc[14, 'cyl'] = 14 # 14번째 행이 이상값을 가지므로 14번째 행의 cyl값을 최대 경계값인 14로 교체한다. 

print(X.loc[14,:])

print(X.loc[X['hp'] > 305.25])

X.loc[30,'hp'] = 305.25 #30번째 행이 hp에서 이상값을 가지므로 최대 경계값인 305.25로 교체한다.

print(X.loc[30,:]) 

print(X_describe.loc['25%'] - 1.5*X_iqr)  ## 각 열의 3사분위수+1.5iqr 확인하기

print(X_describe.loc['min'])

###20230104

def outlier(data, column):
    mean = data[column].mean()
    std = data[column].std()
    lowest = mean - (std*1.5)
    highest = mean + (std*1.5)
    
    print('최소경계값 : ',lowest,'최대경계값 :',highest)
    outlier_index = data[column][(data[column] < lowest)|(data[column]> highest)].index
    
    return outlier_index   

print(outlier(X,'qsec'))
print(X.loc[24,'qsec'])

X.loc[24,'qsec']= 42.245
print(X.loc[24,'qsec']) #최대경계값을 넘긴 이상값을 최대경계값으로 교체해 주고 확인하는 과정

print(outlier(X,'carb'))

print(X.loc[29:30,'carb'])

X.loc[29:30,'carb'] = round(5.235299966447778,3)
print(X.loc[29:30,'carb'])

from sklearn.preprocessing import StandardScaler # sklearn패키지의 preprocessing모듈에서 standardscaler함수를 가져오기

temp = X[['qsec']]

scaler = StandardScaler()

print(scaler.fit_transform(temp))

qsec_s_scaler = pd.DataFrame(scaler.fit_transform(temp))

print(qsec_s_scaler.describe())

from sklearn.preprocessing import MinMaxScaler

#셋중에 원하는 전처리 방식을 scaler에 저장한다.
#원하는 열을 새로운 변수에 저장한다.
#scaler에게 변수의 크기변환을 시키고 타입을 DataFrame으로 바꾼다
#상세결과를 확인할때는 describe()를 사용한다.

print(X.info())

print(X.head()) #gear열이 상수값을 가지므로 type을 int64로 바꾸어주자

X['gear'] = X['gear'].astype('int64')

print(X.info())

print(1111111111111)

print(X['am'].unique()) #뒤에 괄호 꼭 붙이기

print(pd.get_dummies(X['am'], drop_first = True))

print(pd.get_dummies(X, drop_first = True)) #시험때 유용한 코드

X = pd.get_dummies(X, drop_first = True)

condition = X['wt'] < X['wt'].mean()

X.loc[condition, 'wt_class'] = 0

X.loc[~condition, 'wt_class'] = 1

print('1111111111111')
print(X['wt_class'])


print(X['wt'].mean())

X = X.drop(columns = 'wt')
print(X.head(3))


###20230105

import sklearn.model_selection
test = dir(sklearn.model_selection )
for name in test:
    if ',' not in name:
        print(name)
    
    
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(X,Y,test_size = 0.3, random_state = 10)
print("xtrain")
print(x_train.head(3))
print("xtest")
print(x_test.head(3))
print("ytrain")
print(y_train.head(3))


print("ytest")
print(y_test.head(3))


##종속변수가 연속형이면 예측모델, 범주형이면 분류모델로 방향을 잡는다

##예측모델이나 분류모델을 수행하는 공통과정
#1.공부시킬 모델이 구현된 함수들을 sklearn 라이브러리를 통해 가져온다
#2.가져온 모델을 호출하여 준비한다
#3.모델에게 학습데이터를 전해서 공부시킨다.
#4.공부가 완료된 모델을 통해서 우리가 예측해야 할 값을 예상한다.

##모델 평가
#1.sklearn.metrics에서 평가함수를 가져온다
#2.평가함수를 호출하여 평가기준에 따른 수치적인 결과 확인

#선형회귀 분석
#linear_model에서 LinearRegression 모델을 가져오기
from sklearn.linear_model import LinearRegression
model = LinearRegression()

print("xt")
print(x_train)
print("yt")
print(y_train.describe())



model.fit(x_train, y_train) #model에서 train을 공부시키기

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)
#학습이 완료된 모델에 x를 전달하여 y를 예측하기

print(model.intercept_)
print(model.coef_)

print(model.score(x_train, y_train)) #선형회귀 모델 결정계수 구하기
print(model.score(x_test, y_test))

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

##20230106-------------

#1.학습데이터의 결정계수 구하기
#2.테스트 데이터의 결정계수 구하기
#3.테스트 데이터의 mse구하기
#4.테스트 데이터의 rmse구하기
#5.테스트 데이터의 mae구하기

print("error______")
print(r2_score(y_train, y_train_pred))
print(r2_score(y_test, y_test_pred))
print(mean_squared_error(y_test,y_test_pred))
print(np.sqrt(mean_squared_error(y_test,y_test_pred)))
print(mean_absolute_error(y_test,y_test_pred))

#랜덤포레스트 회귀
#sklearn,ensemble,randomforestregressor
#model만들기
#모델에 train data 넣기
#ytrain 예측
#ytest 예측

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state = 10)

model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)

print("RandomForestRegressor")
print(r2_score(y_train, y_train_pred))

print(r2_score(y_test, y_test_pred))

print(mean_squared_error(y_test, y_test_pred))

print(np.sqrt(mean_squared_error(y_test, y_test_pred)))
print(mean_absolute_error(y_test,y_test_pred))

#model 개선
model = RandomForestRegressor(n_estimators = 1000, criterion = 'mae', random_state = 10)

model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)


print("RandomForestRegressor model 개선" )
print(r2_score(y_train, y_train_pred))

print(r2_score(y_test, y_test_pred))

print(mean_squared_error(y_test, y_test_pred))

print(np.sqrt(mean_squared_error(y_test, y_test_pred)))
print(mean_absolute_error(y_test,y_test_pred))

#gradientboostingregressor

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor( n_estimators= 1000, random_state = 10)

model.fit(x_train, y_train)

y_train_pred_grad = model.predict(x_train)

y_test_pred_grad = model.predict(x_test)

print("gradient basic")
print(r2_score(y_train,y_train_pred_grad))

print(r2_score(y_test,y_test_pred_grad))

print(mean_squared_error(y_test,y_test_pred_grad))

print(mean_absolute_error(y_test,y_test_pred_grad))

model = GradientBoostingRegressor(criterion = 'mae', n_estimators= 1000, random_state = 10)

model.fit(x_train, y_train)

y_train_pred_grad = model.predict(x_train)

y_test_pred_grad = model.predict(x_test)

print("gradient focus on mae")
print(r2_score(y_train,y_train_pred_grad))

print(r2_score(y_test,y_test_pred_grad))

print(mean_squared_error(y_test,y_test_pred_grad))

print(mean_absolute_error(y_test,y_test_pred_grad))

#XGBRegressor

#from xgboost import XGBRegressor

#model = XGBRegressor(random_state = 10)

#y_train_pred_xgb = model.predict(x_train)

#y_test_pred_xgb = model.predict(x_test)

#print("XGBRegressor")

#print(r2_score(y_train, y_train_pred_xgb))

#print(r2_score(y_test, y_test_pred_xgb))

#print(mean_squared_error(y_test, y_test_pred_xgb))

#print(mean_absolute_error(y_test, y_test_pred_xgb))

####20230110

#분류모델링 수행

#x_train 변수에서 종속변수인 am_new열은 삭제한 후 , 결과는 x_train2에 저장

print(x_train.info())

x_train2 = x_train.drop(columns = 'am_manual')

y_train2 = x_train['am_manual']


x_test2 = x_test.drop(columns = 'am_manual')

y_test2 = x_test['am_manual']



y_train2.astype(int)
y_test2.astype(int)

print(y_train2.head())
print(y_test2.head())


print(x_train2.head())
print(x_test2.head())

#의사결정나무

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(x_train2, y_train2)

y_test2_pred_tree = model.predict(x_test2)

print(y_test2_pred_tree)

#분류모델의 평가지표: roc, 정확도, 정밀도, 재현율

from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score

print("score")
print(roc_auc_score(y_test2, y_test2_pred_tree))
print(accuracy_score(y_test2, y_test2_pred_tree))
print(precision_score(y_test2, y_test2_pred_tree))
print(recall_score(y_test2, y_test2_pred_tree))

#랜덤포레스트 분류

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(x_train2, y_train2)

y_test2_pred_rf = model.predict(x_test2)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

print(roc_auc_score(y_test2,y_test2_pred_rf))
print(accuracy_score(y_test2,y_test2_pred_rf))
print(precision_score(y_test2,y_test2_pred_rf))
print(recall_score(y_test2,y_test2_pred_rf))


from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(x_train2,y_train2)

print(y_train2)

y_test2_pred_lr = model.predict(x_test2)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test2,y_test2_pred_lr))
