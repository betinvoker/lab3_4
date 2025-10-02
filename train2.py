import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time
from sklearn.model_selection  import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./datasets/PJME_hourly.csv')

df['Datetime'] = pd.to_datetime(df['Datetime'])
# Преобразование datetime в UNIX timestamp (число с плавающей запятой)
df['Datetime'] = df['Datetime'].astype('int64') // 10**9 #Преобразование в секунды
# целевая переменная называется 'PJME_MW'
x = df.drop('PJME_MW', axis=1)
y = df['PJME_MW']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Создание регрессора xgboost
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1200,
                       early_stopping_rounds=50,
                       objective='reg:squarederror',
                       max_depth=3,
                       learning_rate=0.01)

print('Lets GO!')
start = time.time()
s = time.ctime(start)

reg.fit(x_train, y_train, eval_set=[
    (x_train, y_train), (x_test, y_test)],
    verbose=100
)

end = time.time()
e = time.ctime(end)
print(f'all done!\nstarted: {s}\nfinished: {e}\nsec: {end-start}')

predict=reg.predict(x_test)
x_test['Прогноз'] = predict

# Вычисление среднеквадратичной ошибки на тестовом наборе
score = np.sqrt(mean_squared_error(y_test, predict))
print(f'Значение среднеквадратичной ошибки на тестовом наборе: {score}')

