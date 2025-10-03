import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import dask, time
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

df = pd.read_csv('./datasets/PJME_hourly.csv')

df['Datetime'] = pd.to_datetime(df['Datetime'])
# Преобразование datetime в UNIX timestamp (число с плавающей запятой)
df['Datetime'] = df['Datetime'].astype('int64') // 10**9 #Преобразование в секунды
# целевая переменная называется 'PJME_MW'
x = df.drop('PJME_MW', axis=1)
y = df['PJME_MW']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# LightGBM
lgb_model = lgb.LGBMRegressor(learning_rate=0.01,force_row_wise="true")
lgb_model.fit(x_train, y_train)
y_pred_lgb = lgb_model.predict(x_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)

# CatBoost
catboost_model = CatBoostRegressor(verbose=0)
catboost_model.fit(x_train, y_train, verbose=100)
y_pred_catboost = catboost_model.predict(x_test)
mse_catboost = mean_squared_error(y_test, y_pred_catboost)

score = 3083.8952858982234

print(f'MSE XGBoost: {score:.4f}')
print(f'MSE LightGBM: {np.sqrt(mse_lgb):.4f}')
print(f'MSE CatBoost: {np.sqrt(mse_catboost):.4f}')