import dask, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection  import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

# для каждого дня в датасете есть возможность проведения анализа по
# разным критериям: # - по дню недели Monday = 0 . . sunday = 6
# - по часам.... # что очень характерно для энергопотребления.
def create_features(df):
    df = df.copy()
    df['Часы'] = df.index.hour
    df['Дни_недели'] = df.index.dayofweek
    df['Кварталы'] = df.index.quarter
    df['Месяцы'] = df.index.month
    df['Годы'] = df.index.year
    df['Дни_в_году'] = df.index.dayofyear
    df['Дни_в_месяце'] = df.index.day
    df['Недели_в_году'] = df.index.isocalendar().week
    return df

data = pd.read_csv('./datasets/PJME_hourly.csv')

# Преобразование столбца с датами в формат datetime
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)

# Создание DateFrame с дополнительными данными
data = create_features(data)
data = data.reset_index()

# Предполагаем, что последний столбец - целевой
X = data.drop(columns=['Datetime', 'PJME_MW']).values
y = data['PJME_MW'].values
# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_scaled, y_scaled, range(X_scaled.shape[0]), test_size=0.2, random_state=42)

# LightGBM
lgb_model = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              n_estimators=1000, learning_rate=0.01,
                              force_col_wise="true", verbose=100)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)

# CatBoost
catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1,
depth=6, loss_function='RMSE', verbose=100)
catboost_model.fit(X_train, y_train)
y_pred_catboost = catboost_model.predict(X_test)
mse_catboost = mean_squared_error(y_test, y_pred_catboost)

print(f"MSE LightGBM: { np.sqrt(mse_lgb):.4f}")
print(f"MSE CatBoost: { np.sqrt(mse_catboost):.4f}")