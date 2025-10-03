import dask, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection  import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

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

model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                         n_estimators=1000,
                         early_stopping_rounds=50,
                         objective='reg:squarederror',
                         max_depth=3,
                         learning_rate=0.01)

# Обучение модели с отслеживанием потерь
eval_set = [(X_train, y_train), (X_test, y_test)]

# код time
print('Lets GO!')
start = time.time()
s = time.ctime(start)

bst=model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# код time
end = time.time()
e = time.ctime(end)
print(f'all done!\nstarted: {s}\nfinished: {e}\nsec: {end-start}')

results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

y_pred_train = bst.predict(X_train)
y_pred_test = bst.predict(X_test)

len_train=len(results['validation_0']['rmse'])
len_test=len(results['validation_1']['rmse'])
print(f"Train RMSE: { results['validation_0']['rmse'][len_train-1]:.4f}")
print(f"Test RMSE: { results['validation_1']['rmse'][len_test-1]:.4f}")
