import time
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection  import train_test_split

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

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'eval_metric': 'rmse'
}

evals = [(dtrain, 'train'), (dtest, 'test')]
evals_result = {}

# код time
print('Lets GO!')
start = time.time()
s = time.ctime(start)


bst = xgb.train(params=params, dtrain=dtrain,
                evals=evals, evals_result=evals_result,
                num_boost_round=100, verbose_eval=True
)

# код time
end = time.time()
e = time.ctime(end)
print(f'all done!\nstarted: {s}\nfinished: {e}\nsec: {end-start}')

len_train=len(evals_result['train']['rmse'])
len_test=len(evals_result['test']['rmse'])
print(f"Train RMSE: { evals_result['train']['rmse'][len_train-1]:.4f}")
print(f"Test RMSE: { evals_result['test']['rmse'][len_test-1]:.4f}")

# График потерь
plt.figure(figsize=(12, 6))
plt.plot(evals_result['train']['rmse'], label='Training Loss')
plt.plot(evals_result['test']['rmse'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.legend()
plt.show()

y_pred_train = bst.predict(dtrain)
y_pred_test = bst.predict(dtest)

# Обратное преобразование данных
y_train_actual = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
y_test_actual = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
y_pred_train_actual = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
y_pred_test_actual = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

data_test = data.iloc[indices_test].reset_index(drop=True)
data_test['Predicted'] = y_pred_test_actual

time_features = ['Часы', 'Дни_недели', 'Кварталы', 'Месяцы', 'Годы', 
                 'Дни_в_году', 'Дни_в_месяце', 'Недели_в_году']

for feature in time_features:
    plt.figure(figsize=(12, 6))
    plt.plot(data_test.groupby(time_features[0])['PJME_MW'].mean(), 
            label='Real data', color='blue')

    plt.plot(data_test.groupby(feature)['Predicted'].mean(),
    label='Predictions', color='red')
    plt.title(f'Real vs Predicted values by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Value')
    plt.legend()
    plt.show()