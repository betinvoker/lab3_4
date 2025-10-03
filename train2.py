import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.metrics import mean_squared_error

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

df = pd.read_csv('./datasets/PJME_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
# Создание DateFrame с дополнительными данными
df = create_features(df)
df = df.reset_index()
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

# Создание регрессора xgboost
reg2 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1600,
                       early_stopping_rounds=45,
                       objective='reg:squarederror',
                       max_depth=3,
                       learning_rate=0.01)

print('Lets GO!')
start = time.time()
s = time.ctime(start)

x_test = x_test.drop('Прогноз', axis=1)

reg2.fit(x_train, y_train, eval_set=[
    (x_train, y_train), (x_test, y_test)],
    verbose=100
)

end = time.time()
e = time.ctime(end)
print(f'all done!\nstarted: {s}\nfinished: {e}\nsec: {end-start}')

predict2=reg.predict(x_test)
x_test['Прогноз'] = predict2

# Вычисление среднеквадратичной ошибки на тестовом наборе
score2 = np.sqrt(mean_squared_error(y_test, predict2))
print(f'Значение среднеквадратичной ошибки на тестовом наборе: {score2}')

copy = pd.read_csv('./datasets/PJME_hourly.csv')

# Добавление столбца прогноз к набору данных
copy = copy.merge(x_test[['Прогноз']], how='left', left_index=True,
right_index=True)
copy.head()

sns.set_palette('terrain')
ax = copy[['PJME_MW']].plot(figsize=(15, 5))
copy['Прогноз'].plot(ax=ax, style='.')
plt.legend(['Реальные данные', 'Прогноз'])
ax.set_title('Реальные данные потребления и прогноз')
plt.show()

# Задание индекса в DateFrame в виде столбца Дата
copy.set_index("Datetime", inplace=True)
copy.index = pd.to_datetime(copy.index)
# Визуальное представление реальных данных и прогноза за неделю
ax = copy.loc[(copy.index > '04-01-2018') & (copy.index < '04-08-2018')]['PJME_MW']
ax.plot(figsize=(15, 5), title='Потребление энергии за неделю')
copy.loc[(copy.index > '04-01-2018') & (copy.index < '04-08-2018')]['Прогноз'].plot(style='.')
plt.legend(['Реальные данные', 'Прогноз'])
plt.show()

current_palette = sns.color_palette('deep')
fi = pd.DataFrame(data=reg.feature_importances_,
index=reg.feature_names_in_,
columns=['Важность'])
# Сортировка и визуальное представление данных по важности для прогнозирования
fi.sort_values('Важность').plot(kind='barh',figsize=(5, 5),
title='Важность данных',color=sns.color_palette(current_palette)[2])
plt.show()