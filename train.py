import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./datasets/PJME_hourly.csv')
print(df.head())
print(df.tail())

df.set_index('Datetime', inplace=True)
print(df.head())

current_palette = sns.color_palette('deep')
sns.palplot(current_palette)

df.plot(title='PJME Energy use in MegaWatts',
        figsize=(15, 8), style='.', color=sns.color_palette(current_palette)[4])
plt.xticks(rotation=45)
plt.show()

#Индекс относится к объектному типу, поэтому мы преобразуем его в datetime
df.index = pd.to_datetime(df.index)

color = ["green", "White", "Red", "Yellow", "Green", "Grey"]
sns.set_palette(color)
sns.palplot(sns.color_palette())

df.plot(title='PJME Energy use in MegaWatts',
        figsize=(15, 8), style='.', color=sns.color_palette(current_palette)[5])
plt.xticks(rotation=45)
# Сохранение графика в файл
plt.savefig('PJME_Energy_use_in_MegaWatts.png')
print("График сохранен в файл PJME_Energy_use_in_MegaWatts.png")
plt.show()

ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW']
ax.plot(figsize=(15, 5), title='Потребление энергии за неделю',
       color=sns.color_palette(current_palette)[5])
plt.show()

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

# Создание DateFrame с дополнительными данными
df = create_features(df)
print(df)

plt.figure(figsize=(15, 6))
sns.set_style('whitegrid')
sns.boxenplot(x='Часы', y='PJME_MW', data=df, hue='Часы', palette='Blues')
plt.title('распределение потребления электроэнергии по часам')
plt.show()

plt.figure(figsize=(15, 6))
sns.set_style('whitegrid')
sns.boxenplot(x='Месяцы', y='PJME_MW', data=df, hue='Месяцы', palette='Greys')
plt.title('распределение потребления электроэнергии по месяцам')
plt.show()

plt.figure(figsize=(15, 6))
sns.set_style('whitegrid')
sns.boxenplot(x='Годы', y='PJME_MW', data=df, hue='Годы', palette='Accent')
plt.title('распределение потребления электроэнергии по годам')
plt.show()