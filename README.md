# FOREST
# Подключение библиотек
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Считываения файла
df = pd.read_csv("./data/transport_data.csv")

df.drop(df.index[df.label == '-'].tolist(), axis = 0, inplace = True) # Удаляем записи с "-"

df.request_ts = df.request_ts - df.trans_ts # Заменяем requst_ts на разницу между request_ts и trans_ts

df.trans_ts = pd.to_datetime(df.trans_ts, unit='s') # Переводим формат колонки trans_ts формат datetime ???????????????????????????????
df['second_trans_ts'] = df.trans_ts.dt.second
df['minute_trans_ts'] = df.trans_ts.dt.minute
df['hour_trans_ts'] = df.trans_ts.dt.hour
df['dayofweek_trans_ts'] = df.trans_ts.dt.dayofweek
df.drop('trans_ts', axis = 1, inplace=True)
target = df[df.label == '?'].drop('label', axis = 1) # Выносим строки с "?" в отдельную переменную (получится таблица) target + удаляем в этой таблице колонку label ( т.к. там только "?", мы их будем предсказывать, удаляем чтобы не занимать память )
df.drop(df.index[df.label == '?'].tolist(), axis = 0, inplace = True) # Удаляем строки с "?" из df.

model = GradientBoostingClassifier(max_depth = 13, max_features = 'auto', n_estimators = 135) # Создаем модель с параметрами.
model.fit(df.drop(['label'], axis=1), df.label) # Обучаем модель на тренеровочной выборке.

preds = model.predict(target) # Предсказываем target, записываем ответы в preds
with open('results.txt', 'w') as f: # Открываем файл на запись 'results.txt' (если его нету, он создастся). Помечаем его в программе как "f". 
    for p in preds: # Берем каждый элемент из ответов 'preds'
        f.write(p + "\n") # записываем его в файл (\n - перенос строки, каждый элемент на новую строку)
