import yfinance as yf
import yahoofinancials
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') # специальное отображение графиков для pyplot

AAPL = yf.download('AAPL',
                      start='2020-01-01',
                      end='2020-06-06',
                      progress=False)
plt.figure(figsize = (16,8))
plt.title('Close Price History')
plt.plot(AAPL['Close'])
plt.xlabel('Date')
plt.ylabel('Close price USD')
plt.savefig('img_ii1.jpg')

# Создаем новый датафрейм только с колонкой "Close"
data = AAPL.filter(['Close'])
# преобразовываем в нумпаевский массив
dataset = data.values
# Вытаскиваем количество строк в дате для обучения модели (LSTM)
training_data_len = math.ceil(len(dataset) * .8)

#Scale the data (масштабируем)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Создаем датасет для обучения
train_data = scaled_data[0:training_data_len]
# разбиваем на x underscore train и y underscore train
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i])
    y_train.append(train_data[i])

#Конвертируем x_train и y_train в нумпаевский массив
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape data
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],x_train.shape[2]))

#Строим нейронку
model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(LSTM(50,return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#Компилируем модель
model.compile(optimizer='adam',loss = 'mean_squared_error')

#Тренируем модель
model.fit(x_train,y_train,batch_size = 1, epochs = 10)

#Создаем тестовый датасет
test_data = scaled_data[training_data_len - 60:]
#по аналогии создаем x_test и y_test
x_test = []
y_test = dataset[training_data_len:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i])

#опять преобразуем в нумпаевский массив
x_test = np.array(x_test)

#опять делаем reshape
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2]))

#Получаем модель предсказывающую значения
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Получим mean squared error (RMSE) - метод наименьших квадратов
rmse =np.sqrt(np.mean(predictions-y_test)**2)

#Строим график
train = data[:training_data_len]
valid = data[training_data_len: ]
valid['Predictions'] = predictions
#Визуализируем
plt.figure(figsize=(16,8))
plt.title('Model LSTM')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price',fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Pred'], loc = 'lower right')
plt.savefig('img_ii1.jpg')