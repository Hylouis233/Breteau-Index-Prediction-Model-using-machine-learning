from keras import Sequential
from keras.models import load_model
from keras.optimizer_v2.adam import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
min_max_scaler = preprocessing.MinMaxScaler()

# 导入训练集相关数据
data = pd.read_csv('Siming_District_Breteau_Index_1.csv')
data['date'] = pd.to_datetime(data['date'])
data_select_1 = data[['precipitation', 'precipitation7',
                      'precipitation6', 'precipitation5', 'precipitation4', 'precipitation3', 'precipitation2',
                      'precipitation1','breteau_index', 'weather', 'door', 'bonsai', 'tank', 'containers',
                      'channels', 'hole', 'tires', 'litter',
                      'basement', 'other',
                      'households_ins', 'tem_low',
                      'tem_low7', 'tem_low6', 'tem_low5', 'tem_low4', 'tem_low3', 'tem_low2', 'tem_low1', 'tem_mean',
                      'tem_mean7', 'tem_mean6', 'tem_mean5', 'tem_mean4', 'tem_mean3', 'tem_mean2', 'tem_mean1',
                      'tem_high',
                      'tem_high7', 'tem_high6', 'tem_high5', 'tem_high4', 'tem_high3', 'tem_high2', 'tem_high1',
                      'sunshine_hours', 'sunshine_hours7', 'sunshine_hours6', 'sunshine_hours5', 'sunshine_hours4',
                      'sunshine_hours3', 'sunshine_hours2', 'sunshine_hours1',  'humidity', 'humidity7', 'humidity6', 'humidity5', 'humidity4', 'humidity3',
                      'humidity2', 'humidity1', 'pressure', 'pressure7', 'pressure6', 'pressure5', 'pressure4',
                      'pressure3',
                      'pressure2', 'pressure1', 'wind', 'wind7', 'wind6', 'wind5', 'wind4', 'wind3', 'wind2', 'wind1'
                      ]]
data_select_1 = data_select_1.dropna(axis=0, how='any')
data_select_2 = data[['precipitation', 'precipitation7',
                      'precipitation6', 'precipitation5', 'precipitation4', 'precipitation3', 'precipitation2',
                      'precipitation1','breteau_index', 'bonsai', 'tank', 'containers',
                      'channels', 'hole', 'tires', 'litter',
                      'basement', 'other',
                      'households_ins', 'tem_low',
                      'tem_low7', 'tem_low6', 'tem_low5', 'tem_low4', 'tem_low3', 'tem_low2', 'tem_low1', 'tem_mean',
                      'tem_mean7', 'tem_mean6', 'tem_mean5', 'tem_mean4', 'tem_mean3', 'tem_mean2', 'tem_mean1',
                      'tem_high',
                      'tem_high7', 'tem_high6', 'tem_high5', 'tem_high4', 'tem_high3', 'tem_high2', 'tem_high1',
                      'sunshine_hours', 'sunshine_hours7', 'sunshine_hours6', 'sunshine_hours5', 'sunshine_hours4',
                      'sunshine_hours3', 'sunshine_hours2', 'sunshine_hours1',  'humidity', 'humidity7', 'humidity6', 'humidity5', 'humidity4', 'humidity3',
                      'humidity2', 'humidity1', 'pressure', 'pressure7', 'pressure6', 'pressure5', 'pressure4',
                      'pressure3',
                      'pressure2', 'pressure1', 'wind', 'wind7', 'wind6', 'wind5', 'wind4', 'wind3', 'wind2', 'wind1'
                      ]]
data = data_select_2
# 归一化
data = min_max_scaler.fit_transform(data)
Y = data[:, 0:8]
X = data[:, 8:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
X_train = np.array(X_train)
X_train=X_train.reshape(1,-1,67)
X_test = np.array(X_test)
X_test=X_test.reshape(1,-1,67)
Y_train = np.array(Y_train).reshape(1,-1,8)
Y_test = np.array(Y_test).reshape(1,-1,8)

# # BP模型
model = Sequential()
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
model.add(layers.Input(shape=( 1,X_train.shape[2])))
model.add(layers.Dense(units=1024, activation='relu'))
model.add(layers.Dropout(0.2))
# model.add(LSTM(units=512, activation='relu',return_sequences=True))
# model.add(LSTM(units=256, activation='relu',return_sequences=True))
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=64, activation='sigmoid'))
model.add(layers.Dense(units=8, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()
#模型可视化
plot_model(model, show_shapes=True)

# 训练模型
filepath = "rainfit.hdf5"
callbacks_list = [ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')]
history = model.fit(X_train, Y_train, epochs=300, batch_size=60, verbose=2, shuffle=False,
                    callbacks=callbacks_list, validation_data=(X_test, Y_test))
model.save('rainfit.h5')

model=load_model('rainfit.h5')
# 模型预测
Y_pred = model.predict(X_test)
Y_fit=model.predict(X_train)

# 反归一化
Y_pred=Y_pred[0,:,:]
Y_fit=Y_fit[0,:,:]
Y_pred = pd.DataFrame(Y_pred)
Y_fit=pd.DataFrame(Y_fit)
X_test=X_test.reshape(-1,67)
X_train=X_train.reshape(-1,67)
Y_test=Y_test.reshape(-1,8)
Y_train=Y_train.reshape(-1,8)
data_pred = pd.concat([pd.DataFrame(Y_pred), pd.DataFrame(X_test)], axis=1)
data_pred = min_max_scaler.inverse_transform(data_pred)
data_real = pd.concat([pd.DataFrame(Y_test), pd.DataFrame(X_test)], axis=1)
data_real = min_max_scaler.inverse_transform(data_real)
Y_pred=data_pred[:,0:8]
Y_test=data_real[:,0:8]
data_fit=pd.concat([pd.DataFrame(Y_fit), pd.DataFrame(X_train)], axis=1)
data_fit = min_max_scaler.inverse_transform(data_fit)
data_fit_real=pd.concat([pd.DataFrame(Y_train), pd.DataFrame(X_train)], axis=1)
data_fit_real = min_max_scaler.inverse_transform(data_fit_real)
Y_train=data_fit_real[:,0:8]
Y_fit=data_fit[:,0:8]
Y_fit[Y_fit<0]=0


