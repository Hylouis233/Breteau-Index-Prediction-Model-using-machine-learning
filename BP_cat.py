import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Conv1D, LSTM
from keras.models import load_model
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
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

data=pd.read_csv('Siming_District_Breteau_Index_season.csv')
data['date']=pd.to_datetime(data['date'])
data_test=pd.read_csv('fujian_total_predict.csv')
data_test['date']=pd.to_datetime(data['date'])
def assign_value(x):
    if x >= 0 and x < 6:
        return 0
    elif x >= 6 and x < 11:
        return 1
    elif x>=11 and x<20:
        return 2
    else:
        return 3
    # if x<20:
    #     return 0
    # else:
    #     return 1
data['breteau_index'] = data['breteau_index'].apply(assign_value)

data_select_2=data[['breteau_index','bonsai','tank','containers',
                    'channels','hole','tires','litter',
                    'basement','other',
                    'households_ins','tem_low',
                    'tem_low7','tem_low6','tem_low5','tem_low4','tem_low3','tem_low2','tem_low1','tem_mean',
                    'tem_mean7','tem_mean6','tem_mean5','tem_mean4','tem_mean3','tem_mean2','tem_mean1','tem_high',
                    'tem_high7','tem_high6','tem_high5','tem_high4','tem_high3','tem_high2','tem_high1',
                    'sunshine_hours','sunshine_hours7','sunshine_hours6','sunshine_hours5','sunshine_hours4',
                    'sunshine_hours3','sunshine_hours2','sunshine_hours1','precipitation','precipitation7',
                    'precipitation6','precipitation5','precipitation4','precipitation3','precipitation2',
                    'precipitation1','humidity','humidity7','humidity6','humidity5','humidity4','humidity3',
                    'humidity2','humidity1','pressure','pressure7','pressure6','pressure5','pressure4','pressure3',
                    'pressure2','pressure1','wind','wind7','wind6','wind5','wind4','wind3','wind2','wind1','spring',
                    'summer','autumn','winter'
]]

# #选择数据集
# data=data_select_2
# data_select_2=data_select_2.values
# # 归一化
# data = min_max_scaler.fit_transform(data)
# Y = data_select_2[:, 0:1]
# X = data[:, 1:]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9)
# X_train = np.array(X_train)
# X_train=X_train.reshape(1,-1,78)
# X_test = np.array(X_test)
# X_test=X_test.reshape(1,-1,78)
# Y_train = np.array(Y_train).reshape(1,-1,1)
# Y_test = np.array(Y_test).reshape(1,-1,1)

#样本外推

data['breteau_index'] = data['breteau_index'].apply(assign_value)
data_test['breteau_index']=data_test['breteau_index'].apply(assign_value)


data_test_select_2=data_test[['breteau_index', 'bonsai', 'tank', 'containers',
                      'channels', 'hole', 'tires', 'litter',
                      'basement', 'other',
                      'households_ins', 'tem_low',
                      'tem_low7', 'tem_low6', 'tem_low5', 'tem_low4', 'tem_low3', 'tem_low2', 'tem_low1', 'tem_mean',
                      'tem_mean7', 'tem_mean6', 'tem_mean5', 'tem_mean4', 'tem_mean3', 'tem_mean2', 'tem_mean1',
                      'tem_high',
                      'tem_high7', 'tem_high6', 'tem_high5', 'tem_high4', 'tem_high3', 'tem_high2', 'tem_high1',
                      'sunshine_hours', 'sunshine_hours7', 'sunshine_hours6', 'sunshine_hours5', 'sunshine_hours4',
                      'sunshine_hours3', 'sunshine_hours2', 'sunshine_hours1', 'precipitation', 'precipitation7',
                      'precipitation6', 'precipitation5', 'precipitation4', 'precipitation3', 'precipitation2',
                      'precipitation1', 'humidity', 'humidity7', 'humidity6', 'humidity5', 'humidity4', 'humidity3',
                      'humidity2', 'humidity1', 'pressure', 'pressure7', 'pressure6', 'pressure5', 'pressure4',
                      'pressure3',
                      'pressure2', 'pressure1', 'wind', 'wind7', 'wind6', 'wind5', 'wind4', 'wind3', 'wind2', 'wind1',
                      'spring',
                      'summer', 'autumn', 'winter'
                      ]]
data_select_2 = data[['breteau_index', 'bonsai', 'tank', 'containers',
                      'channels', 'hole', 'tires', 'litter',
                      'basement', 'other',
                      'households_ins', 'tem_low',
                      'tem_low7', 'tem_low6', 'tem_low5', 'tem_low4', 'tem_low3', 'tem_low2', 'tem_low1', 'tem_mean',
                      'tem_mean7', 'tem_mean6', 'tem_mean5', 'tem_mean4', 'tem_mean3', 'tem_mean2', 'tem_mean1',
                      'tem_high',
                      'tem_high7', 'tem_high6', 'tem_high5', 'tem_high4', 'tem_high3', 'tem_high2', 'tem_high1',
                      'sunshine_hours', 'sunshine_hours7', 'sunshine_hours6', 'sunshine_hours5', 'sunshine_hours4',
                      'sunshine_hours3', 'sunshine_hours2', 'sunshine_hours1', 'precipitation', 'precipitation7',
                      'precipitation6', 'precipitation5', 'precipitation4', 'precipitation3', 'precipitation2',
                      'precipitation1', 'humidity', 'humidity7', 'humidity6', 'humidity5', 'humidity4', 'humidity3',
                      'humidity2', 'humidity1', 'pressure', 'pressure7', 'pressure6', 'pressure5', 'pressure4',
                      'pressure3',
                      'pressure2', 'pressure1', 'wind', 'wind7', 'wind6', 'wind5', 'wind4', 'wind3', 'wind2', 'wind1',
                      'spring',
                      'summer', 'autumn', 'winter'
                      ]]

# 选择数据集
data = data_select_2
data_select_2 = data_select_2.values
data_test=data_test_select_2
data_test_select_2=data_test_select_2.values

# 归一化
data=np.concatenate([data_test,data],axis=0)  #前4300是福建省的数据，后710是厦门市
data = min_max_scaler.fit_transform(data)
Y = data_test_select_2[:4300, 0:1]
X = data[:4300, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.95, random_state=0)
Y_test=Y
X_test=X
X_train = np.array(X_train)
X_train = X_train.reshape(1, -1, 78)
X_test = np.array(X_test)
X_test = X_test.reshape(1, -1, 78)
Y_train = np.array(Y_train).reshape(1, -1, 1)
Y_test = np.array(Y_test).reshape(1, -1, 1)

# # # BP模型
# model = Sequential()
# model.add(layers.Input(shape=( 1,X_train.shape[2])))
# model.add(layers.Dense(units=1024, activation='relu'))
# model.add(layers.Dropout(0.2))
# # model.add(LSTM(units=512, activation='relu',return_sequences=True))
# # model.add(LSTM(units=256, activation='relu',return_sequences=True))
# model.add(layers.Dense(units=128, activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(units=64, activation='relu'))
# # model.add(layers.Dense(units=4, activation='softmax'))
# model.add(layers.Dense(units=1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC','mse','accuracy'])
# model.summary()
# # #模型可视化
# # plot_model(model, show_shapes=True)

#读取最佳模型
# model = load_model('dataset2_binary.h5')
model=load_model('dataset2_multi.h5')
#sigmod+tanh
# # 独热编码
# Y_test = to_categorical(Y_test, 4)
# Y_train = to_categorical(Y_train, 4)
#训练模型
# filepath = "dataset2_multi.hdf5"
filepath = "dataset2_binary.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(X_train, Y_train, epochs=1000, batch_size=120, verbose=2, shuffle=False,
                    callbacks=callbacks_list, validation_data=(X_test, Y_test))
model.save('dataset2_binary.h5')





# 模型预测
Y_pred = model.predict(X_test)
Y_pred=Y_pred[-1,:]
Y_fit=model.predict(X_train)
Y_fit=Y_fit[-1,:]
# #独热编码转换，binary不用
Y_pred = np.argmax(Y_pred, axis=1)
Y_fit=np.argmax(Y_fit, axis=1)
#binary用这个
Y_pred = (Y_pred > 0.5).astype(int)
Y_fit = (Y_fit > 0.5).astype(int)
# # 模型评估
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import r2_score
# from math import sqrt
#
# # mse = mean_squared_error(Y_test, Y_pred)
# # mae = mean_absolute_error(Y_test, Y_pred)
# # rmse = sqrt(mse)
# # r2 = r2_score(Y_test, Y_pred)
# # print('mse:', mse)
# # print('mae:', mae)
# # print('rmse:', rmse)
# # print('r2:', r2)
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np
#
# best_column_r2 = None
# best_r2 = -1
# best_column_mse= None
# best_mse = 1000000000
# best_column_mae= None
# best_mae = 1000000000
# best_column_rmse= None
# best_rmse = 1000000000
#
# for col in range(Y_pred.shape[-1]):
#     Y_true_col = Y_test[0]
#     Y_pred_col = Y_pred[:, col]
#
#     mse = mean_squared_error(Y_true_col, Y_pred_col)
#     mae = mean_absolute_error(Y_true_col, Y_pred_col)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(Y_true_col, Y_pred_col)
#
#     print(f"列 {col+1} 的 MSE: {mse}")
#     print(f"列 {col+1} 的 MAE: {mae}")
#     print(f"列 {col+1} 的 RMSE: {rmse}")
#     print(f"列 {col+1} 的 R-squared: {r2}")
#
#     if r2 > best_r2:
#         best_r2 = r2
#         best_column = col
#     if mse < best_mse:
#         best_mse = mse
#         best_column_mse = col
#     if mae < best_mae:
#         best_mae = mae
#         best_column_mae = col
#     if rmse < best_rmse:
#         best_rmse = rmse
#         best_column_rmse = col
#
# print(f"最佳决定系数位于列 {best_column+1}")
# print(f"最佳MSE位于列 {best_column_mse+1}")
# print(f"最佳MAE位于列 {best_column_mae+1}")
# print(f"最佳RMSE位于列 {best_column_rmse+1}")



# 反归一化
Y_pred = pd.DataFrame(Y_pred)
Y_fit=Y_fit.reshape(-1,1)
Y_fit=pd.DataFrame(Y_fit)
X_test=X_test.reshape(-1,78)
X_train=X_train.reshape(-1,78)
Y_test=Y_test.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
Y_test=pd.DataFrame(Y_test)
Y_train=pd.DataFrame(Y_train)
# data_pred = pd.concat([pd.DataFrame(Y_pred), pd.DataFrame(X_test)], axis=1)
# data_pred = min_max_scaler.inverse_transform(data_pred)
# data_real = pd.concat([pd.DataFrame(Y_test), pd.DataFrame(X_test)], axis=1)
# data_real = min_max_scaler.inverse_transform(data_real)
# Y_pred=data_pred[:,0:1]
# Y_test=data_real[:,0:1]
# data_fit=pd.concat([pd.DataFrame(Y_fit), pd.DataFrame(X_train)], axis=1)
# data_fit = min_max_scaler.inverse_transform(data_fit)
# data_fit_real=pd.concat([pd.DataFrame(Y_train), pd.DataFrame(X_train)], axis=1)
# data_fit_real = min_max_scaler.inverse_transform(data_fit_real)
# Y_train=data_fit_real[:,0:1]
# Y_fit=data_fit[:,0:1]



# 模型可视化
plt.figure(figsize=(20,12))
plt.plot(history.history['loss'][0:200])
# plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'][0:200])
# plt.plot(history.history['val_accuracy'])
plt.title('model loss')
# plt.title('model accuracy')
plt.ylabel('loss')
# plt.ylabel('accuracy')
# plt.ylim(0.6,1)
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 模型预测可视化
# plt.figure(figsize=(20,12))
# plt.scatter(Y_test,Y_pred)
# plt.title('model predict')
# plt.ylabel('breteau_index_predict')
# plt.xlabel('breteau_index_real')
# plt.legend(['real', 'predict'], loc='upper left')
# plt.show()

# coefficients_predict = np.polyfit(np.concatenate(Y_test),np.concatenate(Y_pred), 1)
# polynomial_predict = np.poly1d(coefficients_predict)
# coefficients_fitted = np.polyfit(np.concatenate(Y_train), np.concatenate(Y_fit), 1)
# polynomial_fitted = np.poly1d(coefficients_fitted)
# x_predict=np.linspace(0,100,100)
# y_predict=polynomial_predict(x_predict)
# x_fitted=np.linspace(0,100,100)
# y_fitted=polynomial_fitted(x_fitted)

# plt.figure(figsize=(20,12))
# plt.scatter(Y_test,Y_pred,color='green',label='predict',alpha=0.5)
# plt.plot(x_predict,y_predict,color='salmon',label='predict',alpha=0.5,linestyle='dashed')
# plt.scatter(Y_train,Y_fit,color='blue',label='fit',alpha=0.3)
# plt.plot(x_fitted,y_fitted,color='darksalmon',label='fit',linestyle='dashed')
# plt.plot([0,100],[0,100],color='thistle',linestyle='dotted')
# plt.xlabel('real')
# plt.ylabel('predict/fit')
# plt.legend()
# plt.show()
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix

print('DNN_test:',classification_report(Y_test,Y_pred))
print('DNN_train:',classification_report(Y_train,Y_fit))
DNN_test_report=classification_report(Y_test,Y_pred,output_dict=True)
DNN_train_report=classification_report(Y_train,Y_fit,output_dict=True)
DNN_test_df=pd.DataFrame(DNN_test_report).transpose()
DNN_train_df=pd.DataFrame(DNN_train_report).transpose()
with pd.ExcelWriter('fujian_mul_DNN.xlsx') as writer:
    DNN_test_df.to_excel(writer, sheet_name='DNN_test')
    DNN_train_df.to_excel(writer, sheet_name='DNN_train')

#绘制混淆矩阵
thresholds = np.linspace(0, 1, 10000)
fpr_list = []
tpr_list = []
for threshold in thresholds:
    y_pred_binary = (Y_pred > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_binary).ravel()
    fpr_list.append(fp / (fp + tn))
    tpr_list.append(tp / (tp + fn))
roc_auc = auc(fpr_list, tpr_list)
plt.figure(figsize=(10, 6))
plt.plot(fpr_list, tpr_list, lw=2, label=f'DNN = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('1 - Specificity (False Positive Rate)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()