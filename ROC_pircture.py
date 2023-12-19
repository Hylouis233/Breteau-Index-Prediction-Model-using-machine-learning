import joblib
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
import pickle
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, roc_auc_score

# 导入训练集相关数据
data = pd.read_csv('Siming_District_Breteau_Index_season.csv')
data['date'] = pd.to_datetime(data['date'])
data_test=pd.read_csv('fujian_total_predict.csv')
data_test['date']=pd.to_datetime(data_test['date'])

def assign_value(x):
    # if x >= 0 and x < 6:
    #     return 0
    # elif x >= 6 and x < 11:
    #     return 1
    # elif x>=11 and x<20:
    #     return 2
    # else:
    #     return 3
    if x < 20:
        return 0
    else:
        return 1

# season=pd.DataFrame(columns=['spring','summer','autumn','winter'])
# for i in range(len(data_test)):
#     if data_test['month'][i] in [3,4,5]:
#         season.loc[i]=[1,0,0,0]
#     elif data_test['month'][i] in [6,7,8]:
#         season.loc[i]=[0,1,0,0]
#     elif data_test['month'][i] in [9,10,11]:
#         season.loc[i]=[0,0,1,0]
#     else:
#         season.loc[i]=[0,0,0,1]
# data_test=pd.concat([data_test,season],axis=1)
# data_test.to_csv('fujian_total_predict.csv',index=False)


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

model = load_model('dataset2_binary.h5')
# 模型预测
Y_pred = model.predict(X_test)
Y_fit = model.predict(X_train)
Y_pred = Y_pred.reshape(-1, 1)
Y_pred = pd.DataFrame(Y_pred)
Y_fit = Y_fit.reshape(-1, 1)
Y_fit = pd.DataFrame(Y_fit)
X_test = X_test.reshape(-1, 78)
X_train = X_train.reshape(-1, 78)
Y_test = Y_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = pd.DataFrame(Y_test)
Y_train = pd.DataFrame(Y_train)
thresholds = np.linspace(0, 1, 10000)
fpr_list = []
tpr_list = []
for threshold in thresholds:
    y_pred_binary = (Y_pred > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_binary).ravel()
    fpr_list.append(fp / (fp + tn))
    tpr_list.append(tp / (tp + fn))
roc_auc = auc(fpr_list, tpr_list)
rdf = joblib.load('rdf_binary.pkl')
clf = joblib.load('clf_binary.pkl')
svm = joblib.load('svm_binary.pkl')
logistic = joblib.load('logistic_binary.pkl')
with open('clf_binary.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('rdf_binary.pkl', 'wb') as f:
    pickle.dump(rdf, f)
with open('svm_binary.pkl', 'wb') as f:
    pickle.dump(svm, f)
with open('logistic_binary.pkl', 'wb') as f:
    pickle.dump(logistic, f)
clf.fit(X_train, Y_train)
rdf.fit(X_train, Y_train)
svm.fit(X_train, Y_train)
logistic.fit(X_train, Y_train)
Y_pred_clf = clf.predict(X_test)
Y_pred_rdf = rdf.predict(X_test)
Y_pred_svm = svm.predict(X_test)
Y_train_pred_clf = clf.predict(X_train)
Y_train_pred_rdf = rdf.predict(X_train)
Y_train_pred_svm = svm.predict(X_train)
Y_pred_logistic = logistic.predict(X_test)
Y_train_pred_logistic = logistic.predict(X_train)

Y_pred_clf = clf.predict_proba(X_test)[:, 1]
Y_pred_rdf = rdf.predict_proba(X_test)[:, 1]
Y_pred_svm = svm.predict_proba(X_test)[:, 1]
Y_pred_logistic = logistic.predict_proba(X_test)[:, 1]
Y_pred_total=(Y_pred.values.reshape(-1)*0.97+Y_pred_rdf*0.8+Y_pred_svm*0.25+Y_pred_clf*0.5+Y_pred_logistic*0.77)/(0.97+0.75+0.8+0.77)
thresholds = np.linspace(0, 1, 10000)
tpr_list_clf = []
tnr_list_clf = []
tpr_list_rdf = []
tnr_list_rdf = []
tpr_list_svm = []
tnr_list_svm = []
tpr_list_logistic = []
tnr_list_logistic = []
tpr_list_total = []
tnr_list_total = []
for threshold in thresholds:
    Y_pred_clf_binary = (Y_pred_clf >= threshold).astype(int)
    Y_pred_rdf_binary = (Y_pred_rdf >= threshold).astype(int)
    Y_pred_svm_binary = (Y_pred_svm >= threshold).astype(int)
    Y_pred_logistic_binary = (Y_pred_logistic >= threshold).astype(int)
    Y_pred_total_binary = (Y_pred_total >= threshold).astype(int)

    tn_clf, fp_clf, fn_clf, tp_clf = confusion_matrix(Y_test, Y_pred_clf_binary).ravel()
    tn_rdf, fp_rdf, fn_rdf, tp_rdf = confusion_matrix(Y_test, Y_pred_rdf_binary).ravel()
    tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(Y_test, Y_pred_svm_binary).ravel()
    tn_logistic, fp_logistic, fn_logistic, tp_logistic = confusion_matrix(Y_test, Y_pred_logistic_binary).ravel()
    tn_total, fp_total, fn_total, tp_total = confusion_matrix(Y_test, Y_pred_total_binary).ravel()

    tpr_list_clf.append(tp_clf / (tp_clf + fn_clf))
    tnr_list_clf.append(1 - (tn_clf / (tn_clf + fp_clf)))

    tpr_list_rdf.append(tp_rdf / (tp_rdf + fn_rdf))
    tnr_list_rdf.append(1 - (tn_rdf / (tn_rdf + fp_rdf)))

    tpr_list_svm.append(tp_svm / (tp_svm + fn_svm))
    tnr_list_svm.append(1 - (tn_svm / (tn_svm + fp_svm)))

    tpr_list_logistic.append(tp_logistic / (tp_logistic + fn_logistic))
    tnr_list_logistic.append(1 - (tn_logistic / (tn_logistic + fp_logistic)))

    tpr_list_total.append(tp_total / (tp_total + fn_total))
    tnr_list_total.append(1 - (tn_total / (tn_total + fp_total)))

roc_auc_clf = auc(tnr_list_clf, tpr_list_clf)
roc_auc_rdf = auc(tnr_list_rdf, tpr_list_rdf)
roc_auc_svm = auc(tnr_list_svm, tpr_list_svm)
roc_auc_logistic = auc(tnr_list_logistic, tpr_list_logistic)
roc_auc_total = auc(tnr_list_total, tpr_list_total)

tpr_list_clf[-1] = 0
tnr_list_clf[-1] = 0

from scipy.stats import norm


def AUC_CI(auc, label, alpha=0.05):
    label = np.array(label)  # 防止label不是array类型
    n1, n2 = np.sum(label == 1), np.sum(label == 0)
    q1 = auc / (2 - auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2))
    confidence_level = 1 - alpha
    z_lower, z_upper = norm.interval(confidence_level)
    lowerb, upperb = auc + z_lower * se, auc + z_upper * se
    return (lowerb, upperb)

auc_clf=AUC_CI(roc_auc_clf, Y_test)
auc_rdf=AUC_CI(roc_auc_rdf, Y_test)
auc_svm=AUC_CI(roc_auc_svm, Y_test)
auc_logistic=AUC_CI(roc_auc_logistic, Y_test)
auc_total=AUC_CI(roc_auc_total, Y_test)
auc_dnn=AUC_CI(roc_auc, Y_test)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(12, 10))
plt.plot(fpr_list, tpr_list, lw=2, label=f'DNN (AUC = {roc_auc:.2f} (95% CI: {auc_dnn[0]:.2f} - {round(auc_dnn[1]):.2f}))')
plt.plot(tnr_list_clf, tpr_list_clf, color='darkorange', lw=2, label=f'Decision Tree (AUC = {roc_auc_clf:.2f} (95% CI: {auc_clf[0]:.2f} - {auc_clf[1]:.2f}))')
plt.plot(tnr_list_rdf, tpr_list_rdf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rdf:.2f} (95% CI: {auc_rdf[0]:.2f} - {auc_rdf[1]:.2f}))')
plt.plot(tnr_list_svm, tpr_list_svm, color='blue', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f} (95% CI: {auc_svm[0]:.2f} - {auc_svm[1]:.2f}))')
plt.plot(tnr_list_logistic, tpr_list_logistic, color='red', lw=2,
         label=f'Logistic Regression (AUC = {roc_auc_logistic:.2f} (95% CI: {auc_logistic[0]:.2f} - {auc_logistic[1]:.2f}))')
# plt.plot(tnr_list_total, tpr_list_total, color='mediumseagreen', lw=2,
#             label=f'Ensemble (AUC = {roc_auc_total:.2f} (95% CI: {auc_total[0]:.2f} - {round(auc_total[1]):.2f}))')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('1 - Specificity (False Positive Rate)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
plt.savefig('ROC_fujian.pdf')

#Delong检验
from Delong import DelongTest
models = {
    'clf': Y_pred_clf,
    'rdf': Y_pred_rdf,
    'svm': Y_pred_svm,
    'logistic': Y_pred_logistic,
    'DNN': Y_pred,
    'Ensemble': Y_pred_total
}

results = []

for model1_name, model1_pred in models.items():
    for model2_name, model2_pred in models.items():
        if model1_name != model2_name:
            test = DelongTest(model1_pred, model2_pred, Y_test[0])
            z, p_value = test._compute_z_p()
            if float(z)>=0:
                results.append({
                    'Model1': model1_name,
                    'Model2': model2_name,
                    'z-score': z,
                    'p-value': p_value
                })
df = pd.DataFrame(results)
df.to_csv('delong_test_results.csv', index=False)


#保存模型
import pickle
pickle.dump(clf, open('clf_binary.pkl', 'wb'))
pickle.dump(rdf, open('rdf_binary.pkl', 'wb'))
pickle.dump(svm, open('svm_binary.pkl', 'wb'))
pickle.dump(logistic, open('logistic_binary.pkl', 'wb'))

