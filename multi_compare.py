import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, auc
import graphviz
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge
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
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import pickle
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, roc_auc_score

a='_binary'
data = pd.read_csv('Siming_District_Breteau_Index_season.csv')
data['date'] = pd.to_datetime(data['date'])
data_test=pd.read_csv('fujian_total_predict.csv')
data_test['date']=pd.to_datetime(data_test['date'])

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

model = load_model(f'dataset2{a}.h5')
rdf=joblib.load(f'rdf{a}.pkl')
clf=joblib.load(f'clf{a}.pkl')
svm=joblib.load(f'svm{a}.pkl')
logistic=joblib.load(f'logistic{a}.pkl')
with open(f'clf{a}.pkl','wb') as f:
    pickle.dump(clf,f)
with open(f'rdf{a}.pkl','wb') as f:
    pickle.dump(rdf,f)
with open(f'svm{a}.pkl','wb') as f:
    pickle.dump(svm,f)
with open(f'logistic{a}.pkl','wb') as f:
    pickle.dump(logistic,f)
predict_DNN=model.predict(X_test)
if a=='_multi':
    predict_DNN=np.argmax(predict_DNN, axis=1)
else:
    predict_DNN=predict_DNN.reshape(-1,1)

real_data=Y_test.reshape(-1,1)
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
Y_test= np.vectorize(assign_value)(real_data)

def bootstrap_sample(data, target):
    n = len(data)
    indices = np.random.choice(n, n, replace=False)
    sample_data = data.iloc[indices].reset_index(drop=True)
    sample_target = target.iloc[indices].reset_index(drop=True)
    return sample_data, sample_target


def predict_models(data):
    predict_RF = rdf.predict(data)
    predict_SVM = svm.predict(data)
    predict_DT = clf.predict(data)
    predict_Logistic = logistic.predict(data)
    y_pred_total = (predict_DNN * 0.97 + predict_RF * 0.8 + predict_SVM * 0.25 + predict_DT * 0.5 + predict_Logistic * 0.77) / (
                    0.97 + 0.75 + 0.8 + 0.77)
    y_pred_total = np.round(y_pred_total)

    return {
        'RF': predict_RF,
        'SVM': predict_SVM,
        'DT': predict_DT,
        'Logistic': predict_Logistic,
        'Total': y_pred_total
    }


accuracy = {key: [] for key in ['RF', 'SVM', 'DT', 'Logistic', 'Total']}
precision = {key: [] for key in ['RF', 'SVM', 'DT', 'Logistic', 'Total']}
recall = {key: [] for key in ['RF', 'SVM', 'DT', 'Logistic', 'Total']}
f1 = {key: [] for key in ['RF', 'SVM', 'DT', 'Logistic', 'Total']}

X_test= pd.DataFrame(X_test.reshape(-1,78))
Y_test= pd.DataFrame(Y_test.reshape(-1,1))
for _ in range(20):
    bootstrap_sample_data, bootstrap_sample_target = bootstrap_sample(X_test, Y_test)
    predictions = predict_models(bootstrap_sample_data.values)

    for model, y_pred in predictions.items():
        report = classification_report(bootstrap_sample_target, Y_test, output_dict=True)
        accuracy[model].append(report['accuracy'])
        precision[model].append(report['weighted avg']['precision'])
        recall[model].append(report['weighted avg']['recall'])
        f1[model].append(report['weighted avg']['f1-score'])

results = {
    "Comparison": [],
    "Metric": [],
    "T-Value": [],
    "P-Value": []
}

models = ['RF', 'SVM', 'DT', 'Logistic', 'Total']
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        t_stat_acc, p_val_acc = ttest_rel(accuracy[models[i]], accuracy[models[j]])
        t_stat_prec, p_val_prec = ttest_rel(precision[models[i]], precision[models[j]])
        t_stat_rec, p_val_rec = ttest_rel(recall[models[i]], recall[models[j]])
        t_stat_f1, p_val_f1 = ttest_rel(f1[models[i]], f1[models[j]])

        results["Comparison"].append(f"{models[i]} vs {models[j]}")
        results["Metric"].append("Accuracy")
        results["T-Value"].append(t_stat_acc)
        results["P-Value"].append(p_val_acc)

        results["Comparison"].append(f"{models[i]} vs {models[j]}")
        results["Metric"].append("Precision")
        results["T-Value"].append(t_stat_prec)
        results["P-Value"].append(p_val_prec)

        results["Comparison"].append(f"{models[i]} vs {models[j]}")
        results["Metric"].append("Recall")
        results["T-Value"].append(t_stat_rec)
        results["P-Value"].append(p_val_rec)

        results["Comparison"].append(f"{models[i]} vs {models[j]}")
        results["Metric"].append("F1-Score")
        results["T-Value"].append(t_stat_f1)
        results["P-Value"].append(p_val_f1)

df_results = pd.DataFrame(results)
df_results.to_csv("binary_fujian_model_comparison.csv", index=False)


