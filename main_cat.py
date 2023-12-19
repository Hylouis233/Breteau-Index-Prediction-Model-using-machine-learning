import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    if x<20:
        return 0
    else:
        return 1
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

# svm=SVC(probability=True)
# logistic=LogisticRegression(multi_class="multinomial")
# #binary
# clf=DecisionTreeClassifier()
# rdf=RandomForestClassifier(min_weight_fraction_leaf= 0.0, min_samples_split= 20, min_samples_leaf= 1, min_impurity_decrease= 0.0, max_depth= 20, ccp_alpha= 0.0
# )
# #multi
# rdf=RandomForestClassifier(min_weight_fraction_leaf= 0.0, min_samples_split= 10, min_samples_leaf= 2, min_impurity_decrease= 0.0, max_depth= 90, ccp_alpha= 0.0)
# clf=DecisionTreeClassifier()
# rdf=joblib.load('rdf_siming.pkl')
from sklearn.model_selection import RandomizedSearchCV

#读取模型
rdf=joblib.load('rdf_binary.pkl')
clf=joblib.load('clf_binary.pkl')
svm=joblib.load('svm_binary.pkl')
logistic=joblib.load('logistic_binary.pkl')

# rdf=joblib.load('rdf_multi.pkl')
# clf=joblib.load('clf_multi.pkl')
# svm=joblib.load('svm_multi.pkl')
# logistic=joblib.load('logistic_multi.pkl')

#data=原始data
# Y=data.iloc[:,5:6]
# X=data.iloc[:,6:]

#dataselect后：
Y=data_test.iloc[:,0:1]
X=data_test.iloc[:,1:]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.95)
Y_train=Y_train.values.ravel()
Y_test=Y_test.values.ravel()
clf.fit(X_train,Y_train)
rdf.fit(X_train,Y_train)
svm.fit(X_train,Y_train)
logistic.fit(X_train,Y_train)
Y_pred_clf=clf.predict(X_test)
Y_pred_rdf=rdf.predict(X_test)
Y_pred_svm=svm.predict(X_test)
Y_train_pred_clf=clf.predict(X_train)
Y_train_pred_rdf=rdf.predict(X_train)
Y_train_pred_svm=svm.predict(X_train)
Y_pred_logistic=logistic.predict(X_test)
Y_train_pred_logistic=logistic.predict(X_train)

#
# # 随机森林调参
# param_dist = {
#     'max_depth': [10, 20, 30, 40,50,60,70,80,90,100],
#     'min_samples_split': [10, 20, 30, 40, 50],
#     'min_samples_leaf': [1, 2, 4,5,8,10],
#     'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
#     'min_impurity_decrease': [0.0, 0.1, 0.2],
#     'ccp_alpha': [0.0, 0.1, 0.2],
# }
# kf = KFold(n_splits=10, shuffle=True, random_state=42)
# # 创建RandomizedSearchCV对象
# random_search = RandomizedSearchCV(
#     estimator=RandomForestClassifier(),
#     param_distributions=param_dist,
#     n_iter=100,  # 增加了随机采样的次数，控制迭代次数
#     cv=kf,
#     scoring='neg_mean_squared_error',
#     random_state=114514,
#     n_jobs=-1
# )
#
# # 在训练集上执行随机搜索
# random_search.fit(X_train, Y_train)
# #
# #
# # # # 打印最佳参数组合和对应的性能指标
# print("Best parameters:", random_search.best_params_)
# print("Best negative mean squared error:", random_search.best_score_)
# # Best parameters: {'min_weight_fraction_leaf': 0.0, 'min_samples_split': 10, 'min_samples_leaf': 2, 'min_impurity_decrease': 0.1, 'max_depth': 30, 'ccp_alpha': 0.0}
# # Best negative mean squared error: -103.89750584684889
# #dataselect_2:
# # min_weight_fraction_leaf= 0.0, min_samples_split= 10, min_samples_leaf= 2, min_impurity_decrease= 0.1, max_depth= 30, ccp_alpha= 0.2
# # Best negative mean squared error: -128.61055908296203
#
# # 决策树调参
# param_dist = {
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2', None],
#     'random_state': [None, 42],
#     'max_leaf_nodes': [None, 10, 50, 100, 500],
#     'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3],
#     'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3],
#     'ccp_alpha': [0.0, 0.1, 0.2, 0.3]
# }
# kf = KFold(n_splits=10, shuffle=True, random_state=42)
# # 创建RandomizedSearchCV对象
# random_search = RandomizedSearchCV(
#     estimator=DecisionTreeClassifier(),
#     param_distributions=param_dist,
#     n_iter=100,  # 增加了随机采样的次数，控制迭代次数
#     cv=kf,
#     scoring='precision',
#     random_state=114514,
#     n_jobs=-1
# )
#
# # 在训练集上执行随机搜索
# random_search.fit(X_train, Y_train)
# print("Best parameters:", random_search.best_params_)

#保存模型
import pickle
# with open('clf_multi.pkl','wb') as f:
#     pickle.dump(clf,f)
# with open('rdf_multi.pkl','wb') as f:
#     pickle.dump(rdf,f)
# with open('svm_multi.pkl','wb') as f:
#     pickle.dump(svm,f)
# with open('logistic_multi.pkl','wb') as f:
#     pickle.dump(logistic,f)
with open('clf_binary.pkl','wb') as f:
    pickle.dump(clf,f)
with open('rdf_binary.pkl','wb') as f:
    pickle.dump(rdf,f)
with open('svm_binary.pkl','wb') as f:
    pickle.dump(svm,f)
with open('logistic_binary.pkl','wb') as f:
    pickle.dump(logistic,f)

#报告灵敏度、特异度
from sklearn.metrics import classification_report
print('clf_test:',classification_report(Y_test,Y_pred_clf))
print('rdf_test:',classification_report(Y_test,Y_pred_rdf))
print('svm_test:',classification_report(Y_test,Y_pred_svm))
print('clf_train:',classification_report(Y_train,Y_train_pred_clf))
print('rdf_train:',classification_report(Y_train,Y_train_pred_rdf))
print('svm_train:',classification_report(Y_train,Y_train_pred_svm))
print('logistic_test:',classification_report(Y_test,logistic.predict(X_test)))
print('logistic_train:',classification_report(Y_train,logistic.predict(X_train)))

#保存
import pandas as pd
from sklearn.metrics import classification_report
clf_test_report = classification_report(Y_test, Y_pred_clf, output_dict=True)
rdf_test_report = classification_report(Y_test, Y_pred_rdf, output_dict=True)
svm_test_report = classification_report(Y_test, Y_pred_svm, output_dict=True)
clf_train_report = classification_report(Y_train, Y_train_pred_clf, output_dict=True)
rdf_train_report = classification_report(Y_train, Y_train_pred_rdf, output_dict=True)
svm_train_report = classification_report(Y_train, Y_train_pred_svm, output_dict=True)
logistic_test_report = classification_report(Y_test, logistic.predict(X_test), output_dict=True)
logistic_train_report = classification_report(Y_train, logistic.predict(X_train), output_dict=True)
clf_test_df = pd.DataFrame(clf_test_report).transpose()
rdf_test_df = pd.DataFrame(rdf_test_report).transpose()
svm_test_df = pd.DataFrame(svm_test_report).transpose()
clf_train_df = pd.DataFrame(clf_train_report).transpose()
rdf_train_df = pd.DataFrame(rdf_train_report).transpose()
svm_train_df = pd.DataFrame(svm_train_report).transpose()
logistic_test_df = pd.DataFrame(logistic_test_report).transpose()
logistic_train_df = pd.DataFrame(logistic_train_report).transpose()
with pd.ExcelWriter('fujian_binary_classification_reports.xlsx') as writer:
    clf_test_df.to_excel(writer, sheet_name='clf_test_report')
    rdf_test_df.to_excel(writer, sheet_name='rdf_test_report')
    svm_test_df.to_excel(writer, sheet_name='svm_test_report')
    clf_train_df.to_excel(writer, sheet_name='clf_train_report')
    rdf_train_df.to_excel(writer, sheet_name='rdf_train_report')
    svm_train_df.to_excel(writer, sheet_name='svm_train_report')
    logistic_test_df.to_excel(writer, sheet_name='logistic_test_report')
    logistic_train_df.to_excel(writer, sheet_name='logistic_train_report')



#返回系数
df_clf=pd.DataFrame(clf.feature_importances_)
df_clf.index=clf.feature_names_in_
df_clf=df_clf.sort_values(by=0,ascending=False)
plt.figure(figsize=(15,10))
plt.bar(df_clf.index[0:10],df_clf[0][0:10])
for a,b in zip(df_clf.index[0:10],df_clf[0][0:10]):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=10)
plt.xticks(rotation=45)
plt.title('DecisionTreeRegressor')
plt.show()

df_clf.to_csv('clf_siming_cat.csv')
df_rdf=pd.DataFrame(rdf.feature_importances_)
df_rdf.index=rdf.feature_names_in_
df_rdf=df_rdf.sort_values(by=0,ascending=False)
plt.figure(figsize=(15,10))
plt.bar(df_rdf.index[0:20],df_rdf[0][0:20])
for a,b in zip(df_rdf.index[0:20],df_rdf[0][0:20]):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=10)
plt.xticks(rotation=45)
plt.title('RandomForestRegressor')
plt.show()
df_rdf.to_csv('rdf_siming_cat.csv')


#
# #多项式拟合
# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X_train)
# X_poly_test = poly.fit_transform(X_test)
# score=-99999
# for i in np.arange(0,10,0.01):
#     ridge = Ridge(alpha=i)
#     ridge.fit(X_poly, Y_train)
#     Y_pred_poly_reg=ridge.predict(X_poly_test)
#     score_ridge=ridge.score(X_poly_test,Y_test)
#     if score_ridge>score:
#         score=score_ridge
#         Y_pred_poly_reg_best=Y_pred_poly_reg
#         ridge_best=ridge
#     print(ridge_best)
#
# #返回岭回归参数Ridge(alpha=0.46)
# df_ridge=pd.DataFrame(ridge_best.coef_)
# df_ridge.to_csv('ridge_siming.csv')
# Y_train_poly_reg_best=ridge_best.predict(X_poly)



plt.figure(figsize=(10,10))
plt.scatter(Y_test,Y_pred_clf,color='red',label='DecisionTreeRegressor')
plt.scatter(Y_test,Y_pred_rdf,color='green',label='RandomForestRegressor_test')
plt.scatter(Y_train,Y_train_pred_rdf,color='blue',label='RandomForestRegressor_train')
plt.scatter(Y_test,Y_pred_clf,color='green',label='DecisionTreeRegressor_test')
plt.scatter(Y_train,Y_train_pred_clf,color='blue',label='DecisionTreeRegressor_train')
# plt.scatter(Y_test,Y_pred_poly_reg_best,color='green',label='PolynomialFeatures')
# plt.scatter(Y_train,Y_train_poly_reg_best,color='blue',label='PolynomialFeatures_train')
plt.plot([0,100],[0,100],color='thistle',linestyle='dotted')
plt.legend()
plt.show()
#多项式R2
# print('poly_test:',r2_score(Y_test,Y_pred_poly_reg_best))

# print(clf.score(X_test,Y_test))
print(rdf.score(X_test,Y_test))
print(rdf.score(X_train,Y_train))
# print(ridge_best.score(X_poly_test,Y_test))
#保存模型
joblib.dump(clf,'clf_siming.pkl')
joblib.dump(rdf,'rdf_siming_1.pkl')

#绘制rdf的灵敏度及特异度
# Y_pred_rdf=rdf.predict(X_test)
# Y_pred_rdf_train=rdf.predict(X_train)
# Y_pred_rdf=Y_pred_rdf.reshape(-1,1)
# Y_pred_rdf_train=Y_pred_rdf_train.reshape(-1,1)
# Y_test=Y_test.reshape(-1,1)
# Y_train=Y_train.reshape(-1,1)
# Y_pred_rdf=np.hstack((Y_pred_rdf,Y_test))
# Y_pred_rdf_train=np.hstack((Y_pred_rdf_train,Y_train))
# Y_pred_rdf=pd.DataFrame(Y_pred_rdf)
# Y_pred_rdf_train=pd.DataFrame(Y_pred_rdf_train)
# Y_pred_rdf=Y_pred_rdf.sort_values(by=0,ascending=False)
# Y_pred_rdf_train=Y_pred_rdf_train.sort_values(by=0,ascending=False)
# Y_pred_rdf=Y_pred_rdf.values
# Y_pred_rdf_train=Y_pred_rdf_train.values
# Y_pred_rdf=Y_pred_rdf.reshape(-1,1)
# Y_pred_rdf_train=Y_pred_rdf_train.reshape(-1,1)

#绘制曲线
Y_pred_clf = clf.predict_proba(X_test)[:, 1]
Y_pred_rdf = rdf.predict_proba(X_test)[:, 1]
Y_pred_svm = svm.predict_proba(X_test)[:, 1]
Y_pred_logistic = logistic.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 10000)
tpr_list_clf = []
tnr_list_clf = []
tpr_list_rdf = []
tnr_list_rdf = []
tpr_list_svm = []
tnr_list_svm = []
tpr_list_logistic = []
tnr_list_logistic = []
for threshold in thresholds:
    Y_pred_clf_binary = (Y_pred_clf >= threshold).astype(int)
    Y_pred_rdf_binary = (Y_pred_rdf >= threshold).astype(int)
    Y_pred_svm_binary = (Y_pred_svm >= threshold).astype(int)
    Y_pred_logistic_binary = (Y_pred_logistic >= threshold).astype(int)

    tn_clf, fp_clf, fn_clf, tp_clf = confusion_matrix(Y_test, Y_pred_clf_binary).ravel()
    tn_rdf, fp_rdf, fn_rdf, tp_rdf = confusion_matrix(Y_test, Y_pred_rdf_binary).ravel()
    tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(Y_test, Y_pred_svm_binary).ravel()
    tn_logistic, fp_logistic, fn_logistic, tp_logistic = confusion_matrix(Y_test, Y_pred_logistic_binary).ravel()
    tpr_list_clf.append(tp_clf / (tp_clf + fn_clf))
    tnr_list_clf.append(1-(tn_clf / (tn_clf + fp_clf)))

    tpr_list_rdf.append(tp_rdf / (tp_rdf + fn_rdf))
    tnr_list_rdf.append(1-(tn_rdf / (tn_rdf + fp_rdf)))

    tpr_list_svm.append(tp_svm / (tp_svm + fn_svm))
    tnr_list_svm.append(1-(tn_svm / (tn_svm + fp_svm)))

    tpr_list_logistic.append(tp_logistic / (tp_logistic + fn_logistic))
    tnr_list_logistic.append(1-(tn_logistic / (tn_logistic + fp_logistic)))
roc_auc_clf = auc(tnr_list_clf, tpr_list_clf)
roc_auc_rdf = auc(tnr_list_rdf, tpr_list_rdf)
roc_auc_svm = auc(tnr_list_svm, tpr_list_svm)
roc_auc_logistic = auc(tnr_list_logistic, tpr_list_logistic)

tpr_list_clf[-1]=0
tnr_list_clf[-1]=0
# 绘制ROC曲线
plt.figure(figsize=(12, 10))
plt.plot(tnr_list_clf, tpr_list_clf, color='darkorange', lw=2, label=f'Decision Tree (AUC = {roc_auc_clf:.2f})')
plt.plot(tnr_list_rdf, tpr_list_rdf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rdf:.2f})')
plt.plot(tnr_list_svm, tpr_list_svm, color='blue', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot(tnr_list_logistic, tpr_list_logistic, color='red', lw=2,
         label=f'Logistic Regression (AUC = {roc_auc_logistic:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()