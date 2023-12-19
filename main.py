import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import graphviz
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge

data=pd.read_csv('Siming_District_Breteau_Index_season.csv')
data['date']=pd.to_datetime(data['date'])


data_select_1=data[['breteau_index','weather','door','bonsai','tank','containers',
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
                    'pressure2','pressure1','wind','wind7','wind6','wind5','wind4','wind3','wind2','wind1'
]]
data_select_1=data_select_1.dropna(axis=0,how='any')

data_select_2=data[['bonsai','tank','containers',
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

#选择数据集
data=data_select_2

clf=DecisionTreeRegressor()
# rdf=RandomForestRegressor(min_weight_fraction_leaf= 0.0, min_samples_split= 10, min_samples_leaf= 2, min_impurity_decrease= 0.1, max_depth= 30, ccp_alpha= 0.0)
# rdf=RandomForestRegressor(min_weight_fraction_leaf= 0.0, min_samples_split= 10, min_samples_leaf= 2, min_impurity_decrease= 0.1, max_depth= 30, ccp_alpha= 0.2)
# rdf=joblib.load('rdf_siming.pkl')
rdf=RandomForestRegressor()
from sklearn.model_selection import (RandomizedSearchCV)


#data=原始data
# Y=data.iloc[:,5:6]
# X=data.iloc[:,6:]

#dataselect后：
Y=data.iloc[:,0:1]
X=data.iloc[:,1:]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
Y_train=Y_train.values.ravel()
Y_test=Y_test.values.ravel()
clf.fit(X_train,Y_train)
rdf.fit(X_train,Y_train)
Y_pred_clf=clf.predict(X_test)
Y_pred_rdf=rdf.predict(X_test)
Y_train_pred_clf=clf.predict(X_train)
Y_train_pred_rdf=rdf.predict(X_train)


# # 定义超参数的范围
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
#     estimator=RandomForestRegressor(),
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
#
#
# # 打印最佳参数组合和对应的性能指标
# print("Best parameters:", random_search.best_params_)
# print("Best negative mean squared error:", random_search.best_score_)
# # Best parameters: {'min_weight_fraction_leaf': 0.0, 'min_samples_split': 10, 'min_samples_leaf': 2, 'min_impurity_decrease': 0.1, 'max_depth': 30, 'ccp_alpha': 0.0}
# # Best negative mean squared error: -103.89750584684889
# #dataselect_2:
# # min_weight_fraction_leaf= 0.0, min_samples_split= 10, min_samples_leaf= 2, min_impurity_decrease= 0.1, max_depth= 30, ccp_alpha= 0.2
# # Best negative mean squared error: -128.61055908296203

#R2
print('clf_test:',r2_score(Y_test,Y_pred_clf))
print('rdf_test:',r2_score(Y_test,Y_pred_rdf))

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

df_clf.to_csv('clf_siming.csv')
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
df_rdf.to_csv('rdf_siming.csv')



#多项式拟合
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_test)
score=-99999
for i in np.arange(0,10,0.01):
    ridge = Ridge(alpha=i)
    ridge.fit(X_poly, Y_train)
    Y_pred_poly_reg=ridge.predict(X_poly_test)
    score_ridge=ridge.score(X_poly_test,Y_test)
    if score_ridge>score:
        score=score_ridge
        Y_pred_poly_reg_best=Y_pred_poly_reg
        ridge_best=ridge
    print(ridge_best)

#返回岭回归参数Ridge(alpha=0.46)
df_ridge=pd.DataFrame(ridge_best.coef_)
df_ridge.to_csv('ridge_siming.csv')
Y_train_poly_reg_best=ridge_best.predict(X_poly)



plt.figure(figsize=(10,10))
# plt.scatter(Y_test,Y_pred_clf,color='red',label='DecisionTreeRegressor')
# plt.scatter(Y_test,Y_pred_rdf,color='green',label='RandomForestRegressor_test')
# plt.scatter(Y_train,Y_train_pred_rdf,color='blue',label='RandomForestRegressor_train')
# plt.scatter(Y_test,Y_pred_clf,color='green',label='DecisionTreeRegressor_test')
# plt.scatter(Y_train,Y_train_pred_clf,color='blue',label='DecisionTreeRegressor_train')
plt.scatter(Y_test,Y_pred_poly_reg_best,color='green',label='PolynomialFeatures')
plt.scatter(Y_train,Y_train_poly_reg_best,color='blue',label='PolynomialFeatures_train')
plt.plot([0,100],[0,100],color='thistle',linestyle='dotted')
plt.legend()
plt.show()
#多项式R2
print('poly_test:',r2_score(Y_test,Y_pred_poly_reg_best))

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


