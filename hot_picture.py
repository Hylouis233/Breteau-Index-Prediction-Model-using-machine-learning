import pandas as pd
import numpy as np

all_corr=pd.read_csv('all_corr.csv',index_col=0)
for i in range(0,all_corr.shape[0]):
    for j in range(0,all_corr.shape[1]):
        if i>j:
            all_corr.iloc[j,i]=all_corr.iloc[i,j]
all_corr_total=all_corr
all_corr_total.to_csv('all_corr_total.csv')

all_corr_total_nostar=pd.read_csv('all_corr_total_nostar.csv',index_col=0)
#制作热图
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(20,20),dpi=300)
sns.heatmap(all_corr_total_nostar,annot=False,cmap='rainbow',vmin=-1,vmax=1)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=8)
plt.title("All Variable Correlation Matrix",fontsize=12)
ax.xaxis.set_label_position('top')
plt.savefig('all_corr_right_front.tiff')
plt.show()



#BF
import pandas as pd
import numpy as np

all_corr=pd.read_csv('bf_corr.csv',index_col=0)
for i in range(0,all_corr.shape[0]):
    for j in range(0,all_corr.shape[1]):
        if i>j:
            all_corr.iloc[j,i]=all_corr.iloc[i,j]
all_corr_total=all_corr
all_corr_total.to_csv('bf_corr_total.csv')

all_corr_total_nostar=pd.read_csv('bf_corr_total_nostar.csv',index_col=0)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(20,20))
sns.heatmap(all_corr_total_nostar,annot=True,cmap='rainbow',vmin=-1,vmax=1)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=8)
plt.title("Random Forest Variable Correlation Matrix",fontsize=12)
plt.savefig('rf_corr_font.tiff')
plt.show()

#调整图片的坐标轴标签

#DT
all_corr=pd.read_csv('DT_corr.csv',index_col=0)
for i in range(0,all_corr.shape[0]):
    for j in range(0,all_corr.shape[1]):
        if i>j:
            all_corr.iloc[j,i]=all_corr.iloc[i,j]
all_corr_total=all_corr
all_corr_total.to_csv('DT_corr_total.csv')

all_corr_total_nostar=pd.read_csv('DT_corr_total_nostar.csv',index_col=0)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(20,20))
sns.heatmap(all_corr_total_nostar,annot=True,cmap='rainbow',vmin=-1,vmax=1)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=8)
plt.title("Decision Tree Variable Correlation Matrix",fontsize=12)
plt.savefig('DT_corr.tiff')
plt.show()

#ALL
all_corr=pd.read_csv('ML_corr.csv',index_col=0)
for i in range(0,all_corr.shape[0]):
    for j in range(0,all_corr.shape[1]):
        if i>j:
            all_corr.iloc[j,i]=all_corr.iloc[i,j]
all_corr_total=all_corr
all_corr_total.to_csv('ML_corr_total.csv')

all_corr_total_nostar=pd.read_csv('ML_corr_total_nostar.csv',index_col=0)
#制作热图
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(20,20))
sns.heatmap(all_corr_total_nostar,annot=False,cmap='rainbow',vmin=-1,vmax=1)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=8)
plt.title("Screened Variables Union Correlation Matrixx",fontsize=12)
plt.savefig('Machine_Learning_corr.tiff')
plt.show()