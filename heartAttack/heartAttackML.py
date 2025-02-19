import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,roc_curve

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("heart.csv")

describe=df.describe()

print(df.info())

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trtbps    303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalachh  303 non-null    int64  
 8   exng      303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slp       303 non-null    int64  
 11  caa       303 non-null    int64  
 12  thall     303 non-null    int64  
 13  output    303 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 33.3 KB
None
"""

print(df.isnull().sum())

"""
age         0
sex         0
cp          0
trtbps      0
chol        0
fbs         0
restecg     0
thalachh    0
exng        0
oldpeak     0
slp         0
caa         0
thall       0
output      0
"""

"""
Index(['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'],
      dtype='object')
"""

categorical_list=['sex', 'cp','fbs', 'restecg','exng','slp','caa', 'thall', 'output']

df_categorical=df.loc[:,categorical_list]

for i in df_categorical:
    plt.figure()
    sns.countplot(x=i,data=df_categorical,hue="output")
    plt.title(i)

numeric_list=['age','trtbps','chol','thalachh','oldpeak','output']
df_numeric=df.loc[:,numeric_list]
sns.pairplot(df_numeric,hue="output",diag_kind="kde")
plt.show()

scaler=StandardScaler()
scaled_array=scaler.fit_transform(df[numeric_list[:-1]])

df_dummy=pd.DataFrame(scaled_array,columns=numeric_list[:-1])
df_dummy=pd.concat([df_dummy,df.loc[:,"output"]],axis=1)

data_melted=pd.melt(df_dummy,id_vars="output",var_name="features",value_name="value")

plt.figure()
sns.boxplot(x="features",y="value",hue="output",data=data_melted)
plt.show()

plt.figure()
sns.swarmplot(x="features",y="value",hue="output",data=data_melted)
plt.show()

plt.figure()
sns.catplot(x="exng",y="age",hue="output",col="sex",kind="swarm",data=df)
plt.show()

plt.figure()
sns.heatmap(df.corr(),annot=True,fmt=".1f",linewidths=0.7)
plt.show()

numeric_list=['age','trtbps','chol','thalachh','oldpeak']
df_numeric=df.loc[:,numeric_list]

for i in numeric_list:
    Q1=np.percentile(df.loc[:,i],25)
    Q3=np.percentile(df.loc[:,i],75)
    
    IQR=Q3-Q1
    
    print(f"{i} old shape{df.loc[:,i].shape}")
    
    upper=np.where(df.loc[:,i]>=(Q3+2.5*IQR))
    lower=np.where(df.loc[:,i]<=(Q1-2.5*IQR))
    
    try:
        df.drop(upper[0],inplace=True)
    except:
        print("hata")
        
    try:
        df.drop(lower[0],inplace=True)
    except:
        print("hata")
        
    print(f"new shape{df.shape}")  
    
df1=  df.copy()

df1=pd.get_dummies(df1,columns=categorical_list[:-1],drop_first=True)

x=df1.drop(["output"],axis=1)
y=df1[["output"]]

scaler= StandardScaler()
x[numeric_list[:-1]]=scaler.fit_transform(x[numeric_list[:-1]])  

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred_prob=logreg.predict_proba(x_test)
y_pred=np.argmax(y_pred_prob,axis=1)   

print(f"test accuracy: {accuracy_score(y_test,y_pred)}")

fpr,tpr,threshold=roc_curve(y_test, y_pred_prob[:,1])

plt.plot([0,1],[0,1],"k--")
plt.plot(fpr, tpr,label="Logistic Regression")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("Logistic Regression ROC Curve")

lr=LogisticRegression()
penalty=["l1","l2"]

parameters={"penalty":penalty}

lr_searcher=GridSearchCV(lr, parameters) 
lr_searcher.fit(x_train,y_train)

print(f"best parameters: {lr_searcher.best_params_}")
y_pred=lr_searcher.predict(x_test)
print(f"test accuracy: {accuracy_score(y_test,y_pred)}")
