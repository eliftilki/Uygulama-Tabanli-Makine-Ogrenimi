import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import missingno as msno
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import precision_score, confusion_matrix

from sklearn import tree

df=pd.read_csv("water_potability.csv")

describe=df.describe()

df.info();

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3276 entries, 0 to 3275
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   ph               2785 non-null   float64
 1   Hardness         3276 non-null   float64
 2   Solids           3276 non-null   float64
 3   Chloramines      3276 non-null   float64
 4   Sulfate          2495 non-null   float64
 5   Conductivity     3276 non-null   float64
 6   Organic_carbon   3276 non-null   float64
 7   Trihalomethanes  3114 non-null   float64
 8   Turbidity        3276 non-null   float64
 9   Potability       3276 non-null   int64  
dtypes: float64(9), int64(1)
memory usage: 256.1 KB
"""

d = df["Potability"].value_counts().reset_index()  # DataFrame'e çevir
d.columns = ["Potability", "count"]  # Sütun isimlerini düzenle

fig = px.pie(d, values="count", names="Potability", hole=0.35, opacity=0.8,
             labels={"Potability": "Potability", "count": "Number of Samples"})

fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside", textinfo="percent+label")
fig.show()

fig.write_html("potability_pie_chart.html")
sns.clustermap(df.corr(), cmap="vlag", dendrogram_ratio=(0.1, 0.2), annot=True, 
               linewidths=0.8, figsize=(10,10))
plt.show()

non_potable=df.query("Potability==0")
potable=df.query("Potability==1")

plt.figure()
for ax,col in enumerate(df.columns[:9]):
    plt.subplot(3,3,ax+1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col],label="Non Potable")
    sns.kdeplot(x=potable[col],label="Potable")
    plt.legend()
    
plt.tight_layout() 

msno.matrix(df)
plt.show()   

print(df.isnull().sum())

df["ph"] = df["ph"].fillna(value=df["ph"].mean())
df["Sulfate"] = df["Sulfate"].fillna(value=df["Sulfate"].mean())
df["Trihalomethanes"] = df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].mean())



print(df.isnull().sum())


x=df.drop("Potability",axis=1).values
y=df["Potability"].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

x_train_max=np.max(x_train)
x_train_min=np.min(x_train)
x_train=(x_train-x_train_min)/(x_train_max-x_train_min)
x_test=(x_test-x_train_min)/(x_train_max-x_train_min)

models=[("DTC",DecisionTreeClassifier(max_depth=3)),("RF",RandomForestClassifier())]

finalResult=[]
cmList=[]

for name,model in models:
    model.fit(x_train,y_train)
    model_result=model.predict(x_test)
    score=precision_score(y_test, model_result)
    finalResult.append((name,score))
    cm=confusion_matrix(y_test, model_result)
    cmList.append((name,cm))
    
print(finalResult)
for name,i in cmList:
    plt.figure()
    sns.heatmap(i,annot=True,linewidths=0.8,fmt=".0f")
    plt.title(name)
    plt.show()
    
dt_clf=models[0][1]
plt.figure(figsize=(25,20))
tree.plot_tree(dt_clf,feature_names=df.columns.tolist()[:-1],
               class_names=["0","1"],
               filled=True,
               precision=5)

plt.show()

model_params={
    "Ranfom Forest":
        {
            "model":RandomForestClassifier(),
            "params":
                {
                    "n_estimators":[10,50,100],
                    "max_features":["auto","sqrt","log2"],
                    "max_depth":list(range(1,21,3))}}}
    
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=2)
scores=[]
for model_name,params in model_params.items():
    rs=RandomizedSearchCV(params["model"],params["params"],cv=cv,n_iter=10)
    rs.fit(x,y)
    scores.append([model_name,dict(rs.best_params_),rs.best_score_])
    
print(scores)

"""
[['Ranfom Forest', {'n_estimators': 100, 'max_features': 'sqrt', 
   'max_depth': 19}, 0.6747572612176502]]
"""
