import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings("ignore")

california=fetch_california_housing()
X=california.data
y=california.target

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

lin_reg=LinearRegression()
lin_reg.fit(X_train_scaled,y_train)

y_pred_lin=lin_reg.predict(X_test_scaled)

mse_lin=mean_squared_error(y_test, y_pred_lin)
r2_lin=r2_score(y_test, y_pred_lin)

print(f"Linear regression MSE:{mse_lin} R2:{r2_lin}")

"Linear regression MSE:0.5554933399188379 R2:0.6100057029068585"


ridge_params={"alpha":[0.1,1,10,100]}
ridge=Ridge()
ridge_grid=GridSearchCV(ridge, ridge_params,cv=5)
ridge_grid.fit(X_train_scaled,y_train)

y_pred_ridge=ridge_grid.predict(X_test_scaled)

mse_ridge=mean_squared_error(y_test, y_pred_ridge)
r2_ridge=r2_score(y_test, y_pred_ridge)

print(f"Ridge regression MSE:{mse_ridge} R2:{r2_ridge}")

"Ridge regression MSE:0.5555594993951625 R2:0.6099592544319425"

lasso_params={"alpha":[0.1,1,10,100]}
lasso=Lasso(max_iter=10000)
lasso_grid=GridSearchCV(lasso, lasso_params,cv=5)
lasso_grid.fit(X_train_scaled,y_train)

y_pred_lasso=lasso_grid.predict(X_test_scaled)

mse_lasso=mean_squared_error(y_test, y_pred_lasso)
r2_lasso=r2_score(y_test, y_pred_lasso)

print(f"Lasso regression MSE:{mse_lasso} R2:{r2_lasso}")

"Lasso regression MSE:0.6987006121351905 R2:0.5094644084337281"


elastic_params={"alpha":[0.1,1,10],"l1_ratio":[0.2,0.5,0.8]}
elastic=ElasticNet(max_iter=10000)
elastic_grid=GridSearchCV(elastic, elastic_params,cv=5)
elastic_grid.fit(X_train_scaled,y_train)

y_pred_elastic=elastic_grid.predict(X_test_scaled)

mse_elastic=mean_squared_error(y_test, y_pred_elastic)
r2_elastic=r2_score(y_test, y_pred_elastic)

print(f"elastic regression MSE:{mse_elastic} R2:{r2_elastic}")

"elastic regression MSE:0.6043212451624923 R2:0.5505846595427055"

models=["Linear Regression","Ridge Regression","Lasso Regression", "ElasticNet Regression"]
mses=[mse_lin,mse_ridge,mse_lasso,mse_elastic]
r2s=[r2_lin,r2_ridge,r2_lasso,r2_elastic]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.barh(models, mses, color="skyblue")
plt.xlabel("Mean Square Error")
plt.title("Model MSE Comparison")

plt.subplot(1,2,2)
plt.barh(models, r2s, color="lightgreen")
plt.xlabel("R2 Score")
plt.title("Model R2 Score Comparison")

plt.tight_layout()
plt.show()
