import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

digits=load_digits()
X,y=digits.data,digits.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

pca=PCA(n_components=0.95)
X_train_pca=pca.fit_transform(X_train_scaled)
X_test_pca=pca.transform(X_test_scaled)

rf_params={"n_estimators":[50,100,200]}
rf=RandomForestClassifier(random_state=42)
rf_grid=GridSearchCV(rf, rf_params,cv=5)
rf_grid.fit(X_train_pca, y_train)

svm_params={"C":[0.1,1,10],"kernel":["linear","rbf"]}
svm=SVC()
svm_grid=GridSearchCV(svm, svm_params,cv=5)
svm_grid.fit(X_train_pca, y_train)

knn_params={"n_neighbors":[3,5,7]}
knn=KNeighborsClassifier()
knn_grid=GridSearchCV(knn, knn_params,cv=5)
knn_grid.fit(X_train_pca, y_train)


best_svm=svm_grid.best_estimator_
best_rf=rf_grid.best_estimator_
best_knn=knn_grid.best_estimator_

voting_clf=VotingClassifier(estimators=[("svm",best_svm),("rf",best_rf),("knn",best_knn)],
                            voting="hard")

voting_clf.fit(X_train_pca,y_train)
y_pred=voting_clf.predict(X_test_pca)

cm=confusion_matrix(y_test, y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=digits.target_names )
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(f"voting classifier accuracy: {voting_clf.score(X_test_pca,y_test)}")

"""
 voting classifier accuracy: 0.9777777777777777
"""
