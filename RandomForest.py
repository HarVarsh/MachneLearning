from sklearn import datasets

iris=datasets.load_iris()
print(iris.target_names)

print(iris.feature_names)

X,y=datasets.load_iris( return_X_y = True ) 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)


from sklearn.ensemble import RandomForestClassifier

import pandas as pd

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
print("ACCURACY OF THE MODEL:",metrics.accuracy_score(y_test, y_pred))