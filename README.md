# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:

### Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

__Developed by: SESHAN J__

__Register Number: 212224220092__

```py
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```

```py
print("data.info():")
data.info()
```

```py
print("isnull() and sum():")
data.isnull().sum()
```

```py
print("data value counts():")
data["left"].value_counts()
```

```py
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```

```py
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```

```py
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```

```py
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

```py
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

```py
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```
## Output:

<img width="1732" height="291" alt="image" src="https://github.com/user-attachments/assets/ad2a7195-13b8-455c-93c2-75a43a3b6702" />

<img width="1724" height="636" alt="image" src="https://github.com/user-attachments/assets/30eedfc9-87b2-4c7d-b547-89a6b1635108" />

<img width="1168" height="579" alt="image" src="https://github.com/user-attachments/assets/a13f6f60-c490-4525-ae32-67e25352696c" />

<img width="858" height="235" alt="image" src="https://github.com/user-attachments/assets/086cf289-a9f0-48ee-b73a-41d1d1413da0" />

<img width="1040" height="171" alt="image" src="https://github.com/user-attachments/assets/657efd1e-7a94-48b4-b282-5949bea706bb" />

<img width="1038" height="180" alt="image" src="https://github.com/user-attachments/assets/93928e16-e14c-489a-a7de-e902e5042403" />

<img width="994" height="55" alt="image" src="https://github.com/user-attachments/assets/abb53353-828b-497d-b8f1-83b50743248e" />

<img width="1032" height="47" alt="image" src="https://github.com/user-attachments/assets/ec74a3fe-a5c4-49d3-95ef-ad8e3c626145" />

<img width="814" height="604" alt="image" src="https://github.com/user-attachments/assets/05d1f352-b946-4052-9e24-8ae0886842fb" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
