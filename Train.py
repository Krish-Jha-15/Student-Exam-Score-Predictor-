import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pickle


data_set=pd.read_csv('Data_Set/student_performance_data1.csv')
data_set.head()

data_set['Attendance_Percentage']=data_set['Attendance_Percentage'].fillna(
    data_set['Attendance_Percentage'].mean()
)

data_set['Sleep_Hours']=data_set['Sleep_Hours'].fillna(
    data_set['Sleep_Hours'].mean()
)
data_set['Screen_Time']=data_set['Screen_Time'].fillna(
    data_set['Screen_Time'].mean()
)

X=data_set[['Hours_Studied','Attendance_Percentage','Previous_Exam_Score','Assignments_Completed','Internet_Hours','Screen_Time','Class_Participation','Travel_Time_Minutes','Part_Time_Job']]
Y=data_set['Final_Score']

X['Part_Time_Job']=X['Part_Time_Job'].map(
    {
        'No':0,
        'Yes':1
    }
)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

Scaler=StandardScaler()
X_train_s=Scaler.fit_transform(X_train)
X_test_s=Scaler.transform(X_test)

Model_LR=LinearRegression()
Model_LR.fit(X_train_s,Y_train)
y_pred_LR=Model_LR.predict(X_test_s)

Model_Knn=KNeighborsRegressor()
Model_Knn.fit(X_train_s,Y_train)
y_pred_Knn=Model_Knn.predict(X_test_s)

Model_DT=DecisionTreeRegressor()
Model_DT.fit(X_train_s,Y_train)
y_pred_DT=Model_DT.predict(X_test_s)

print("Bias (Intercept):", Model_LR.intercept_)
print("Weights (Coefficients):", Model_LR.coef_)


R2_LR=r2_score(Y_test,y_pred_LR)
R2_Knn=r2_score(Y_test,y_pred_Knn)
R2_DT=r2_score(Y_test,y_pred_DT)

print('R2 of Linear-regression:',R2_LR)
print('R2 of KNN:',R2_Knn)
print('R2 of Decsion Tree:',R2_DT)

models = ["LR", "KNN", "DT"]
scores = [R2_LR,R2_Knn,R2_DT]

plt.bar(models, scores)
plt.title("Model Comparison")
plt.ylabel("R2 Score")

plt.savefig(
    "screenshots/model_comparison.png"
)

scores = {
    "Linear Regression": R2_LR,
    "KNN": R2_Knn,
    "Decision Tree": R2_DT
}

model_dict = {
    "Linear Regression": Model_LR,
    "KNN": Model_Knn,
    "Decision Tree": Model_DT
}

best_model_name = max(scores, key=scores.get)
best_model=model_dict[best_model_name]

print("Best Model:", best_model_name)

pickle.dump(
    best_model,
    open("Model/trained_model.pkl","wb")
)

pickle.dump(
    Scaler,
    open("Model/Scaler.pkl", "wb")
)