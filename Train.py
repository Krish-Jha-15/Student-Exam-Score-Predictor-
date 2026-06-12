import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error,mean_squared_error

data_set=pd.read_csv('Data_Set/student_performance_data1.csv')
data_set.head()

# print(data_set.isnull().sum())

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
Y=data_set[['Final_Score']]

X['Part_Time_Job']=X['Part_Time_Job'].map(
    {
        'No':0,
        'Yes':1
    }
)

#print(X[X.isnull().any(axis=1)])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

Scaler=StandardScaler()
X_train_s=Scaler.fit_transform(X_train)
X_test_s=Scaler.transform(X_test)

Model=LinearRegression()
Model.fit(X_train_s,Y_train)
y_pred=Model.predict(X_test_s)
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': Model.coef_[0]
})

print(coef_df.sort_values(by='Coefficient', ascending=False))
print(data_set.corr(numeric_only=True)['Final_Score'].sort_values(ascending=False))

print("Bias (Intercept):", Model.intercept_)
print("Weights (Coefficients):", Model.coef_)

print('R2:',r2_score(Y_test,y_pred))
print('MAE:',mean_absolute_error(Y_test,y_pred))
print('MSE:',mean_squared_error(Y_test,y_pred))

