import pickle
import numpy as np

model=pickle.load(
     open("Model/trained_model.pkl", "rb")
)

scaler=pickle.load(
     open("Model/Scaler.pkl", "rb")
)

#'','','','',''
hours=float(input("Hours_Studied"))
Attendance_Percentage=float(input("Attendance_Percentage(0 to 100)"))
Previous_Exam_Score=float(input("Previous_Exam_Score(0 to 100)"))
Internet_Hours=float(input("Internet_Hours"))
Screen_Time=float(input("Screen_Time"))
Class_Participation=float(input("Class_Participation"))
Travel_Time_Minutes=float(input("Travel_Time_Minutes"))
Part_Time_Job=input("Part_Time_Job(Yes/No)")
Assignments_Completed=float(input("Assignments_Completed"))

if Part_Time_Job.lower() == "yes":
    Part_Time_Job=1
else:
    Part_Time_Job=0

data=np.array([[

    hours,
    Attendance_Percentage,
    Previous_Exam_Score,
    Assignments_Completed,
    Internet_Hours,
    Screen_Time,
    Class_Participation,
    Travel_Time_Minutes,
    Part_Time_Job

]])

data_scaled=scaler.transform(data)

prediction=model.predict(data_scaled)


print(
    "\nPredicted Final Score:",
    round(prediction[0], 2)
)