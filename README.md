Post-Pandemic Cardiac Disease Prediction : 

A Machine Learning project to predict heart disease, mortality rate, and disease classification in vaccinated adults after the COVID-19 pandemic.

Technologies & Libraries Used :

.NumPy – numerical operations
.Pandas – dataset loading & preprocessing
.Matplotlib / Seaborn – data visualization
.Scikit-learn (sklearn) – ML algorithms, evaluation metrics
.RandomForestClassifier
.RandomForestRegressor
.Train-test split

The dataset includes post-pandemic heart-health parameters such as:

Patient Details :
Age
Sex (0 = Female, 1 = Male)
Clinical Features
Chest Pain Type (cp) – 1 to 4
Resting Blood Pressure (trestbps)
Cholesterol (chol)
Fasting Blood Sugar (fbs)
Resting ECG (restecg)
Maximum Heart Rate (thalach)
Exercise Induced Angina (exang)
ST Depression (oldpeak)
Slope of ST segment (slope)
Major Vessels (ca)
Thalassemia (thal)
Pandemic-Related Inputs :
Vaccine Dose 1 (0/1)
Vaccine Dose 2 (0/1)
Mortality Rate (%)

Output Label for Classification :

(Heart disease type)
0. No Heart Disease
1.Coronary artery disease
2.Heart Failure
3.Arrhythmias
4.Valvular Heart Disease
5.Cardiomyopathy
6.Congenital Heart Disease
7.Pericarditis
8.Myocarditis
9.Hypertensive Heart Disease
10.Rheumatic Heart Disease
