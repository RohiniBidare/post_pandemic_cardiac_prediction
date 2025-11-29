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

Classification Report:
Train Accuracy: 0.95
Test Accuracy: 0.6133333333333333
Streamlit Page :
![WhatsApp Image 2025-11-24 at 00 53 00_7f9b7fde](https://github.com/user-attachments/assets/78a22a50-f494-4fe7-9215-a0b549799e15)

![WhatsApp Image 2025-11-24 at 00 53 00_9f9b58aa](https://github.com/user-attachments/assets/e32f4390-b523-4452-ad79-875d76178738)

