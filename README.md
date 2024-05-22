# Cardiovascular-Risk-Prediction

**Introduction:**
Predicting coronary heart disease in advance helps raise awareness for the disease. Preventive measurements like changing diet plans and exercise can slow down the progression of CHD.
Early prediction can result in early diagnosis. So, we can treat the disease at an early stage and avoid more invasive treatment.

**Problem Statement:**
The goal is to predict whether the patient has a 10-year risk of future coronary heart disease (CHD).

**Project Description:**
The dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts.
The classification goal is to predict whether the patient has a 10-year risk of future coronary heart disease (CHD).
The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes. Each attribute is a potential risk factor. 
There are both demographic, behavioral, and medical risk factors.

**Demographic:**

*Sex:* male or female ("M" or "F")

*Age:* Age of the patient (Continuous)

*Education:* The level of education of the patient (categorical values - 1,2,3,4)

**Behavioral:**

*is_smoking:* whether or not the patient is a current smoker ("YES" or "NO")

*Cigs Per Day:* the number of cigarettes that the person smoked on average in one day.(continuous)

**Medical (history):**

*BP Meds:* whether or not the patient was on blood pressure medication (Nominal)

*Prevalent Stroke:* whether or not the patient had previously had a stroke (Nominal)

*Prevalent Hyp:* whether or not the patient was hypertensive (Nominal)

*Diabetes:* whether or not the patient had diabetes (Nominal)

**Medical (current):**

*Tot Chol:* total cholesterol level (Continuous)

*Sys BP:* systolic blood pressure (Continuous)

*Dia BP:* diastolic blood pressure (Continuous)

*BMI:* Body Mass Index (Continuous)

*Heart Rate:* heart rate (Continuous)

*Glucose:* glucose level (Continuous)

**Predict variable (desired target):**

*TenYearCHD:* 10-year risk of coronary heart disease CHD(binary: “1”, means “Yes”, “0” means “No”)


**Overview:**

This project aims to develop a machine learning model to predict the risk of heart disease in individuals based on various health factors and lifestyle attributes.
Heart disease is a significant public health concern globally, and early identification of individuals at high risk can facilitate proactive interventions to prevent adverse cardiovascular events.



**Project Steps:**

*Data Preprocessing:* Perform data cleaning and preprocessing tasks, including handling missing values, encoding categorical variables, and scaling numerical features as necessary.

*Exploratory Data Analysis (EDA):* Conduct exploratory analysis to understand the distribution of features, identify correlations between variables,
and gain insights into potential predictors of heart disease risk.

*Feature Selection:* Select relevant features that have the most significant impact on predicting heart disease risk. 
This may involve techniques such as statistical tests, feature importance analysis, or domain knowledge.

*Model Selection:* Evaluate different classification algorithms such as Logistic Regression, Decision Trees, Random Forests,Adaboost,Naive Bayes,Support Vector Machines (SVM), and K-nearest neigbhour (KNN) 
to determine the most suitable model for the task.

*Model Training:* Train the selected machine learning model on the training dataset, tuning hyperparameters as needed to optimize performance.

*Model Evaluation:* Evaluate the trained model's performance using appropriate metrics such as accuracy, Confusion Matrix, precision, recall, F1-score, and ROC-AUC curve.

*Model Interpretation:* Interpret the model's predictions and identify key factors contributing to the risk of heart disease. Visualize decision boundaries, 
feature importance scores, and any other relevant insights to explain the model's behavior.



















