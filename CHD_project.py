#!/usr/bin/env python
# coding: utf-8

# - Project Description:
#  
# The dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts.
# The classification goal is to predict whether the patient has a 10-year risk of future coronary heart disease (CHD). 
# The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes.
# Each attribute is a potential risk factor. There are both demographic, behavioral, and medical risk factors.
# 
# 

# Demographic:
# 
# - Sex: male or female ("M" or "F")
# 
# - Age: Age of the patient (Continuous)
# 
# - Education: The level of education of the patient (categorical values - 1,2,3,4)
# 
# Behavioral:
# 
# - is_smoking: whether or not the patient is a current smoker ("YES" or "NO")
# - Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(continuous)
#     
# Medical (history):
# 
# - BP Meds: whether or not the patient was on blood pressure medication (Nominal)
# - Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
# - Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
# - Diabetes: whether or not the patient had diabetes (Nominal)
#     
# Medical (current):
# 
# - Tot Chol: total cholesterol level (Continuous)
# - Sys BP: systolic blood pressure (Continuous)
# - Dia BP: diastolic blood pressure (Continuous)
# - BMI: Body Mass Index (Continuous)
# - Heart Rate: heart rate (Continuous)
# - Glucose: glucose level (Continuous)
# 
# Predict variable (desired target):
# 
# - TenYearCHD: 10-year risk of coronary heart disease CHD(binary: “1”, means “Yes”, “0” means “No”)

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # Data Gathering

# In[27]:


df= pd.read_csv('C:\\Data Science\\data_cardiovascular_risk.csv')
df


# In[28]:


df['TenYearCHD'].value_counts()


# values count shows that the data is imbalanced

# In[29]:


df.shape


# In[30]:


df.describe(include= 'all')


# In[31]:


# Dropping the id column (not relevant)
df.drop(columns=['id'], inplace=True)


# In[32]:


df.head()


# In[33]:


# Checking Dataset Duplicate Value Count
df.duplicated().sum()


# In[34]:


# Checking missing values/null values count for each column
df.isnull().sum()


# In[35]:


a = df.isnull().sum().reset_index()
sns.pointplot(x ='index', y = 0, data = a )
plt.xticks(rotation=90)


# from the above graph we have seen that the variable Glucose contain more number of null values

# In[36]:


# Check Unique Values for each variable using a for loop
for i in df.columns.tolist():
  print("No. of unique values in",i,"is",df[i].nunique())


# In[37]:


# Defining 3 lists containing the column names of
# This is defined based on the number of unique values for each attribute
dependent_var = ['TenYearCHD']

categorical_var = [i for i in df.columns if df[i].nunique()<=4]
continuous_var = [i for i in df.columns if i not in categorical_var]


# In[38]:


categorical_var


# In[39]:


continuous_var


# # EDA

# In[40]:


# Dependent variable


# In[41]:


#color palette selection
colors = sns.color_palette("Paired")

# plotting data on chart
plt.figure(figsize=(10,6))
explode = [0,0.1]
textprops = {'fontsize':11}
plt.pie(df['TenYearCHD'].value_counts(), labels=['Not CHD(%)','CHD(%)'], startangle=90, colors=colors, explode = explode, autopct="%1.1f%%",textprops = textprops)
plt.title('Ten Year CHD (%)', fontsize=15)
plt.show()


# # Interpretation:
# From the above chart we come to know that 15.1% are classified as positive for 10 year CHD whereas
# the remaining 84.9%  are classified as negative for 10 year CHD.

# In[181]:


## catagorical variable


# In[182]:


sns.countplot(data=df,x='education')
plt.title("education distribution")


# In[183]:


sns.countplot(data=df,x='sex')
plt.title("sex distribution")


# In[184]:


sns.countplot(data=df,x='is_smoking')
plt.title('smoking distribution')


# In[185]:


sns.countplot(data=df,x='BPMeds')
plt.title('BPMeds distribution')


# In[186]:


sns.countplot(data=df,x='prevalentStroke')
plt.title('prevalentStroke distribution')


# In[187]:


sns.countplot(data=df,x='prevalentHyp')
plt.title('prevalentHyp  distribution')


# In[23]:


sns.countplot(data=df,x='diabetes')
plt.title('diabetes distribution')


# In[24]:


sns.countplot(data=df,x='TenYearCHD')
plt.title('TenYearCHD  distribution')


# The distribution of different categories in the categorical columns can be seen.
# The education column has the highest for the 1 category followed by 2 3 and 4.
# The gender distribution is not even with high count for females. 
# The 'is_smoking' column is showing same result. 
# Bp_meds, prevalent_stroke, prevalent_hyp and diabetes are imbalanced, they have very few counts for the positive cases.
# Finally the ten_year_chd is also imbalanced with few positive cases compared to the negative cases.

# In[25]:


df.head()


# In[26]:


# Visualizing by pie chart
Male=df[df["sex"]=='M'].sum()
Female=df[df["sex"]=='F'].sum()

# Set labels
Sex_grp={"Male":Male["TenYearCHD"],"Female":Female["TenYearCHD"]}
plt.gcf().set_size_inches(7,7)
plt.pie(Sex_grp.values(),labels=Sex_grp.keys(), explode=[0.05, 0.05], autopct ='%1.1f%%', shadow = True);
plt.title("Repartition of sex group", fontsize=15)
plt.show()


# From the above chart, we got to know that the gender distribution is not even with high count for females. 
# 53.2% ratio are there for males and 46.8% ratio for females.

# In[27]:


# Ploting Bar Chart
# Group by Age and get average CHD for 10 year, and precent change
avg_CHD = df.groupby('age')['TenYearCHD'].mean().reset_index()

# Plot average CHD over different age
plt.figure(figsize=(10,5))
ax = sns.barplot(x= avg_CHD['age'], y= avg_CHD['TenYearCHD'])
ax.set_ylabel("Ten Year CHD")
ax.set_xlabel("Age")
ax.set_title('Average Ten Year CHD vs Age')
plt.show()


# From above bar plot we can clearly see that the average CHD is high for above 65+ aged peoples. But a sudden drop in 67 and 69 year old group (CHD value is low there). And also for below 65 year, the CHD is much less.

# In[ ]:





# In[28]:


#continuous variables 


# In[248]:


g = sns.FacetGrid(df, col="TenYearCHD")
g.map(sns.histplot, "age");


# people having age group 50-55 have greater risk of 10 years of CHD

# sns.violinplot(data=df,x='TenYearCHD',y='age',palette='colorblind')

# In[29]:


sns.violinplot(data=df,x='TenYearCHD',y='BMI',palette='colorblind')


# In[30]:


sns.violinplot(data=df,x='TenYearCHD',y='age',palette='colorblind')


# In[31]:


sns.violinplot(data=df,x='TenYearCHD',y='glucose',palette='colorblind')


# In[32]:


sns.violinplot(data=df,x='TenYearCHD',y='heartRate',palette='colorblind')


# In[33]:


sns.violinplot(data=df,x='TenYearCHD',y='cigsPerDay',palette='colorblind')


# For age vs ten_year_chd, we see that the density for positive cases is high at higher age as compared to lower age
# indicating that the positive cases are higher in older people.
# 
# For cigs_per_day, the negative cases are more for the non smokers compared to the positive cases for non smokers.
# 
# For ten_year_chd and glucose, the negative cases have high density compared to the positive cases for the same value of glucose.
# 
# The remaining charts do not provide much information.

# In[34]:


# Plot for Ten year CHD for smoking cigarette
sns.countplot(x='is_smoking',hue='TenYearCHD',data=df)
plt.ylabel("Ten Year CHD")
plt.xlabel("Smoking")
plt.title('CHD vs Smoking Cigarette')
plt.show()


# From above barplot we got to know that There is low chances of CHD for non smokers compare to smoking persons.

# In[35]:


plt.figure(figsize=(15,15))
for i in range(0, len(continuous_var)):
    plt.subplot(5,3,i+1)
    sns.kdeplot(df[continuous_var[i]])


# In[36]:


plt.figure(figsize=(15,15))
for i in range(0, len(continuous_var)):
    plt.subplot(5,3,i+1)
    sns.scatterplot(df[continuous_var[i]])


# In[37]:


plt.figure(figsize=(15,22))
for i in range(0, len(continuous_var)):
    plt.subplot(2,4,i+1)
    sns.boxplot(df[continuous_var[i]])
    plt.xlabel(continuous_var[i])


# From above barplot we got to know that There are more outliers in glucose column.

# In[ ]:


# Pair Plot visualization code
sns.pairplot(df, hue="TenYearCHD")
plt.show()


# In[154]:


corr = df[continuous_var].corr()
mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(14, 7))
    ax = sns.heatmap(corr , mask=mask, vmin = -1,vmax=1, annot = True, cmap="YlGnBu")


# # Feature Engineering & Data Pre-processing

# In[42]:


# Checking missing values/null values count for each column
df.isnull().sum()


# # Replacing null values

# In[43]:


df['education'] = df['education'].fillna(df['education'].mode()[0])
df['BPMeds'] =df['BPMeds'].fillna(df['BPMeds'].mode()[0])


# In[44]:


df['cigsPerDay'] = df['cigsPerDay'].fillna(df['cigsPerDay'].median())
df['totChol'] = df['totChol'].fillna(df['totChol'].median())
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['heartRate'] = df['heartRate'].fillna(df['heartRate'].median())
df['glucose'] = df['glucose'].fillna(df['glucose'].median())


# In[45]:


df.head()


# In[46]:


df.isnull().sum()


# # Encoding

# In[47]:


# Replacing the string values of the binary column with 0 and 1

df['sex'] = np.where(df['sex'] == 'M',1,0)
df['is_smoking'] = np.where(df['is_smoking'] == 'YES',1,0)


# In[48]:


df.info()


# In[49]:


df = pd.get_dummies(df, columns=['education'])


# In[50]:


df.head()


# In[51]:


df['education_1.0'] = np.where(df['education_1.0'] == 'False',0,1)


# In[52]:


df['education_2.0'] = np.where(df['education_2.0'] == 'False',0,1)


# In[53]:


df['education_3.0'] = np.where(df['education_3.0'] == 'False',0,1)


# In[54]:


df['education_4.0'] = np.where(df['education_4.0'] == 'False',0,1)


# In[55]:


df.head()


# # Train Test Split

# In[56]:


x=df.drop(['TenYearCHD'],axis=1)


# In[57]:


y=df['TenYearCHD']


# In[58]:


from sklearn.model_selection import train_test_split, GridSearchCV , RandomizedSearchCV


# In[59]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=1,stratify = y)


# # Model Training

# # 1) Logistic Regression

# In[60]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score


# In[61]:


log = LogisticRegression()
log.fit(x_train,y_train)


# In[66]:


# Testing data evaluation
y_pred_test = log.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('confusion matrix :\n',cnf_matrix)

acc_score_LR = accuracy_score(y_test,y_pred_test)
print("Accuracy Score : ",acc_score_LR)

clf_report = classification_report(y_test , y_pred_test)
print('classification_report : \n',clf_report)



# In[67]:


# Training data evaluation
y_pred_train = log.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score_LR = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score_LR)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# In[68]:


print("Accuracy Score :  0.8547197640117994")


# # Feature scaling

# In[69]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler


# In[70]:


std_scaler = StandardScaler()
std_scaler.fit(x_train)
std_scaler.fit_transform(x_train)


# In[71]:


std_scaler.transform(x_test)


# # 2) K- Nearest Neighbor
# 

# In[72]:


from sklearn.neighbors import KNeighborsClassifier


# In[73]:


knn_model = KNeighborsClassifier()
knn_model.fit(x_train,y_train)


# In[74]:


# Testing data evaluation
y_pred_test= knn_model.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('confusion matrix :\n',cnf_matrix)

acc_score_knn = accuracy_score(y_test,y_pred_test)
print("Accuracy Score : ",acc_score_knn)

clf_report = classification_report(y_test , y_pred_test)
print('classification_report : \n',clf_report)



# In[75]:


# Training data evaluation
y_pred_train= knn_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score_LR = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score_knn)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# In[76]:


print("Accuracy Score :  0.8451327433628318")


# # 3) SVM

# In[219]:


from sklearn.svm import SVC


# In[220]:


svc_model = SVC()
svc_model.fit(x_train,y_train)


# In[221]:


# testing data evaluation
y_pred_test= svc_model.predict(x_test)

acc= accuracy_score(y_test,y_pred_test)
print('Accuracy is:', acc)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('Confusion Matrix:\n', cnf_matrix)

Clf = classification_report(y_test,y_pred_test)
print('Classification Report:\n', Clf)


# In[222]:


# training data evaluation
y_pred_train = svc_model.predict(x_train)

acc= accuracy_score(y_train,y_pred_train)
print('Accuracy is:', acc)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('Confusion Matrix', cnf_matrix)

Clf = classification_report(y_train,y_pred_train)
print('Classification Report', Clf)


# # Hyperparameter Tunning

# In[223]:


svm_clf =SVC()
hyp = {'C': np.arange(0,10,0.1),
      'kernel':['rbf'],
      'gamma': np.arange(0,10,0.1)}

rscv_svm = RandomizedSearchCV(svm_clf, hyp, cv=3)
rscv_svm.fit(x_train, y_train)
rscv_svm.best_params_


# In[224]:


rscv_svm.best_estimator_


# In[225]:


# training data evaluation
y_pred_train = rscv_svm.predict(x_train)

acc= accuracy_score(y_train,y_pred_train)
print('Accuracy is:', acc)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('Confusion Matrix', cnf_matrix)

Clf = classification_report(y_train,y_pred_train)
print('Classification Report', Clf)


# # 4) Decision Tree

# In[77]:


from sklearn.tree import DecisionTreeClassifier


# In[78]:


dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)


# In[79]:


# Testing data evaluation
y_pred_test = dt_model.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('confusion matrix :\n',cnf_matrix)

acc_score_DT = accuracy_score(y_test,y_pred_test)
print("Accuracy Score : ",acc_score_DT)

clf_report = classification_report(y_test , y_pred_test)
print('classification_report : \n',clf_report)


# In[80]:


# Training data evaluation
y_pred_train= dt_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# since the training dataset accuracy is 1 hence the model is overfit so we use hyperparametric tunning

# # Hyperparameter tunning
overfitting (99 vs 72)

hyperparameter tuning()
max_depth
min_sample_leaf
min_sample_split

pruning

# In[81]:


dt_model=DecisionTreeClassifier()
hyp = {'criterion':['gini','entropy'],
       'max_depth':np.arange(2,10),
       'min_samples_split':np.arange(2,10),
       'min_samples_leaf':np.arange(2,10)
        
      
      }
rscv_model = RandomizedSearchCV(dt_model,hyp,random_state=10,cv=5)
rscv_model.fit(x_train,y_train)

rscv_model.best_estimator_
# In[82]:


# Training data evaluation
y_pred_train= rscv_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# In[83]:


print("Accuracy Score :  0.849188790560472")


# # 5)Random Forest
# 

# In[84]:


from sklearn.ensemble import RandomForestClassifier


# In[85]:


rf_model = RandomForestClassifier()

rf_model.fit(x_train,y_train)


# In[86]:


rf_model=rscv_model.best_estimator_
rf_model.fit(x_train,y_train)


# In[87]:


# Testing data evaluation
y_pred_test = rf_model.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('confusion matrix :\n',cnf_matrix)

acc_score_RF = accuracy_score(y_test,y_pred_test)
print("Accuracy Score : ",acc_score_RF)

clf_report = classification_report(y_test , y_pred_test)
print('classification_report : \n',clf_report)


# In[88]:


# Training data evaluation
y_pred_train= rf_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score_rf = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score_rf)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# # Hyperparametric Tunning

# In[89]:


rf_model = RandomForestClassifier()
hyp = hyp = {'criterion':['gini','entropy'],
       'max_depth':np.arange(2,10),
       'min_samples_split':np.arange(2,10),
       'min_samples_leaf':np.arange(2,10),
             'n_estimators':np.arange(60,100),
             
      
      }
rscv_model = RandomizedSearchCV(rf_model,hyp,random_state=52,cv=5)
rscv_model.fit(x_train,y_train)


# In[90]:


rscv_model.best_estimator_


# In[91]:


# Training data evaluation
y_pred_train= rscv_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score_rf = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score_rf)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# # 6) Adaboost

# In[122]:


from sklearn.ensemble import AdaBoostClassifier


# In[236]:


ada_boost = AdaBoostClassifier(n_estimators=1,random_state=1)
ada_boost.fit(x_train,y_train)


# In[237]:


# Testing data evaluation

y_pred_test = ada_boost.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('Confusion Matrix is\n', cnf_matrix)

acc = accuracy_score(y_test,y_pred_test)
print('Accuracy is', acc)

clf_report= classification_report(y_test,y_pred_test)
print('Classification report', clf_report)


# In[238]:


# Training data evaluation

y_pred_train = ada_boost.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('Confusion Matrix is\n', cnf_matrix)

acc = accuracy_score(y_train,y_pred_train)
print('Accuracy is', acc)

clf_report= classification_report(y_train,y_pred_train)
print('Classification report', clf_report)


# In[240]:


ada_boost.feature_importances_


# In[241]:


s1 = pd.Series(ada_boost.feature_importances_, index=x.columns)
s1.sort_values().plot(kind ='barh')

From Adaboost Age is factor mostly affecting CHD
# # Naive Bayes

# In[243]:


import os
import glob
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[245]:


def evaluation(string, model, ind_var, act):
    print(string.center(50,'*'))
    pred = model.predict(ind_var)
    cnf_matrix = confusion_matrix(act,pred)
    print('Confusion Matrix :\n', cnf_matrix)
    
    accuracy = accuracy_score(act,pred)
    print('Accuracy :', accuracy)
    
    clf_report = classification_report(act, pred)
    print('Classification Report :\n', clf_report)


# # 1 . Gaussian NB

# In[246]:


gnb_model = GaussianNB()
gnb_model.fit(x_train,y_train)


# In[44]:


print('GaussianNB Model Results'.center(80,'*'))
evaluation('Train Data Evaluation', gnb_model, x_train, y_train)
print('*#'*30)
evaluation('Test Data Evaluation', gnb_model, x_test, y_test)


# # 2. Multinomial NB

# In[45]:


mnb_model = MultinomialNB()
mnb_model.fit(x_train,y_train)


# In[46]:


print('MultinomialNB Model Results'.center(80,'*'))
evaluation('Train Data Evaluation', mnb_model, x_train, y_train)
print('*#'*30)
evaluation('Test Data Evaluation', mnb_model, x_test, y_test)


# # 3. Bernoulli NB

# In[47]:


bnb_model = BernoulliNB()
bnb_model.fit(x_train,y_train)


# In[48]:


print('BernoulliNB Model Results'.center(80,'*'))
evaluation('Train Data Evaluation', bnb_model, x_train, y_train)
print('*#'*30)
evaluation('Test Data Evaluation', bnb_model, x_test, y_test)


# In[ ]:





# # Balancing data

# In[92]:


df['TenYearCHD'].value_counts()


# count of 1's are very much less than o's so the data is imbalanced

# In[93]:


pip install -U imbalanced-learn


# In[94]:


from imblearn.over_sampling import SMOTE


# In[95]:


smote = SMOTE(sampling_strategy='not majority')
x_sm,y_sm = smote.fit_resample(x,y)


# In[96]:


y_sm=pd.DataFrame(y_sm)


# In[97]:


y_sm.value_counts()


# In[98]:


x = df.drop(['TenYearCHD'],axis=1)
y = df['TenYearCHD']
x_train , x_test , y_train , y_test = train_test_split(x_sm,y_sm,test_size=0.2,random_state=11,stratify = y_sm)


# In[99]:


x_train.isnull().sum()


# In[100]:


y_train.value_counts()


# # 1) Logistic Regression

# In[101]:


log_clf = LogisticRegression()
log_clf.fit(x_train,y_train)


# In[102]:


# Testing data evaluation
y_pred_test = log_clf.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('confusion matrix :\n',cnf_matrix)

acc_score_LR = accuracy_score(y_test,y_pred_test)
print("Accuracy Score : ",acc_score_LR)

clf_report = classification_report(y_test , y_pred_test)
print('classification_report : \n',clf_report)


# In[103]:


# Training data evaluation
y_pred_train= log_clf.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score_LR = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score_LR)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# In[104]:


LR=print("Accuracy Score :  0.7412071211463309")
LR


# # Feature Scaling

# In[105]:


std_scaler = StandardScaler()
std_scaler.fit(x_train)


# In[106]:


std_scaler.fit_transform(x_train)


# In[107]:


std_scaler.transform(x_test)


# # 2) KNN

# In[108]:


knn_model = KNeighborsClassifier()
knn_model.fit(x_train,y_train)


# In[109]:


# Testing data evaluation
y_pred_test = knn_model.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('confusion matrix :\n',cnf_matrix)

acc_score_knn = accuracy_score(y_test,y_pred_test)
print("Accuracy Score : ",acc_score_LR)

clf_report = classification_report(y_test , y_pred_test)
print('classification_report : \n',clf_report)


# In[110]:


# Training data evaluation
y_pred_train = knn_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score_LR = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score_LR)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# In[3]:


knn=print('Accuracy Score :  0.8682153712548849')
knn


# # 3) SVM

# In[99]:


svc_model = SVC()
svc_model.fit(x_train,y_train)


# In[102]:


# testing data evaluation
y_pred_test = svc_model.predict(x_test)

acc= accuracy_score(y_test,y_pred_test)
print('Accuracy is:', acc)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('Confusion Matrix:\n', cnf_matrix)

Clf = classification_report(y_test,y_pred_test)
print('Classification Report:\n', Clf)


# In[103]:


# training data evaluation
y_pred_train = svc_model.predict(x_train)

acc= accuracy_score(y_train,y_pred_train)
print('Accuracy is:', acc)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('Confusion Matrix:\n', cnf_matrix)

Clf = classification_report(y_train,y_pred_train)
print('Classification Report:\n', Clf)


# In[4]:


SVM=print("Accuracy is: 0.6671732522796353")
SVM


# # 4) Naive Bayes

# In[115]:


import os
import glob
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import warnings
warnings.filterwarnings('ignore')

since the dependent variable is catagorical which takes two values "1" or "0" thus they follows Bernoulli Distribution.
# # Evaluation

# In[117]:


def evaluation(string, model, ind_var, act):
    print(string.center(50,'*'))
    pred = model.predict(ind_var)
    cnf_matrix = confusion_matrix(act,pred)
    print('Confusion Matrix :\n', cnf_matrix)
    
    accuracy = accuracy_score(act,pred)
    print('Accuracy :', accuracy)
    
    clf_report = classification_report(act, pred)
    print('Classification Report :\n', clf_report)


# In[112]:


bnb_model = BernoulliNB()
bnb_model.fit(x_train,y_train)


# In[118]:


print('BernoulliNB Model Results'.center(80,'*'))
evaluation('Train Data Evaluation', bnb_model, x_train, y_train)
print('*#'*30)
evaluation('Test Data Evaluation', bnb_model, x_test, y_test)


# In[5]:


NB=print("Accuracy score is : 0.7430555555555556")
NB


# # 5) Decision Tree

# In[121]:


dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)


# In[122]:


# Testing data evaluation
y_pred_test= dt_model.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('confusion matrix :\n',cnf_matrix)

acc_score_DT = accuracy_score(y_test,y_pred_test)
print("Accuracy Score : ",acc_score_DT)

clf_report = classification_report(y_test , y_pred_test)
print('classification_report : \n',clf_report)


# In[123]:


# Training data evaluation
y_pred_train= dt_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# In[124]:


print("Accuracy Score :  1.0")

 since the Accuracy Score is  1.0 hence the model is overfit so we go for hyperparametric tunning
# In[125]:


#HYPER PARAMETER TUNNING

dt_model=DecisionTreeClassifier()
hyp = {'criterion':['gini','entropy'],
       'max_depth':np.arange(2,10),
       'min_samples_split':np.arange(2,10),
       'min_samples_leaf':np.arange(2,10)
        
      
      }
rscv_model = RandomizedSearchCV(dt_model,hyp,random_state=10,cv=5)
rscv_model.fit(x_train,y_train)


# In[126]:


rscv_model.best_estimator_


# In[127]:


# Training data evaluation
rscv_model_train= rscv_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# In[7]:


DT=print('Accuracy Score :  1.0')
DT


# # 6) Random Forest

# In[111]:


rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)


# In[112]:


rf_model=rscv_model.best_estimator_
rf_model.fit(x_train,y_train)


# In[113]:


# Testing data evaluation
y_pred_test = rf_model.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('confusion matrix :\n',cnf_matrix)

acc_score_RF = accuracy_score(y_test,y_pred_test)
print("Accuracy Score : ",acc_score_RF)

clf_report = classification_report(y_test , y_pred_test)
print('classification_report : \n',clf_report)


# In[114]:


# Training data evaluation
y_pred_train= rf_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score_rf = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score_rf)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# # HYPERPARAMETER TUNNING

# In[115]:


rf_model = RandomForestClassifier()
hyp = hyp = {'criterion':['gini','entropy'],
       'max_depth':np.arange(2,10),
       'min_samples_split':np.arange(2,10),
       'min_samples_leaf':np.arange(2,10),
             'n_estimators':np.arange(60,100),
             
      
      }
rscv_model = RandomizedSearchCV(rf_model,hyp,random_state=52,cv=5)
rscv_model.fit(x_train,y_train)


# In[116]:


rscv_model.best_estimator_


# In[119]:


# Training data evaluation
y_pred_train= rscv_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('confusion matrix :\n',cnf_matrix)

acc_score_rf = accuracy_score(y_train,y_pred_train)
print("Accuracy Score : ",acc_score_rf)

clf_report = classification_report(y_train,y_pred_train)
print('classification_report : \n',clf_report)


# In[120]:


RF=print("Accuracy Score :  0.8738601823708206")
RF


# # 7) Adaboost

# In[123]:


ada_boost = AdaBoostClassifier(n_estimators=1,random_state=1)
ada_boost.fit(x_train,y_train)


# In[124]:


# Testing data evaluation

y_pred_test = ada_boost.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred_test)
print('Confusion Matrix is\n', cnf_matrix)

acc = accuracy_score(y_test,y_pred_test)
print('Accuracy is', acc)

clf_report= classification_report(y_test,y_pred_test)
print('Classification report', clf_report)


# In[125]:


# Training data evaluation

y_pred_train = ada_boost.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print('Confusion Matrix is\n', cnf_matrix)

acc = accuracy_score(y_train,y_pred_train)
print('Accuracy is', acc)

clf_report= classification_report(y_train,y_pred_train)
print('Classification report', clf_report)


# In[126]:


AB=print("Accuracy is 0.6417716022579244")
AB


# In[127]:


ada_boost.feature_importances_


# In[128]:


s1 = pd.Series(ada_boost.feature_importances_, index=x.columns)
s1


# In[129]:


s1 = pd.Series(ada_boost.feature_importances_, index=x.columns)
s1.sort_values().plot(kind ='barh')


# From Adaboost Age is factor mostly affecting CHD

# In[131]:


data=[['LR',0.7412071211463309,0.5046,0.748154581,0.602702065],
      ['knn',0.8682153712548849,0.4313,0.749023013,0.54739867],
      ['SVM',0.6671732522796353,0.48291,0.6443769,0.552079634],
      ['NB',0.7430555555555556,0.53855,0.778983934,0.636828853],
      ['DT',1.0,0.5,1,0.666666667],
      ['RF',0.8738601823708206,0.506,0.885366913,0.643964823],
      ['AB',0.6417716022579244,0.4529,0.581415545,0.509173805]]
data


# In[132]:


import pandas as pd


# In[133]:


df = pd.DataFrame(data, columns=['Model', 'Accuracy','Recall','Precision','f1_score'])
df


# from above data we conclude that the Random Forest is the best fitted model as the Accuracy is 0.873860	

# # ROC Curve

# In[138]:


rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)


# In[139]:


## Y prediction for test data


# In[140]:


y_test_pred=rf_model.predict(x_test)
y_test_pred


# In[141]:


y_test


# In[142]:


from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score # evalution matrix  # evalution matrix


# In[143]:


y_test_pred_prob= rf_model.predict_proba(x_test) # for prediction of observationes
y_test_pred_prob


# In[144]:


fpr , tpr, thresh = roc_curve(y_test, y_test_pred_prob[:,1])


# In[145]:


fpr


# In[146]:


tpr


# In[147]:


# by this graph we check how good our classifier works


# In[148]:


plt.plot(fpr, tpr) 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('ROC_Curve.png')


# In[ ]:





# In[149]:


## Y prediction for train data


# In[150]:


y_train


# In[151]:


y_train_pred_prob= rf_model.predict_proba(x_train) # for prediction of observationes
y_train_pred_prob


# In[152]:


fpr , tpr, thresh = roc_curve(y_train, y_train_pred_prob[:,1])


# In[153]:


fpr


# In[154]:


tpr


# In[155]:


plt.plot(fpr, tpr) 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('ROC_Curve.png')


# In[ ]:





# # Predictions

# In[135]:


y_pred_train= rscv_model.predict(x_train)
y_pred_train


# In[136]:


RF_df=pd.DataFrame(y_pred_train,columns=['Prediction'])
RF_df


# In[137]:


#color palette selection
colors = sns.color_palette("Paired")

# plotting data on chart
plt.figure(figsize=(10,6))
explode = [0,0.1]
textprops = {'fontsize':11}
plt.pie(RF_df['Prediction'].value_counts(), labels=['Not CHD(%)','CHD(%)'], startangle=90, colors=colors, explode = explode, autopct="%1.1f%%",textprops = textprops)
plt.title('Prediction for 10 years CHD', fontsize=15)
plt.show()


# # Conclusion:
According to the prediction of our best fitted model i.e Random Forest we conclude that there will be a slightly
higher chances of pateint having Coronanry Heart Disease.
# In[ ]:




