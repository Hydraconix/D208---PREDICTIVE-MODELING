# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 02:40:54 2021

@author: Hydraconix
"""

#Standard Data Science Imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Statistics packages
import pylab
from statsmodels.formula.api import logit
import statistics
from scipy import stats

# Scikit-learn
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


# Load data set into Pandas Dataframe
churn_df = pd.read_csv(r'C:\Users\Hydraconix\Desktop\churn_clean.csv')

# Checking for Null Values
churn_df.isna().sum()

#Summary Statistics
churn_df.Age.describe()

churn_df.Children.describe()

churn_df.Income.describe()

churn_df.Outage_sec_perweek.describe()

churn_df.Yearly_equip_failure.describe()

churn_df.Tenure.describe()

churn_df.MonthlyCharge.describe()

churn_df.Bandwidth_GB_Year.describe()

# Rename Last 8 Survey Columns for better description of variables
churn_df.rename(columns = {'Item1' : 'TimelyResponse',
                           'Item2' : 'Fixes' ,
                           'Item3' : 'Replacements' ,
                           'Item4' : 'Reliability' ,
                           'Item5' : 'Options' ,
                           'Item6' : 'Respectfulness' ,
                           'Item7' : 'Courteous' ,
                           'Item8' : 'Listening'},
                          inplace=True)

# Remove less meaningful demographic variables from statistics description
churn_df = churn_df.drop(columns=['CaseOrder' ,
                               'Customer_id' ,
                               'Interaction' ,
                               'UID' ,
                               'City' ,
                               'State' ,
                               'County' ,
                               'Zip' ,
                               'Lat' ,
                               'Lng' ,
                               'Population' ,
                               'Area' ,
                               'TimeZone' ,
                               'Job' ,
                               'Marital' ,
                               'PaymentMethod'])

# Converting binary categorical variables to numeric variables
churn_df['DummyChurn'] = [1 if v == 'Yes' else 0 for v in churn_df['Churn']]
churn_df['DummyTechie'] = [1 if v == 'Yes' else 0 for v in churn_df['Techie']]
churn_df['DummyPort_modem'] = [1 if v == 'Yes' else 0 for v in churn_df['Port_modem']]
churn_df['DummyTablet'] = [1 if v == 'Yes' else 0 for v in churn_df['Tablet']]
churn_df['DummyPhone'] = [1 if v == 'Yes' else 0 for v in churn_df['Phone']]
churn_df['DummyMultiple'] = [1 if v == 'Yes' else 0 for v in churn_df['Multiple']]
churn_df['DummyOnlineSecurity'] = [1 if v == 'Yes' else 0 for v in churn_df['OnlineSecurity']]
churn_df['DummyOnlineBackup'] = [1 if v == 'Yes' else 0 for v in churn_df['OnlineBackup']]
churn_df['DummyDeviceProtection'] = [1 if v == 'Yes' else 0 for v in churn_df['DeviceProtection']]
churn_df['DummyTechSupport'] = [1 if v == 'Yes' else 0 for v in churn_df['TechSupport']]
churn_df['DummyStreamingTV'] = [1 if v == 'Yes' else 0 for v in churn_df['StreamingTV']]
churn_df['DummyStreamingMovies'] = [1 if v == 'Yes' else 0 for v in churn_df['StreamingMovies']]
churn_df['DummyPaperlessBilling'] = [1 if v == 'Yes' else 0 for v in churn_df['PaperlessBilling']]


# Converting ordinal categorical data into numeric variables
churn_df['DummyInternetService'] = churn_df.InternetService.map({'None' : 0, 'DSL' : 1, 'Fiber Optic' : 2})
churn_df['DummyContract'] = churn_df.Contract.map({'Month-to-month' : 0, 'One year' : 1, 'Two Year' : 2})
churn_df['DummyGender'] = churn_df.Gender.map({'Nonbinary' : 0, 'Male' : 1, 'Female' : 2})


# Drop original categorical variables from dataframe
churn_df = churn_df.drop(columns=['Gender' , 
                                  'Churn' , 
                                  'Techie' , 
                                  'Contract' ,       
                                  'Port_modem' , 
                                  'Tablet' ,
                                  'InternetService' , 
                                  'Phone' , 
                                  'Multiple' ,
                                  'OnlineSecurity' ,
                                  'OnlineBackup', 
                                  'DeviceProtection' ,
                                  'TechSupport' ,
                                  'StreamingTV', 
                                  'StreamingMovies',
                                  'PaperlessBilling'])

#Create histograms of continuous variables
churn_df[['Children', 
          'Age' , 
          'Income' , 
          'Outage_sec_perweek' , 
          'Email' , 
          'Contacts' , 
          'Yearly_equip_failure' , 
          'Tenure' , 'MonthlyCharge' ,
          'Bandwidth_GB_Year' ,
          'DummyGender' ,
          'DummyInternetService' ,
          'DummyContract']].hist()
plt.savefig('churn_pyplot.jpg')
plt.tight_layout()

# Create Seaborn boxplots for continuous variables
sns.boxplot('Tenure' , data = churn_df)
plt.show()

sns.boxplot('MonthlyCharge' , data = churn_df)
plt.show()

sns.boxplot('Bandwidth_GB_Year' , data = churn_df)
plt.show()

# Run scatterplots to show direct or inverse relationships between target & independent variables
sns.scatterplot(x=churn_df['Children'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Age'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Income'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['DummyGender'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Outage_sec_perweek'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Email'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Contacts'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Yearly_equip_failure'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['DummyTechie'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Tenure'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['MonthlyCharge'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Bandwidth_GB_Year'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['TimelyResponse'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Fixes'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Replacements'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Reliability'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Options'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Respectfulness'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Courteous'], y=churn_df['DummyChurn'], color='red')
plt.show()

sns.scatterplot(x=churn_df['Listening'], y=churn_df['DummyChurn'], color='red')
plt.show()


# Provide a copy of the prepared data set
churn_df.to_csv('churn_prepared_log.csv')

# Construct an initial logistic regression model from all predictors that were identified in Part C2
churn_logit_model = logit("DummyChurn ~ Children + Age + Income + Outage_sec_perweek + Email +Contacts +Yearly_equip_failure +Tenure + MonthlyCharge + TimelyResponse + Fixes + Replacements + Reliability + Options + Respectfulness + Courteous + Listening", data=churn_df).fit()
print(churn_logit_model.params)
print(churn_logit_model.summary())

churn_logit_model2 = logit("DummyChurn ~ Children + Age + Income + Outage_sec_perweek + Email +Contacts +Yearly_equip_failure +Tenure + MonthlyCharge + TimelyResponse + Fixes + Replacements + Reliability + Options + Respectfulness + Courteous + Listening + Bandwidth_GB_Year + DummyTechie + DummyPort_modem + DummyTablet + DummyPhone + DummyMultiple + DummyOnlineSecurity + DummyOnlineBackup + DummyDeviceProtection + DummyTechSupport + DummyStreamingTV + DummyStreamingMovies + DummyPaperlessBilling + DummyInternetService + DummyContract + DummyGender", data=churn_df).fit()
print(churn_logit_model2.params)
print(churn_logit_model2.summary())

# Create dataframe for heatmap bivariate analysis of correlation
churn_bivariate = churn_df[['DummyChurn', 'Children', 'Age', 'Income', 
                            'Outage_sec_perweek', 'Yearly_equip_failure', 'DummyTechie', 'DummyContract', 
                            'DummyPort_modem', 'DummyTablet', 'DummyInternetService', 
                            'DummyPhone', 'DummyMultiple', 'DummyOnlineSecurity', 
                            'DummyOnlineBackup', 'DummyDeviceProtection', 
                            'DummyTechSupport', 'DummyStreamingTV', 
                            'DummyPaperlessBilling','Email', 'Contacts',  
                            'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'TimelyResponse', 'Fixes', 
                            'Replacements', 'Reliability', 'Options', 'Respectfulness', 
                            'Courteous', 'Listening']]

# Run Seaborn heatmap
sns.heatmap(churn_bivariate.corr(), annot=False)
plt.show()

churn_bivariate = churn_df[['DummyChurn', 'Bandwidth_GB_Year', 'Children',
                            'Tenure', 'TimelyResponse', 'Fixes', 
                            'Replacements', 'Respectfulness', 
                            'Courteous', 'Listening']]

sns.heatmap(churn_bivariate.corr(), annot=True)
plt.show()

# Run reduced logistic regression
churn_logit_model_reduced = logit("DummyChurn ~ DummyTechie + DummyContract + DummyInternetService + DummyStreamingTV", data=churn_df).fit()
print(churn_logit_model_reduced.summary())

# Confusion Matrix

# Import the prepared dataset
dataset = pd.read_csv('churn_prepared_log.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Logistic Regression model on the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred = classifier.predict(X_test)

# Make the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


y_predict_test = classifier.predict(X_test)
cm2 = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm2, annot=True)

# Classification Report
print(classification_report(y_test, y_predict_test))





