# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:20:55 2021

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
from statsmodels.formula.api import ols
import statistics
from scipy import stats

# Scikit-learn
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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
          'Tenure' , 'MonthlyCharge' , 'Bandwidth_GB_Year']].hist()
plt.savefig('churn_pyplot.jpg')
plt.tight_layout()

# Create Seaborn boxplots for continuous variables
sns.boxplot('Tenure' , data = churn_df)
plt.show()

sns.boxplot('MonthlyCharge' , data = churn_df)
plt.show()

sns.boxplot('Bandwidth_GB_Year' , data = churn_df)
plt.show()


# Run Scatterplots to show direct or inverse relationships between the target & independent variables
sns.scatterplot(x=churn_df['Children'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['Age'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()

sns.scatterplot(x=churn_df['Income'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['Outage_sec_perweek'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['Email'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['Contacts'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['Yearly_equip_failure'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['Tenure'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['MonthlyCharge'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['TimelyResponse'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['Fixes'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['DummyTechie'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['DummyGender'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['DummyContract'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()


sns.scatterplot(x=churn_df['DummyInternetService'], y=churn_df['Bandwidth_GB_Year'],
                color='red')
plt.show()

# Provide a copy of the prepared data set
churn_df.to_csv('churn_prepared.csv')
df = churn_df.columns
print(df)

# Develop the initial estimated regression equation that could be used to predict the Bandwidth_GB_Year, given the only continuous variables
lm_bandwidth = ols("Bandwidth_GB_Year ~ Children + Age + Income + Outage_sec_perweek + Email +Contacts +Yearly_equip_failure +Tenure + MonthlyCharge + TimelyResponse + Fixes + Replacements + Reliability + Options + Respectfulness + Courteous + Listening", data=churn_df).fit()
print(lm_bandwidth.params)
print(lm_bandwidth.summary())

# Model including all dummy variables
lm_bandwidth = ols("Bandwidth_GB_Year ~ Children + Age + Income + Outage_sec_perweek + Email +Contacts +Yearly_equip_failure +Tenure + MonthlyCharge + TimelyResponse + Fixes + Replacements + Reliability + Options + Respectfulness + Courteous + Listening + DummyChurn + DummyTechie + DummyPort_modem + DummyTablet + DummyPhone + DummyMultiple + DummyOnlineSecurity + DummyOnlineBackup + DummyDeviceProtection + DummyTechSupport + DummyStreamingTV + DummyStreamingMovies + DummyPaperlessBilling + DummyInternetService + DummyContract + DummyGender", data=churn_df).fit()
print(lm_bandwidth.params)
print(lm_bandwidth.summary())

# Create dataframe for heatmap bivariate analysis of correlation
churn_bivariate = churn_df[['Bandwidth_GB_Year', 
                            'Children', 
                            'Age', 
                            'Income',
                            'Outage_sec_perweek', 
                            'Yearly_equip_failure',
                            'DummyTechie', 
                            'DummyContract',
                            'DummyPort_modem', 
                            'DummyTablet', 
                            'DummyInternetService',
                            'DummyPhone', 
                            'DummyMultiple', 
                            'DummyOnlineSecurity',
                            'DummyOnlineBackup', 
                            'DummyDeviceProtection',
                            'DummyTechSupport',
                            'DummyStreamingTV',
                            'DummyPaperlessBilling',
                            'DummyGender' ,
                            'DummyStreamingMovies' ,
                            'Email', 
                            'Contacts',
                            'Tenure',
                            'MonthlyCharge', 
                            'TimelyResponse',
                            'Fixes',
                            'Replacements', 
                            'Reliability', 
                            'Options',
                            'Respectfulness',
                            'Courteous', 
                            'Listening']]

# Run Seaborn heatmap
sns.heatmap(churn_bivariate.corr(), annot=False)
plt.show()

churn_bivariate = churn_df[['Bandwidth_GB_Year' ,
                            'Children' ,
                            'Tenure' ,
                            'TimelyResponse' ,
                            'Fixes' ,
                            'Replacements' ,
                            'Respectfulness' ,
                            'Courteous' ,
                            'Listening']]

sns.heatmap(churn_bivariate.corr(), annot=True)
plt.show

# Run reduced OLS multiple regression
lm_bandwidth_reduced = ols("Bandwidth_GB_Year ~ Children + Tenure + Fixes + Replacements", data=churn_df).fit()
print(lm_bandwidth.params)
print(lm_bandwidth.summary())


churn_df = pd.read_csv('churn_prepared.csv')
residuals = churn_df['Bandwidth_GB_Year'] = lm_bandwidth_reduced.predict(churn_df[['Children', 'Tenure', 'Fixes','Replacements']])
sns.scatterplot(x=churn_df['Tenure'],y=residuals,color='red')
plt.show()