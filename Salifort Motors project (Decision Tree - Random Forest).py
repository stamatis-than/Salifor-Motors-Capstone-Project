#!/usr/bin/env python
# coding: utf-8

# ## Salifort Motors project
#  
# ### Provide data-driven suggestions for HR

# Currently, there is a high rate of turnover among Salifort employees. (Note: In this context, turnover data includes both employees who choose to quit their job and employees who are let go). Salifort’s senior leadership team is concerned about how many employees are leaving the company. Salifort strives to create a corporate culture that supports employee success and professional development. Further, the high turnover rate is costly in the financial sense. Salifort makes a big investment in recruiting, training, and upskilling its employees. 
# 
# If Salifort could predict whether an employee will leave the company, and discover the reasons behind their departure, they could better understand the problem and develop a solution. 
# 
# As a first step, the leadership team asks Human Resources to survey a sample of employees to learn more about what might be driving turnover.  
# 
# Next, the leadership team asks you to analyze the survey data and come up with ideas for how to increase employee retention. To help with this, they suggest you design a model that predicts whether an employee will leave the company based on their job title, department, number of projects, average monthly hours, and any other relevant data points. A good model will help the company increase retention and job satisfaction for current employees, and save money and time training new employees. 

# ### **What’s likely to make the employee leave the company?**

# ### Deliverables
# 
# Analyze the key factors driving employee turnover, build an effective model, and share recommendations for next steps with the leadership team.
# 
# - Model evaluation
# - Data visualizations
# - Ethical considerations
# - Resources
# - One-page summary of this project

# ### HR dataset 
# 
# In this [dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv), there are 14,999 rows, 10 columns, and these variables: 
# 
# Variable  |Description |
# -----|-----| 
# satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
# last_evaluation|Score of employee's last performance review [0&ndash;1]|
# number_project|Number of projects employee contributes to|
# average_monthly_hours|Average number of hours employee worked per month|
# time_spend_company|How long the employee has been with the company (years)
# Work_accident|Whether or not the employee experienced an accident while at work
# left|Whether or not the employee left the company
# promotion_last_5years|Whether or not the employee was promoted in the last 5 years
# Department|The employee's department
# salary|The employee's salary (U.S. dollars)

# In[173]:


# Import packages

# For data manipulation
import pandas as pd
import numpy as np

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option("display.max_columns", None)

# For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful function
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle


# In[174]:


# Load dataset
df0 = pd.read_csv("HR_capstone_dataset.csv")

# Display first few rows
df0.head(10)


# ### EDA

# In[175]:


# Basic information
df0.info()


# In[176]:


# Descriptive statistics
df0.describe()


# In[177]:


# Rename columns names to standardize

# Display all column names
df0.columns


# In[178]:


# Rename columns
df0 = df0.rename(columns={"average_montly_hours": "average_monthly_hours",
                          "time_spend_company": "tenure",
                          "Work_accident": "work_accident",
                          "Department": "department"})
# Display column names
df0.columns


# In[179]:


# Check for missing values
df0.isna().sum()


# No missing values

# In[180]:


# Check for duplicates
df0.duplicated().sum()


# 3,008 rows contain duplicate values. That is 20% of the data

# In[181]:


# Inspect some rows containing duplicates
df0[df0.duplicated()].head(10)


# Since there are 10 columns with some of them containing continous variables, it seems unlikely for the duplicate observations to be legitimate.

# In[182]:


# Drop duplicates, save resulting dataframe in a new variable
df1 = df0.drop_duplicates(keep="first")

df1.head(10)


# ### Check for outliers

# In[183]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(6,6))
plt.title("Boxplot to detect outliers for tenure", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1["tenure"])
plt.show()


# Investigate how many rows in contain outliers in the `tenure` column

# In[184]:


# Determine the number of rows containing outliers

# 25th percentile
percentile25 = df1["tenure"].quantile(0.25)

# 75th percentile
percentile75 = df1["tenure"].quantile(0.75)

# IQR
iqr = percentile75 - percentile25

# upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("upper limit:", upper_limit)
print("lower limit:", lower_limit)

# subset of data containing outliers
outliers = df1[(df1["tenure"] > upper_limit) | (df1["tenure"] < lower_limit)]

# count rows
print("Number of rows in the data containing outliers in `tenure`:", len(outliers))


# In[185]:


# Number of people who left vs. stayed
print(df1["left"].value_counts())
print()

# Percentage of people who left vs. stayed
print(df1["left"].value_counts(normalize=True))


# ### Data visualizations

# In[186]:


fig, ax = plt.subplots(1, 2, figsize=(22,8))

# Boxplot `average_monthly_hours` for `number_project
# comparing left vs. stayed
sns.boxplot(data=df1,
            x="average_monthly_hours",
            y="number_project",
            hue="left",
            orient="h",
           ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title("Monthly hours by number of projects", fontsize="14")

# Histogram of number of projects distribution
# comparing left vs. stayed
tenure_stay = df1[df1["left"]==0]["number_project"]
tenure_left = df1[df1["left"]==1]["number_project"]
sns.histplot(data=df1,
             x="number_project",
             hue="left",
             multiple="dodge",
             shrink=2,
             ax=ax[1])
ax[1].set_title("Number of projects histogram", fontsize="14")

plt.show()


# Assume a  work week of 40 hour a week and two weeks vacation per year.
# The average number of wokring hours per month per employees working Monday-Friday `= 50 weeks * 40 hours per week / 12 months = 166.67 hours per month`
# 
# It seems that employees here are overworked.
# 
# There are two groups of employees who left the company: (A) those who worked less hours and in fewer projects and (B) those who worked more hours and in more projects.
# 
# Group A employees were assigned 2 projects and it may be due to the fact that they most probably were let go.
# On the other hand Group B employees that were assigned 7 projects all left.
# 
# Employees who worked on 3 or 4 projects seem to have the best retention.

# In[187]:


# Get value counts of stayed/left for employees with  projects
df1[df1["number_project"]==7]["left"].value_counts()


# In[188]:


# Scatterplot of `average_monthly_hours` vs. `satisfaction_level
# comparing left vs. stayed
plt.figure(figsize=(16,9))
sns.scatterplot(data=df1,
                x="average_monthly_hours",
                y="satisfaction_level",
                hue="left",
                alpha=0.4)
plt.axvline(x=166.67,
            color="#ff6361",
            label="166.67 hrs/mo",
            ls="--")
plt.legend()
plt.title("Monthly hours by last evaluation score", fontsize="14");


# There is a sizeble group of employees that left, who worked ~240-315 hours per month and with their satisfaction level being close to zero.
# 
# Another sizeable group of employees that left, worked ~130-160 hours per month with a satisfaction level around 0.4.
# 
# The last group of employees that left worked ~210-280 hours per week with their satisfaction level ranging ~0.7-0.9.
# 
# (The strange shape of the distributions is indicatice of data manipulation or synthetic data.)

# In[189]:


# Visualize satisfaction levels by tenure

fig, ax = plt.subplots(1, 2, figsize=(22,8))

# Boxplot of distributions of `satisfaction_level` by `tenure`, employees stayed vs. left
sns.boxplot(data=df1,
            x="satisfaction_level",
            y="tenure",
            hue="left",
            orient="h",
            ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title("Satisfaction by tenure", fontsize="14")

# Histogram of distribution of `tenure`, employees stayed vs. left
tenure_stay = df1[df1["left"]==0]["tenure"]
tenure_left = df1[df1["left"]==1]["tenure"]
sns.histplot(data=df1,
             x="tenure",
             hue="left",
             multiple="dodge",
             shrink=5,
             ax=ax[1])
ax[1].set_title("Tenure histogram", fontsize="14")

plt.show();


# Most of the employees work from 2 to 4 years, while few of the employees work for more than 6 years.
# 
# For the employees who left, satisfaction levels drop significantly up to 4 years, while after 5 years satisfaction levels are quite high.
# 
# For the employees who stayed, satisfaction levels show a slight decrease uo tp 5 years and then they rise again.

# In[190]:


# Calculate mean and median satisfaction scores of employees (stayed and left)
df1.groupby(["left"])["satisfaction_level"].agg([np.mean, np.median])


# In[191]:


# Examine salary levels for different tenures

fig, ax = plt.subplots(1, 2, figsize = (22,8))

tenure_short = df1[df1["tenure"] < 7]
tenure_long = df1[df1["tenure"] > 6]

# Short tenure histogram
sns.histplot(data=tenure_short,
             x="tenure",
             hue="salary",
             discrete=1,
             hue_order=["low", "medium", "high"],
             multiple="dodge",
             shrink=0.5,
             ax=ax[0])
ax[0].set_title("Salary histogram / Short tenure", fontsize="14")

# Long tenure histogram
sns.histplot(data=tenure_long,
             x="tenure",
             hue="salary",
             discrete=1,
             hue_order=["low", "medium", "high"],
             multiple="dodge",
             shrink=0.5,
             ax=ax[1])
ax[1].set_title("Salary histogram / Long tenure", fontsize="14")

plt.show();


# There are no disproportionately higher-paid long-tenured employees

# In[192]:


# Scatterplot of `average_monthly_hours` vs. `last_evaluation`
plt.figure(figsize=(16,9))
sns.scatterplot(data=df1,
                x="average_monthly_hours",
                y="last_evaluation",
                hue="left",
                alpha=0.4)
plt.axvline(x=166.67,
            color="#ff6361",
            label="166.67 hrs/mo",
            ls="--")
plt.legend() #labels=["166.67 hrs/mo", "left", "stayed"]
plt.title("Monthly hours by last evaluation score", fontsize="14");


# Two groups of employees who left:
# - Long working hours / High evaluation scores
# - Short working hours / Low evaluation scores
# 
# Working long hours doesn't mean a higher evaluation score.
# 
# Most of the employees work well over 167 hours per month.

# In[193]:


# Examine relationship between `average_monthly_hours` and `promotion_last_5years`

plt.figure(figsize=(16,3))
sns.scatterplot(data=df1,
                x="average_monthly_hours",
                y="promotion_last_5years",
                hue="left",
                alpha=0.4)
plt.axvline(x=166.67,
            color="red",
            ls="--")
plt.legend() #labels=["166.67", "left", "stayed"]
plt.title("Monthly hours by pormotion in the last 5 years", fontsize="14");


# The majority of the employees who left, worked the most hours and were not promoted in the last 5 years.

# In[194]:


# Employees at each department
df1["department"].value_counts()


# In[195]:


# Department distribution of employees / stayed vs. left
plt.figure(figsize=(16,6))
sns.histplot(data=df1,
             x="department",
             hue="left",
             discrete=1,
             hue_order=[0,1],
             multiple="dodge",
             shrink=0.5)
plt.xticks(rotation=45)
plt.title("Counts of stayed/left by department", fontsize=14);


# No department seems to differ significanlty i its proportion of employees who left to those who stayed

# In[196]:


plt.figure(figsize=(16,9))
heatmap = sns.heatmap(df1.corr(numeric_only=True),
                      vmin=-1,
                      vmax=1,
                      annot=True,
                      cmap=sns.color_palette("vlag", as_cmap=True))
plt.xticks(rotation=45)
heatmap.set_title("Correlation Heatmap",
                  fontdict={"fontsize":14}, pad=12);


# There is some positive correlation between number of projects, monthly hours and evaluation scores, and some negative correlation between employee leaves and satisfaction levels.

# ### Insights
# 
# It appears that employees are leaving the company as a result of poor management. Leaving is tied to longer working hours, many projects, and generally lower satisfaction levels. It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. It also appears that if an employee has spent more than six years at the company, they tend not to leave. 

# ### Identify the type of prediction task and the types of models most appropriate for this task

# The goal is to predict whether an employee leaves the company.
# 
# The outcome variable `left` can be either 1 (left) or 0 (stayed) which makes it a categorical outcome variable.
# 
# There are two types of models appropriate for this prediction task: 
# - Logistic Regression model
# - Tree-based Machine Learning model
# 

# ### Build a Tree-based model 

# **Implementation of Decision Tree and Random Forest**

# Before splitting the data, non-numeric variables must be encoded.
# 
# `department` will be dummied and `salary` because it's ordinal its levels will be converted to numbers (0-2).

# In[197]:


# Copy the dataframe
df_enc = df1.copy()

# Encode the `salary` column as an ordinal numeric category
df_enc["salary"] = (df_enc["salary"].astype("category")
                    .cat.set_categories(["low", "medium", "high"])
                    .cat.codes)

# Dummny encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False, dtype=int)

# Display the new datarame
df_enc.head()


# In[198]:


# Isolate the outcome variable
y = df_enc["left"]

y.head()


# In[199]:


# Select the features
X = df_enc.drop("left", axis=1)

X.head()


# In[200]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# **Decision Tree - Round 1**
# 
# Construct a decision tree model and set up cross-validated grid-search to exhuastively search for the best models parameters.

# In[201]:


# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {"max_depth": [4, 6, 8, None],
             "min_samples_leaf": [2, 5, 1],
             "min_samples_split": [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {"accuracy": "accuracy", 
           "precision": "precision", 
           "recall": "recall", 
           "f1": "f1", 
           "roc_auc": "roc_auc"}

# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit="roc_auc")


# In[202]:


get_ipython().run_cell_magic('time', '', '\n# Fit the decision tree model to the training data\ntree1.fit(X_train, y_train)\n')


# In[203]:


# Identify the optimal values for the decision tree parameters
tree1.best_params_


# In[204]:


# Identify the best AUC score on CV
tree1.best_score_


# This strong AUC score indicates that this model can predict employees who will leave very well.

# In[205]:


# Write a funciton to extract scores from the grid search

def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
        
    Returns a pandas df with the F1, recall, precision, accuracy and AUC scores
    for the model with the best mean metric score across all validation folds.
    '''
    
    # Create dictionary that maps imput metric to actual metric name in GridSearchCV
    metric_dict = {"auc": "mean_test_roc_auc",
                   "precision": "mean_test_recall",
                   "f1": "mean_test_f1",
                   "accuracy": "mean_test_accuracy"}
    # Get all the results from the CV ad put them in a DF
    cv_results = pd.DataFrame(model_object.cv_results_)
    
    # Isolate the row of the DF with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]
    
    # Extract Accuracy, Precision, Recall, and F1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
    
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({"model": [model_name],
                          "precision": [precision],
                          "recall": [recall],
                          "F1": [f1],
                          "accuracy": [accuracy],
                          "AUC": [auc]})
    
    return table


# In[206]:


# Use the function to get all CV scores
tree1_cv_results = make_results("decision tree cv", tree1, "auc")
tree1_cv_results


# All those scores from the Decision Tree model are strong indicators of good model performance.
# 
# But Decision Trees are vulnerable to overfitting. Random Forests avoid overfitting by incorporating multiple trees to make predictions.
# 
# It's a good idea to biuld a Random Forest model and set up cross-validated grid-search to search for the best model parameters.

# **Random Forest - Round 1**

# In[207]:


# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {"max_depth": [3, 5, None],
             "max_features": [1.0],
             "max_samples": [0.7, 1.0],
             "min_samples_leaf": [1, 2, 3],
             "min_samples_split": [2, 3, 4],
             "n_estimators": [300, 500]}

# Assign a dictionary of scoring metrics to capture
scoring = {"accuracy": "accuracy", 
           "precision": "precision", 
           "recall": "recall", 
           "f1": "f1", 
           "roc_auc": "roc_auc"}

# Intantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit="roc_auc")


# In[208]:


get_ipython().run_cell_magic('time', '', '\n# Fit the random forest model to the training data\nrf1.fit(X_train, y_train)\n')


# In[209]:


import os

notebook_path = os.getcwd()
print(notebook_path)


# In[210]:


path = '/Users/imac/Capstone project: Salifort Motors, providing data-driven suggestions for HR/'


# In[211]:


def write_pickle(path, model_object, save_as:str):
    '''
    In: 
        path:         path of folder where you want to save the pickle
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the folder indicated
    '''    
    with open(path + save_as + ".pickle", "wb") as to_write:
        pickle.dump(model_object, to_write)


# In[212]:


def read_pickle(path, saved_model_name:str):
    '''
    In: 
        path:             path to folder where you want to read from
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(path + saved_model_name + ".pickle", "rb") as to_read:
        model = pickle.load(to_read)
        
    return model


# In[213]:


# Write pickle
write_pickle(path, rf1, "hr_rf1")


# In[214]:


# Read pickle
rf1 = read_pickle(path, "hr_rf1")


# In[215]:


# Check best AUC score on CV
rf1.best_score_


# In[216]:


# Check best parameters
rf1.best_params_


# In[217]:


# Get all CV scores
rf1_cv_results = make_results("random forest cv", rf1, "auc")
print(tree1_cv_results)
print(rf1_cv_results)


# The evaluation scores of the Random Forest model are better than thos of the Decision Tree model, with an exception of recall that is 0.001 lower.
# 
# This indicates that the Random Forest model outperforms the Decision Tree model.

# In[218]:


# Define a function that gets all the scores from a model's predictions

def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''
    
    preds = model.best_estimator_.predict(X_test_data)
    
    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)
    
    table = pd.DataFrame({"model": [model_name],
                          "precision": [precision],
                          "recall": [recall],
                          "f1": [f1],
                          "accuracy": [accuracy],
                          "AUC": [auc]})
    
    return table


# In[219]:


# Get predicitons on test data
rf1_test_scores = get_scores("random forest1 test", rf1, X_test, y_test)
rf1_test_scores


# The test scores are very similar to the validation scores, which is good. This appears to be a strong model. Since this test set was only used for this model, you can be more confident that your model's performance on this data is representative of how it will perform on new, unseeen data.

# **Feature Engineering**
# 
# In order to improve the model and avoid data leakage from occuring the next round will incorporate feature engineering.
# 
# There is a possibility that the `satisfaciton_level` and `average_monthly_hours` columns are a source of data leakage because either the satisfaction levels reported are not correct or employees that decided to quit or have been informed to be fired may be working fewer hours.  

# In[220]:


# Drop `satisfaction_level` 
df2 = df_enc.drop("satisfaction_level", axis=1)

df2.head()


# In[221]:


# Create `overworked` column. Begin with appending `average_monthly_hours` column values
df2["overworked"] = df2["average_monthly_hours"]

# Inspect max and min average monthly hours values
print("Max hours:", df2["overworked"].max())
print("Min hours:", df2["overworked"].min())


# Previously we've calculated the average monthly hours for someone who works 50 weeks per year, 5 days per week, 8 hours per day at 166.67hrs/month.
# 
# We can define being overworked as working more than 175 hours per month on average.

# In[222]:


# Define `overworked` as working > 175 hrs/week
df2["overworked"] = (df2["overworked"] > 175).astype(int)

df2["overworked"].head()


# In[223]:


# Drop the `average_monthly_hours` column
df2 = df2.drop("average_monthly_hours", axis=1)

df2.head()


# In[224]:


# Isolate the outcome varaible
y = df2["left"]

# Select the features
X = df2.drop("left", axis=1)


# In[225]:


# Create the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# **Decision Tree - Round 2**

# In[226]:


# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {"max_depth": [4, 6, 8, None],
             "min_samples_leaf": [2, 5, 1],
             "min_samples_split": [2, 4, 6]}

# Assign a dictionary of scoring metrics to capture
scoring = {"accuracy": "accuracy", 
           "precision": "precision", 
           "recall": "recall", 
           "f1": "f1", 
           "roc_auc": "roc_auc"}

# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit="roc_auc")


# In[227]:


get_ipython().run_cell_magic('time', '', 'tree2.fit(X_train, y_train)\n')


# In[228]:


# Check best parameters
tree2.best_params_


# In[229]:


# Check best AUC score on CV
tree2.best_score_


# This model performs very well, even without satisfaction levels and detailed hours worked data.
# 

# In[230]:


# Get all CV scores
tree2_cv_results = make_results("decision tree2 cv", tree2, "auc")
print(tree1_cv_results)
print(tree2_cv_results)


# Some of the other scores fell. That's to be expected given fewer features were taken into account in this round of the model. Still, the scores are very good.

# **Random Forest - Round 2**

# In[231]:


# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {"max_depth": [3, 5, None],
             "max_features": [1.0],
             "max_samples": [0.7, 1.0],
             "min_samples_leaf": [1, 2, 3],
             "min_samples_split": [2, 3, 4],
             "n_estimators": [300, 500]}

# Assign a dictionary of scoring metrics to capture
scoring = {"accuracy": "accuracy", 
           "precision": "precision", 
           "recall": "recall", 
           "f1": "f1", 
           "roc_auc": "roc_auc"}

# Instantiate GridSearch
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit="roc_auc")


# In[232]:


get_ipython().run_cell_magic('time', '', 'rf2.fit(X_train, y_train)\n')


# In[233]:


# Write pickle
write_pickle(path, rf2, "hr_rf2")


# In[234]:


# Read pickle
rf2 = read_pickle(path, "hr_rf2")


# In[235]:


# Check best params
rf2.best_params_


# In[236]:


# CHeck best AUC score on CV
rf2.best_score_


# In[237]:


# Get all CV scores
rf2_cv_results = make_results("random forest2 cv", rf2, "auc")
print(tree2_cv_results)
print(rf2_cv_results)


# Again, the scores dropped slightly, but the random forest performs better than the decision tree if using AUC as the deciding metric.

# In[238]:


# Get predictions on test data
rf2_test_scores = get_scores("random forest2 test", rf2, X_test, y_test)
rf2_test_scores


# This seems to be a stable, well-performing final model.
# 
# Plot a confusion matrix to visualize how well it predicts on the test set.

# In[239]:


# Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(values_format='')

plt.show()


# The upper-left quadrant displays the number of true negatives.
# The upper-right quadrant displays the number of false positives.
# The bottom-left quadrant displays the number of false negatives.
# The bottom-right quadrant displays the number of true positives.
# 
# True negatives: The number of people who did not leave that the model accurately predicted did not leave.
# 
# False positives: The number of people who did not leave the model inaccurately predicted as leaving.
# 
# False negatives: The number of people who left that the model inaccurately predicted did not leave
# 
# True positives: The number of people who left the model accurately predicted as leaving
# 
# A perfect model would yield all true negatives and true positives, and no false negatives or false positives.

# The model predicts more false positives than false negatives, which means that some employees may be identified as at risk of quitting or getting fired, when that's actually not the case. But this is still a strong model.

# **Decision Tree Splits**

# In[241]:


# Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree2.best_estimator_,
          max_depth=6,
          fontsize=14,
          feature_names=['last_evaluation', 'number_project', 'tenure', 'work_accident',
                         'promotion_last_5years', 'salary', 'department_IT', 'department_RandD',
                         'department_accounting', 'department_hr', 'department_management',
                         'department_marketing', 'department_product_mng', 'department_sales',
                         'department_support', 'department_technical', 'overworked'],
          class_names=["stayed", "left"],
          filled=True)
plt.show()


# **Decision Tree feature importance**

# In[242]:


# tree2 importances dataframe
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_,
                                 columns=["gini_importance"],
                                 index=X.columns)

tree2_importances = tree2_importances.sort_values(by="gini_importance", ascending=False)

# Extract features with importance > 0
tree2_importances = tree2_importances[tree2_importances["gini_importance"] != 0]
tree2_importances


# In[243]:


# Create barplot to visualize decision tree feature importances
sns.barplot(data=tree2_importances,
            x="gini_importance",
            y=tree2_importances.index,
            orient="h")
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()


# The barplot above shows that in this decision tree model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`.

# **Random Forest feature importance**

# In[244]:


# Get feature impotances
feat_impt = rf2.best_estimator_.feature_importances_

# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]

# Get column labels of top 10 features
feat = X.columns[ind]

# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]

y_df = pd.DataFrame({"Feature":feat, "Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")

fig = plt.figure()
ax1 = fig.add_subplot(111)

y_sort_df.plot(kind="barh",
               ax=ax1,
               x="Feature",
               y="Importance")

ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")

plt.show()


# The plot above shows that in this random forest model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`, and they are the same as the ones used by the decision tree model.

# ### Conclusion, Recommendations, Next Steps
# 
# The models and the feature importances extracted from the models confirm that employees at the company are overworked. 
# 
# To retain employees, the following recommendations could be presented to the stakeholders:
# 
# * Cap the number of projects that employees can work on.
# * Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied. 
# * Either reward employees for working longer hours, or don't require them to do so. 
# * If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear. 
# * Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts. 
# * High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort. 
# 
# **Next Steps**
# 
# It may be justified to still have some concern about data leakage. It could be prudent to consider how predictions change when `last_evaluation` is removed from the data. It's possible that evaluations aren't performed very frequently, in which case it would be useful to be able to predict employee retention without this feature. It's also possible that the evaluation score determines whether an employee leaves or stays, in which case it could be useful to pivot and try to predict performance score. The same could be said for satisfaction score. 
# 
