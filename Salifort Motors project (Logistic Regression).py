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

# In[17]:


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
# Pip 21.3+ is required 
# use `%` or `!`
get_ipython().run_line_magic('pip', 'install xgboost')

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


# In[18]:


# Load dataset
df0 = pd.read_csv("HR_capstone_dataset.csv")

# Display first few rows
df0.head(10)


# ### EDA

# In[19]:


# Basic information
df0.info()


# In[20]:


# Descriptive statistics
df0.describe()


# In[21]:


# Rename columns names to standardize

# Display all column names
df0.columns


# In[22]:


# Rename columns
df0 = df0.rename(columns={"average_montly_hours": "average_monthly_hours",
                          "time_spend_company": "tenure",
                          "Work_accident": "work_accident",
                          "Department": "department"})
# Display column names
df0.columns


# In[23]:


# Check for missing values
df0.isna().sum()


# No missing values

# In[24]:


# Check for duplicates
df0.duplicated().sum()


# 3,008 rows contain duplicate values. That is 20% of the data

# In[25]:


# Inspect some rows containing duplicates
df0[df0.duplicated()].head(10)


# Since there are 10 columns with some of them containing continous variables, it seems unlikely for the duplicate observations to be legitimate.

# In[26]:


# Drop duplicates, save resulting dataframe in a new variable
df1 = df0.drop_duplicates(keep="first")

df1.head(10)


# ### Check for outliers

# In[27]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(6,6))
plt.title("Boxplot to detect outliers for tenure", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1["tenure"])
plt.show()


# Investigate how many rows in contain outliers in the `tenure` column

# In[28]:


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


# In[29]:


# Number of people who left vs. stayed
print(df1["left"].value_counts())
print()

# Percentage of people who left vs. stayed
print(df1["left"].value_counts(normalize=True))


# ### Data visualizations

# In[30]:


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

# In[31]:


# Get value counts of stayed/left for employees with  projects
df1[df1["number_project"]==7]["left"].value_counts()


# In[32]:


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

# In[33]:


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

# In[34]:


# Calculate mean and median satisfaction scores of employees (stayed and left)
df1.groupby(["left"])["satisfaction_level"].agg([np.mean, np.median])


# In[35]:


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

# In[36]:


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

# In[37]:


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

# In[38]:


# Employees at each department
df1["department"].value_counts()


# In[39]:


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

# In[40]:


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

# ### Build a Logistic Regression model 

# **Logistic Regression model assumptions**
# - Outcome variable is categorical
# - Observations are independent of each other
# - No severe multicollinearity among X variables 
# - No extreme outliers
# - Linear relationship between each X variable and the logit of the outcome variable
# - Sufficiently large sample size 

# Before splitting the data, non-numeric variables must be encoded.
# 
# `department` will be dummied and `salary` because it's ordinal its levels will be converted to numbers (0-2).

# In[44]:


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


# In[49]:


# Remove outliers from `tenure`
df_logreg = df_enc[(df_enc["tenure"] >= lower_limit) & (df_enc["tenure"] <= upper_limit)]

df_logreg.head()


# In[50]:


# Isolate the outcome variable
y = df_logreg["left"]

y.head()


# In[51]:


# Select features to use in the model
X = df_logreg.drop("left", axis=1)

X.head()


# In[52]:


# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# In[53]:


# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)


# In[54]:


# Use logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)


# In[55]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot(values_format="")

plt.show();


# True negatives (upper-left quadrant): 2165, The number of people who did not leave that the model accurately predicted did not leave.
# 
# False positives (upper-right quadrant): 156, The number of people who did not leave the model inaccurately predicted as leaving.
# 
# False negatives (bottom-left quadrant): 346, The number of people who left that the model inaccurately predicted did not leave
# 
# True positives (bottom-right quadrant): 125, The number of people who left the model accurately predicted as leaving

# In[56]:


# Check class balance in `left` column
df_logreg["left"].value_counts(normalize=True)


# Approximately 83%-17% split. No need to resample.

# In[57]:


# Classification report for logistic regression model
target_names = ["Predicted would not leave", "Predicted would leave"]
print(classification_report(y_test, y_pred, target_names=target_names))


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
