

# %% 
# Importing Required Libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# %% 
# We use the Titanic dataset
data = pd.read_csv("input/train.csv")
# We split the data into training and testing sets
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])

# %% 
# Display the first few rows of the dataset (For demonstration purposes)
train.head()

# %%
# Dropping Features
# Since our goal is not to make a better classiefier for the problem, let's train a simple model.
# We will drop the following features: Name, Ticket, Cabin, PassengerId
# Lets only use Pclass, Sex, Age, SibSp, Parch and Embarked features.
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

train = train.drop(['PassengerId'], axis=1)
test = test.drop(['PassengerId'], axis=1)

# Convert categorical variables into dummy/indicator variables
train_processed = pd.get_dummies(train)
test_processed = pd.get_dummies(test)

# Filling Null Values
train_processed = train_processed.fillna(train_processed.mean())
test_processed = test_processed.fillna(test_processed.mean())

# Create X_train,Y_train,X_test
X_train = train_processed.drop(['Survived'], axis=1)
Y_train = train_processed['Survived']

X_test  = test_processed.drop(['Survived'], axis=1)
Y_test  = test_processed['Survived']

# Display
print("Processed DataFrame for Training : Survived is the Target, other columns are features.")
display(train_processed.head())


# %%
# Random Forest Classifier
# Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model. It works by training each decision tree separately with a random sample of the data. The decision trees in a random forest model are trained based on a random sample of the training data. It builds multiple decision trees and merges them together to get a more accurate and stable prediction.
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest_preds = random_forest.predict(X_test)
print('The accuracy of the Random Forests model is :\t',metrics.accuracy_score(random_forest_preds,Y_test))

# %% Importing Required Libraries
import lime
import lime.lime_tabular
import matplotlib.pylab as plt # Importing the matplotlib library
from interpret import show
from interpret.blackbox import LimeTabular

# %%
# Create a LIME Explainer for Tabular Data
predict_fn_rf = lambda x: random_forest.predict_proba(x).astype(float) # The predict_fn should return the predicted probabilities 
X = X_train.values # The data that Lime will explain
explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = X_train.columns,class_names=['Will Die','Will Survive'],kernel_width=5) # The explainer

# %% Explaining Instance
test.loc[[421]]
# %%
# Choosen instance refers to an Unlucky (not survived) Male passenger of age 21 travelling in passenger class 3, embarked from Q. Let's see what and how our model predicts his survival.
choosen_instance = X_test.loc[[421]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)
# Model predicted Will Die (Not survived). Biggest effect is person being a male; This has decreased his chances of survival significantly. Next, passenger class 3 also decreases his chances of survival while being 21 increases his chances of survival.
# %%
test.loc[[310]]
# %%
choosen_instance = X_test.loc[[310]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)
#  Model predicted 1 (Fully confident that passenger survives). Biggest effect is person being a female; This has increased her chances of survival significantly. Next, passenger class 1 and Fare>31 has also increases her chances of survival

#fig = exp.as_pyplot_figure() # Creating a plot
#plt.tight_layout() # Tightening the layout
#plt.show() # Showing the plot

# %%
test.loc[[736]]
# %%
choosen_instance = X_test.loc[[736]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)
# Model predicted Will Die. Biggest effect is person being a female; This has increased her chances of survival significantly. Fare value of 34.38 has also played a part incresing her chances. However, beign a passenger in class 3 and her age (48) has significantly decreased her chances of survival.

# %%
test.loc[[788]]
# %%
choosen_instance = X_test.loc[[788]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)
# Model predicted Will Survive. Although passenger class 3 and being a male passenger has decresed his chances of survival. Biggest effect has come from his age being 1 years old; This has increased his chances of survival significantly.
