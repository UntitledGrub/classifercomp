# Classifier Comparison
Comparison of different classifiers (KNN, Logistic Regression, Random Forest) on a binary classification task

## Dependencies
sklearn <br />
pandas <br />
numpy

## Implementation

Five fold cross validation is used to identify the optimal hyperparameters for each of the three classifiers tested. For random forest we use this method to set number of trees and maximum depth of trees, for K-NN to set the value of K, and for logistic regression to set maximum iterations. Possible hyperparameter values and combinations thereof are searched exhaustively (within a preset range) to find the optimal values. 

We then compare the best performing versions of each model on a simple binary classificaiton task. 

## Dataset
The dataset used is the Pima Indians Diabetes Database (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Results
![Figure_2](https://user-images.githubusercontent.com/34168073/190226783-ce35840c-4280-4367-8926-1d410aebd203.png)
![Figure_3](https://user-images.githubusercontent.com/34168073/190226800-2a36ef69-40a2-4078-9fd1-83d3f05422f9.png)
