import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)

TRAIN_FEATURES = train.iloc[1:,0:8]
TRAIN_TARGET = pd.to_numeric(train.iloc[1:,8], errors='coerce')

TEST_FEATURES = test.iloc[1:,0:8]
TEST_TARGET = pd.to_numeric(test.iloc[1:,8], errors='coerce')

##### Cross validation for optimal hyperparameters #####

# Exhaustively searches for optimal combination of hyperparameters values
# Searches specified value ranges for specified hyperparameters
# Parameters: sklearn classifier
# Parameters: hyperparams: a dictionary, str: iterable - hyperparams: iterables containing possible values of the hyperparameters
# Hyperparameter names must be compatible with those of the sk.learn classifier being considered
# Return: tuple - the optimal hyperparameter values for the hyperparameters and value spaces specified
def crossValHyperParams(classifier, hyperparams):
    hyperparamVectors = product(*hyperparams.values())

    paramCompare = dict()
    
    for hpvector in hyperparamVectors:
        classifier.set_params(**{hp: hpvector[i] for (i,hp) in enumerate(hyperparams.keys())}) 
        # The folds here will be stratified because the estimators we're using are classifers
        
        score = cross_validate(classifier, TRAIN_FEATURES, TRAIN_TARGET, cv=5, scoring='recall')
        paramCompare[hpvector] = round(score['test_score'].mean(), 2)

    best = 0
    for hpvector in paramCompare.keys():
        # We only update the optimal if we find hyper parameter values 
        if paramCompare[hpvector] > best:
            best = paramCompare[hpvector]
            optimal = hpvector
    print(paramCompare)
    return optimal, paramCompare

##### KNN-classifier #####

knn = KNeighborsClassifier()
knnoptimalHP, paramGraphData = crossValHyperParams(knn, {'n_neighbors': range(1, 100, 2), 'metric': {'euclidean', 'manhattan', 'minkowski'}})

print(knnoptimalHP) 

# k = 5, distance metric = euclidean acheives recall of 0.6
# k = 89, distance metric = manhattan acheives precision of 0.78
# k = 5, distance metric = euclidean acheives f score of 0.62
# k = 17, distance metric = euclidean acheives accuracy of 0.76

##### RandomForest #####

rf = RandomForestClassifier(random_state=42)
rfoptimalHP, paramGraphData = crossValHyperParams(rf, {'n_estimators': range(1, 50), 'max_leaf_nodes': range(2, 20)})

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter([x[0] for x in paramGraphData.keys()], [y[1] for y in paramGraphData.keys()], [z for z in paramGraphData.values()])
ax.set_ylim(0, 20)
ax.set_zlim(0, 1)
ax.set_xlabel('Number of Trees')
ax.set_ylabel('Maximum Depth')
ax.set_yticks(range(0, 21, 5))
ax.set_zlabel('Average Recall across Cross Validation Folds')
plt.show()

print(rfoptimalHP)

# trees = 1, depth = 8 acheives recall of 0.63
# trees = 20, depth = 2 acheives precision 0.95
# trees = 6, depth = 16 acheives f-score of 0.63
# tress = 13, depth = 16 acheives accuracy of 0.78

##### Logistic Regression #####

lr = LogisticRegression()
lroptimalHP, paramGraphData = crossValHyperParams(lr, {'max_iter': range(1, 100)})
print(lroptimalHP)

# maxIter = 1, achieves recall of 0.61
# maxIter = 3, acheives precision of 0.71
# maxIter = 2, achieves f-score of 0.62
# maxIter = 5, achieves accuracy of 0.77