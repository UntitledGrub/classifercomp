import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)

TRAIN_FEATURES = train.iloc[1:,0:8]
TRAIN_TARGET = pd.to_numeric(train.iloc[1:,8], errors='coerce')

TEST_FEATURES = test.iloc[1:,0:8]
TEST_TARGET = pd.to_numeric(test.iloc[1:,8], errors='coerce')

# Displays the accuracy, precision, recall, f-score and confusion matrix for a classifer
# Parameters: classifierName, str - name of the classifer to display results of
# Parameter: prediciton, arraylike - a classifier's predicted labels for objects
# Parameter: trueLabels, arraylike - the true class labels of objects
# Returns: accuracy, float; precision, float; recall, float; fScore, float
def modelPredictionResults(classifierName, prediction, trueLabels):
  precision, recall, fScore, support = metrics.precision_recall_fscore_support(trueLabels, prediction, average='binary')
  accuracy = metrics.accuracy_score(trueLabels, prediction)
  print('--------------------------------------------------------------')
  print('Model Output Results for', classifierName)
  print('Accuracy:', accuracy)
  print('Precision:', precision)
  print('Recall:', recall)
  print('F-Score:', fScore)
  print('--------------------------------------------------------------')
  # Generation and Visualisation of confusion matrix
  confusionMatrix = metrics.confusion_matrix(trueLabels, prediction)
  tickMarks = range(2)
  plt.xticks(tickMarks, [1,0])
  plt.yticks(tickMarks, [1,0])
  # Initializing confusion matrix heat map
  sns.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap='coolwarm', fmt='g')
  plt.title('Confusion Matrix of ' + classifierName)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  
  plt.show()

  return accuracy, precision, recall, fScore

##### KNN classifier #####
# Optimal parameters for maximising recall: k=5, euclidean distance metric

knn = KNeighborsClassifier(n_neighbors=89, metric='euclidean')
knn.fit(TRAIN_FEATURES, TRAIN_TARGET)

predictions = knn.predict(TRAIN_FEATURES)
knnTrainAcc, knnTrainPrec, knnTrainRecall, knnTrainFScore = modelPredictionResults('KNN Train', predictions, TRAIN_TARGET)

predictions = knn.predict(TEST_FEATURES)
knnTestAcc, knnTestPrec, knnTestRecall, knnTestFScore = modelPredictionResults('KNN Test', predictions, TEST_TARGET)

##### Random Forest classifier #####
# Optimal parameters for maximising recall: 1 tree, 8 max depth

# Random state is set at 42 for reproducibility fo results 
rf = RandomForestClassifier(n_estimators=1, max_depth=16, random_state=42)
rf.fit(TRAIN_FEATURES, TRAIN_TARGET)

predictions = rf.predict(TRAIN_FEATURES)
rfTrainAcc, rfTrainPrec, rfTrainRecall, rfTrainFscore = modelPredictionResults('Random Forest Train', predictions, TRAIN_TARGET)

predicitons = rf.predict(TEST_FEATURES)
rfTestAcc, rfTestPrec, rfTestRecall, rfTestFScore = modelPredictionResults('Random Forest Test', predicitons, TEST_TARGET)

##### Logistic Regression classifier #####
# Optimal parameters for maximising recall: 1 max iter

lr = LogisticRegression(max_iter=1)
lr.fit(TRAIN_FEATURES, TRAIN_TARGET)

predictions = lr.predict(TRAIN_FEATURES)
lrTrainAcc, lrTrainPrec, lrTrainRecall, lrTrainFscore = modelPredictionResults('Logistic Regression Train', predictions, TRAIN_TARGET)

predicitons = lr.predict(TEST_FEATURES)
lrTestAcc, lrTestPrec, lrTestRecall, lrTestFScore = modelPredictionResults('Logistic Regression Test', predicitons, TEST_TARGET)

##### Comparison of classifier performance #####

def dictToDataframe(dictionary):
  return pd.DataFrame(dictionary, columns=[x for x in dictionary.keys()], index=['K-Nearest Neighbours', 'Random Forest', 'Logistic Regression'])

trainPerformance = dictToDataframe({'Accuracy': [knnTrainAcc, rfTrainAcc, lrTrainAcc], 
                    'Precision': [knnTrainPrec, rfTrainPrec, lrTrainPrec], 
                    'Recall': [knnTrainRecall, rfTrainRecall, lrTrainRecall], 
                    'F-Score': [knnTrainFScore, rfTrainFscore, lrTrainFscore]})

testPerformance = dictToDataframe({'Accuracy': [knnTestAcc, rfTestAcc, lrTestAcc], 
                    'Precision': [knnTestPrec, rfTestPrec, rfTestPrec], 
                    'Recall': [knnTestRecall, rfTestRecall, lrTestRecall],
                    'F-Score': [knnTestFScore, lrTestFScore, lrTestFScore]})

trainPerformance.plot.barh()
plt.title('Training Performance of Binary Classifiers')
plt.show()

testPerformance.plot.barh()
plt.title('Test Performance of Binary Classifiers')
plt.show()
