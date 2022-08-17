import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('diabetes.csv')

##### Data Cleaning #####

# In the dataset we are working with, missing data is represented by 0s
# This function replaces missing values in a column with the mean of the non-zero values in the column
# Parameters: columnName, string - the header of the column
# Returns: nothing
def cleanColumn(columnName):
    # Set all 0 entries to pd.NA so we can use pd.df.fillna to quickly replace the missing values
    for row in data.index:
        if data.loc[row, columnName] == 0:
            data.loc[row, columnName] = pd.NA
    
    data[columnName].fillna(data[columnName].mean(skipna=True), inplace=True)

columnsToClean = {'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'}

for columnName in columnsToClean:
    cleanColumn(columnName)

##### Normalising Data #####

# Applies Gaussian normalisation to columns
# Parameters: columnName, string - the header of the column
# Returns: nothing
def normaliseColumn(columnName):
        data[columnName] = (data[columnName]-data[columnName].mean())/data[columnName].std()

# Only the feature columns are normalised, the class labels are left as either 1s or 0s
for columnName in {'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'}:
    normaliseColumn(columnName)

##### Test-Train Splitting #####
 
# stratify=data['Outcome'] maintains relative frequencies of classes in the train and test data
trainInstances, testInstances = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['Outcome'])

# index=False stops a column of row indices from being written to the .csv file
trainInstances.to_csv('train.csv', index=False)
testInstances.to_csv('test.csv', index=False)
