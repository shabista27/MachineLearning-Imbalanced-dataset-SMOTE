'''
Supervised Learning of ADHD data

Author: Shabista Shaikh

'''
# Import Libraries
import numpy as np
import csv
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from numpy.random import RandomState
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTENC

# Set random seed
np.random.seed(0)

# Acquire file name
train = "adhd_6to9.csv"
fields = []
rows = []
f = open("output_6to9.txt", "w+")

# Train the model on training data
with open(train, 'r') as csvfile:
    # Create a csv reader object
    csvreader = csv.reader(csvfile)
    # Column heading
    fields = next(csvreader)
    # Data
    for row in csvreader:
        rows.append(row)

## Filter datapoints for further processing
x = np.array(rows)  # The data matrix
#x = x[x[:,4].astype(int) == 1] # remove datapoints with inconsistent RS score
# x = x[x[:,3].astype(int) != 0] # remove normal datapoints
#x = x[x[:,5].astype(int) < 10] # retain data for age less than 10

## Divide data into diagnosed or undiagnosed
x_d = x[x[:,2].astype(int) == 1] # Extract diagnosed data points
x_d = x # Include undiagnosed samples in training set
#x_u = x[x[:,2].astype(int) == 0] # Extract undiagnosed data points
l_d = x_d[:, [0,1]] # Extract subject labels (Subject Name, Subject ID) for diagnosed subjects 
#l_u = x_u[:, [0,1]] # Extract subject labels (Subject Name, Subject ID) for undiagnosed subjects 
x_d = np.delete(x_d, [0,1,2], 1) # Remove subject and diagnosed labels from diagnosed data
#x_u = np.delete(x_u, [0,1,2], 1) # Remove subject and diagnosed labels from undiagnosed data
x_d = x_d.astype(float) # Convert the data to float
#x_u = x_u.astype(float) # Convert the data to float
y_d = x_d[:, 0] # Extract output label for diagnosed data
#y_u = x_u[:, 0] # Extract output label for undiagnosed data
x_d = np.delete(x_d, 0, 1) # Remove output label from diagnosed data
#x_u = np.delete(x_u, 0, 1) # Remove output label from undiagnosed data


## Define the classifier
#multi_class='multinomial'
logReg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial') # Logistic Regression Classifier
clf = make_pipeline(preprocessing.StandardScaler(), logReg) # Chain a preprocesing step with the Classifier

## Cross-validate the model
cv = StratifiedKFold(n_splits=5) # Balanced K-fold
scores = []
for i, (train_index, test_index) in enumerate(cv.split(x_d, y_d)):
	f.write("\nCross-validation iteration: %d \n\n" % (i + 1))
	x_train, x_test = x_d[train_index], x_d[test_index]
	y_train, y_test = y_d[train_index], y_d[test_index]
	l_train, l_test = l_d[train_index], l_d[test_index]
	sm=SMOTE(random_state=0)
	X_train_res, y_train_res = sm.fit_sample(x_train,y_train)
	clf.fit(X_train_res,y_train_res.ravel())
	scores.append(clf.score(x_test, y_test))
	y_pred=clf.predict(x_test)
	confmat=confusion_matrix(y_test,y_pred)
	f.write("Confusion matrix:\n")
	np.savetxt(f,confmat,fmt="%s")
	perc=1.0 * np.diag(confmat) / np.sum(confmat,axis=1)
	f.write("Class wise Accuarcy:\n")
	np.savetxt(f,perc,fmt="%s")

scores = np.array(scores)
f.write("\nStratified K-fold cross-validation score: %f\n" % scores.mean())

cv = ShuffleSplit(n_splits=100, test_size=.2, random_state=0) # Random Split
scores = cross_val_score(clf, x_d, y_d, cv=cv)
f.write("Random-split cross-validation score (100 runs): %f\n" % scores.mean())

## Generate predictions for undiagnosed data
#clf.fit(x_d, y_d)
#y_pred = clf.predict(x_u)
#out = np.transpose(np.vstack((l_u[:, 0], l_u[:, 1], y_pred, y_u)))
#f.write("\nResult on undiagnosed data\n\n")
#f.write("Subject Name, Subject ID, Predicted Category, Actual Category\n")
#np.savetxt(f, out, fmt="%s")
f.close()
