import pandas as pd
import numpy as np
import matplotlib as mpl
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals.six import StringIO  
from sklearn.tree import plot_tree


def plot_roc_curve(fpr, tpr, auc):
    plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

xls = pd.ExcelFile("feature_sheet.xls")
patient_dataframe = xls.parse('Sheet1', index_col = None, na_values= ['NA'])
colours_palette = {1:"red", 0: "blue"}
yTrain = patient_dataframe['DISEASE']

colours = [colours_palette[c] for c in yTrain]
patient_dataframe = patient_dataframe.drop('DISEASE', 1)
patient_dataframe = patient_dataframe.drop('T2 QUAL', 1)
prod_df = pd.DataFrame(list(zip(patient_dataframe['T2'].tolist(),
                                patient_dataframe['T2/MUSC'].tolist(),
                                patient_dataframe['T2/TON'].tolist(),
                                patient_dataframe['ADC'])),
                                columns=['T2', 'T2/MUSC','T2/TON', 'ADC'])

XTrain = prod_df.to_numpy()


xls_test = pd.ExcelFile("test_data.xls")
test_dataframe = xls_test.parse('Sheet1', index_col = None, na_values= ['NA'])
yTest = test_dataframe['DISEASE']
column_names=['T2', 'T2/MUSC','T2/TON', 'ADC']
test_df = pd.DataFrame(list(zip(test_dataframe['T2'].tolist(),
                                test_dataframe['T2/MUSC'].tolist(),
                                test_dataframe['T2/TON'].tolist(),
                                test_dataframe['ADC'])),
                                columns = column_names)

XTest = test_df.to_numpy()

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=5) 

crossValidation = KFold(n_splits = 5, shuffle = True, random_state=1)
aucs = []
for i, (train, test) in enumerate(crossValidation.split(XTrain,yTrain)):
    clf_entropy.fit(XTrain[train], yTrain[train])
    yScore = clf_entropy.predict_proba(XTrain[test])
    aucs.append(roc_auc_score(yTrain[test], yScore[:,1]))

print("mean AUC : %.2f" % np.mean(np.abs(aucs)))
print("std in accuracy : %.2f" % np.std(aucs))
  
# Performing training 
clf_entropy.fit(XTrain, yTrain)
yPredicProb = clf_entropy.predict_proba(XTrain)
yTestProb = clf_entropy.predict_proba(XTest)
yTestPredict = clf_entropy.predict(XTest)
yPlotScore = yPredicProb[:,1]
yTestScore = yTestProb[:,1]
auc = roc_auc_score(yTrain, yPlotScore)
print('AUC: %.2f' % auc)
print('Accuracy in test: %.2f' % clf_entropy.score(XTest, yTest))
print (list(zip (yTestPredict, yTest)))
fpr, tpr, thresholds = roc_curve(yTrain, yPlotScore)
plot_roc_curve(fpr, tpr, auc)

plot_tree(clf_entropy, feature_names=column_names, class_names=['No Disease', 'Disease'] )
plt.show()

