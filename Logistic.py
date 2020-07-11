import pandas as pd
import numpy as np
import matplotlib as mpl
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score



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

test_df = pd.DataFrame(list(zip(test_dataframe['T2'].tolist(),
                                test_dataframe['T2/MUSC'].tolist(),
                                test_dataframe['T2/TON'].tolist(),
                                test_dataframe['ADC'])),
                                columns=['T2', 'T2/MUSC','T2/TON', 'ADC'])

XTest = test_df.to_numpy()

lr = LogisticRegression(penalty='l1', solver='liblinear')
lr.fit(scale(XTrain), yTrain)

crossValidation = KFold(n_splits = 5, shuffle = True, random_state=1)
aucs = []
for i, (train, test) in enumerate(crossValidation.split(XTrain,yTrain)):
    lr.fit(scale(XTrain[train]), yTrain[train])
    yScore = lr.predict_proba(scale(XTrain[test]))
    aucs.append(roc_auc_score(yTrain[test], yScore[:,1]))

print("mean AUC : %.2f" % np.mean(np.abs(aucs)))
print("std in accuracy : %.2f" % np.std(aucs))

print("Train data score : %.3f"% lr.score(scale(XTrain),yTrain))
print("Test data score : %.3f"% lr.score(scale(XTest),yTest))

for  s, n in sorted(zip(np.transpose(lr.coef_), prod_df.columns)):
    print("Coeff %3.2f \t for feature %s "% (s, n))


yTrainScore = lr.predict(scale(XTrain))
yTestScore = lr.predict(scale(XTest))
#print (list(zip (yTrainScore, yTrain)))
#print (list(zip (yTestScore, yTest)))



yPlotScore = lr.predict_proba(scale(XTrain))[:,1]
auc = roc_auc_score(yTrain, yPlotScore)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(yTrain, yPlotScore)
plot_roc_curve(fpr, tpr, auc)


