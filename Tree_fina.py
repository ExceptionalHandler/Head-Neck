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

xls = pd.ExcelFile("all_data.xls")
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

def TrainTree(XTrain, yTrain, sLabel) : 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=4) 

    clf_entropy.fit(XTrain, yTrain)
    yPredicProb = clf_entropy.predict_proba(XTrain)
    yPlotScore = yPredicProb[:,1]
    fpr, tpr, thresholds = roc_curve(yTrain, yPlotScore)
    plt.plot(fpr, tpr, label= sLabel +'(area = %0.2f)' % roc_auc_score(yTrain, yPlotScore))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    #plt.show()

TrainTree(XTrain, yTrain, 'ROC curve T2 + T2/MUSC + T2/TON + ADC')

XTrainNoT2 = XTrain[:,1:4 ]
TrainTree(XTrainNoT2, yTrain, 'ROC curve T2/MUSC + T2/TON + ADC')

XTrainOnlyADC = XTrain[:,-1 ].reshape(-1,1)
TrainTree(XTrainOnlyADC, yTrain, 'ROC curve Only ADC')

patient_dataframe = patient_dataframe.drop('T2/MUSC', 1)
patient_dataframe = patient_dataframe.drop('T2/TON', 1)
XTrainADCAndT2 = patient_dataframe.to_numpy()
#print(XTrainADCAndT2)
TrainTree(XTrainADCAndT2, yTrain, 'ROC curve ADC + T2')

plt.show()