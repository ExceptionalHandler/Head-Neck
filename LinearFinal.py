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
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2



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

lr = LogisticRegression(penalty='l1', solver='liblinear')
lr.fit(scale(XTrain), yTrain)

for  s, n in sorted(zip(np.transpose(lr.coef_), prod_df.columns)):
    print("Coeff %3.2f \t for feature %s "% (s, n))

selector_f = SelectPercentile(f_classif, percentile=25)
selector_f.fit(scale(XTrain), yTrain)
for  s, n in sorted(zip(selector_f.scores_, prod_df.columns)):
    print("F-score %3.2f \t for feature %s "% (s, n))


selector_c = SelectPercentile(chi2, percentile=25)
selector_c.fit(XTrain, yTrain)
for  s, n in sorted(zip(selector_c.scores_, prod_df.columns)):
    print("Chi-sq-score %3.2f \t for feature %s "% (s, n))

