import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

xls = pd.ExcelFile("feature_sheet.xls")
patient_dataframe = xls.parse('Sheet1', index_col = None, na_values= ['NA'])
#print (patient_dataframe.columns)
colours_palette = {1:"red", 0: "blue"}
output_data = patient_dataframe['DISEASE']
colours = [colours_palette[c] for c in output_data]
#print (colours)
patient_dataframe = patient_dataframe.drop('DISEASE', 1)
patient_dataframe = patient_dataframe.drop('T2 QUAL', 1)
#print(patient_dataframe)
log_f2 = np.log(patient_dataframe['T2'])
log_f3 = np.log(patient_dataframe['T2/TON'])
log_f4 = np.log(patient_dataframe['T2/MUSC'])
log_f5 = np.log(patient_dataframe['ADC'])
patient_df_log = pd.DataFrame(list(zip(log_f2, log_f3, log_f4,log_f5 )))

sq_f2 = np.square(patient_dataframe['T2'])
sq_f3 = np.square(patient_dataframe['T2/TON'])
sq_f4 = np.square(patient_dataframe['T2/MUSC'])
sq_f5 = np.square(patient_dataframe['ADC'])

big_df = pd.DataFrame(list(zip(patient_dataframe['T2'].tolist(), 
                               patient_dataframe['T2/TON'].tolist(), 
                               patient_dataframe['T2/MUSC'].tolist(),
                               patient_dataframe['ADC'].tolist(),
                               log_f2, log_f3, log_f4,log_f5,
                               sq_f2, sq_f3, sq_f4,sq_f5)),
                      columns=['T2', 'T2/TON', 'T2/MUSC', 'ADC', 
                               'lnT2', 'lnT2/TON', 'lnT2/MUSC', 'lnADC',
                               'sqT2', 'sqT2/TON', 'sqT2/MUSC', 'sqADC'])
print(big_df)
#matrix_of_scatterplots = scatter_matrix(big_df, figsize=(30,30), color=colours, diagonal = 'kde')
#plt.show()

X= big_df.to_numpy()
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, output_data)

print(clf.coef_)
print(clf.score(X, output_data))

alt_df = pd.DataFrame(list(zip(sq_f2, sq_f5)),
                      columns=['sqT2', 'sqADC'])

# Z= alt_df.to_numpy()
# clf2 = Perceptron(tol=1e-3, random_state=0)
# clf2.fit(Z, output_data)

# print(clf2.classes_, clf2.coef_)
# print(clf2.score(Z, output_data))

# plot_data(Z, output_data, clf2.decision_function)
# # matrix_of_scatterplots = scatter_matrix(alt_df, figsize=(6,6), color=colours, diagonal = 'kde')
# plt.show()

# Z= patient_dataframe.to_numpy()
# clf2 = Perceptron(tol=1e-3, random_state=0)
# clf2.fit(Z, output_data)

# print(clf2.classes_, clf2.coef_)
# print(clf2.score(Z, output_data))

new_dataframe = patient_dataframe.drop('T2/TON', 1)
new_dataframe = new_dataframe.drop('T2/MUSC', 1)

A= new_dataframe.to_numpy()
clf3 = Perceptron(tol=1e-3, random_state=0)
clf3.fit(A, output_data)

print(new_dataframe)
print(clf3.classes_, clf3.coef_)
print(clf3.score(A, output_data))






