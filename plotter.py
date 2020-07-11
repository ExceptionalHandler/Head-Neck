import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
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

sq_f2 = np.power(patient_dataframe['T2'], 2)
sq_f3 = np.square(patient_dataframe['T2/TON'])
sq_f4 = np.square(patient_dataframe['T2/MUSC'])
sq_f5 = np.square(patient_dataframe['ADC'])

columns_s=['T2', 'T2/TON', 'T2/MUSC', 'ADC', 'lnT2', 
         'lnT2/TON', 'lnT2/MUSC', 'lnADC','sqT2', 
         'sqT2/TON', 'sqT2/MUSC', 'sqADC']
big_df = pd.DataFrame(list(zip(patient_dataframe['T2'].tolist(), 
                               patient_dataframe['T2/TON'].tolist(), 
                               patient_dataframe['T2/MUSC'].tolist(),
                               patient_dataframe['ADC'].tolist(),
                               log_f2, log_f3, log_f4,log_f5,
                               sq_f2, sq_f3, sq_f4,sq_f5)),
                            columns = columns_s)
big_df['prod1'] = big_df['T2']*big_df['ADC']
big_df['ratio1'] = big_df['T2']/big_df['ADC']
big_df['prod_sqrt'] = np.sqrt(big_df['prod1'])

#big_df['result'] = output_data

#print(big_df)
#matrix_of_scatterplots = scatter_matrix(big_df, figsize=(30,30), color=colours, diagonal = 'kde')
#plt.show()
X= big_df.to_numpy()
selector_f = SelectPercentile(f_classif, percentile=25)
selector_f.fit(scale(X), output_data)
for  s, n in sorted(zip(selector_f.scores_, big_df.columns)):
    print("F-score %3.2f \t for feature %s "% (s, n))
# clf = Perceptron(tol=1e-3, random_state=0)
# clf.fit(X, output_data)

# print(clf.coef_)
# print(clf.score(X, output_data))

# alt_df = pd.DataFrame(list(zip(sq_f2, sq_f5)),
#                       columns=['sqT2', 'sqADC'])

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

# new_dataframe = patient_dataframe.drop('T2/TON', 1)
# new_dataframe = new_dataframe.drop('T2/MUSC', 1)

# A= new_dataframe.to_numpy()
# clf3 = Perceptron(tol=1e-3, random_state=0)
# clf3.fit(A, output_data)

# print(new_dataframe)
# print(clf3.classes_, clf3.coef_)
# print(clf3.score(A, output_data))

prod_df = pd.DataFrame(list(zip(big_df['T2'].tolist(),
                                big_df['T2/MUSC'].tolist(),
                                big_df['T2/TON'].tolist(),
                                big_df['ADC'])),
                                columns=['T2', 'T2/MUSC','T2/TON', 'ADC'])

B = prod_df.to_numpy()
# matrix_of_scatterplots = scatter_matrix(prod_df, figsize=(30,30), color=colours, diagonal = 'kde')
# plt.show()
clf3 = Perceptron(tol=1e-3, random_state=3)
clf3.fit(scale(B), output_data)

# print(prod_df)

print(clf3.classes_, clf3.coef_)
print(clf3.score(scale(B), output_data))

####################################################################

lr = LogisticRegression(penalty='l1', solver='liblinear')
lr.fit(scale(X), output_data)

crossValidation = KFold(n_splits = 5, shuffle = True, random_state=1)
aucs = []
for i, (train, test) in enumerate(crossValidation.split(X,output_data)):
    lr.fit(scale(X[train]), output_data[train])
    yScore = lr.predict_proba(scale(X[test]))
    aucs.append(roc_auc_score(output_data[test], yScore[:,1]))

print("mean AUC : %.2f" % np.mean(np.abs(aucs)))
print("std in accuracy : %.2f" % np.std(aucs))

print(lr.score(scale(X),output_data))


#print (list(zip (yScore, output_data)))

xls_test = pd.ExcelFile("test_data.xls")
test_dataframe = xls_test.parse('Sheet1', index_col = None, na_values= ['NA'])
yTest = test_dataframe['DISEASE']
test_dataframe = test_dataframe.drop('DISEASE', 1)
test_dataframe = test_dataframe.drop('T2 QUAL', 1)
log_f2 = np.log(test_dataframe['T2'])
log_f3 = np.log(test_dataframe['T2/TON'])
log_f4 = np.log(test_dataframe['T2/MUSC'])
log_f5 = np.log(test_dataframe['ADC'])

sq_f2 = np.power(test_dataframe['T2'], 2)
sq_f3 = np.square(test_dataframe['T2/TON'])
sq_f4 = np.square(test_dataframe['T2/MUSC'])
sq_f5 = np.square(test_dataframe['ADC'])

columns_s1=['T2', 'T2/TON', 'T2/MUSC', 'ADC', 'lnT2', 
         'lnT2/TON', 'lnT2/MUSC', 'lnADC','sqT2', 
         'sqT2/TON', 'sqT2/MUSC', 'sqADC']
big1_df = pd.DataFrame(list(zip(test_dataframe['T2'].tolist(), 
                               test_dataframe['T2/TON'].tolist(), 
                               test_dataframe['T2/MUSC'].tolist(),
                               test_dataframe['ADC'].tolist(),
                               log_f2, log_f3, log_f4,log_f5,
                               sq_f2, sq_f3, sq_f4,sq_f5)),
                            columns = columns_s1)
big1_df['prod1'] = big1_df['T2']*big1_df['ADC']
big1_df['ratio1'] = big1_df['T2']/big1_df['ADC']
big1_df['prod_sqrt'] = np.sqrt(big1_df['prod1'])



XTest = big1_df.to_numpy()
yScore = lr.predict_proba(scale(X))

yScoreTest = lr.predict(scale(XTest))
yProbTest = lr.predict_proba(scale(XTest))
print(lr.score(scale(XTest),yTest))
print (list(zip (yScoreTest, yTest)))

yPlotScore = yScore[:,1]
auc = roc_auc_score(output_data, yPlotScore)
print('AUC on Train: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(output_data, yPlotScore)
plot_roc_curve(fpr, tpr, auc)



for  s, n in sorted(zip(np.transpose(lr.coef_), big_df.columns)):
    print("Coeff %3.2f \t for feature %s "% (s, n))

#prod_df['Result'] = lr.predict_proba(scale(B))[:,1]
#matrix_of_scatterplots = scatter_matrix(prod_df, figsize=(30,30), color=colours, diagonal = 'kde')
#plt.show()




