import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection  import GridSearchCV
from sklearn.svm import SVC


xls = pd.ExcelFile("feature_sheet.xls")
patient_dataframe = xls.parse('Sheet1', index_col = None, na_values= ['NA'])

y = patient_dataframe['DISEASE']

patient_dataframe = patient_dataframe.drop('DISEASE', 1)
patient_dataframe = patient_dataframe.drop('T2 QUAL', 1)
X = patient_dataframe.to_numpy()

learning_algo= SVC(class_weight='balanced', random_state=101)
search_space=[{'kernel':['linear'], 'C':np.logspace(-3, 3, 7)},
              {'kernel':['rbf'], 'degree' : [2,3,4], 'C':np.logspace(-3, 3, 7), 'gamma' : np.logspace(-3,2,6)}]

gridsearch = GridSearchCV(learning_algo, n_jobs=4, param_grid=search_space, scoring='accuracy', refit=True, cv=10)

gridsearch.fit(X,y)

cv_performance = gridsearch.best_score_

print("Score : ", gridsearch.best_score_)
print("Best params :", gridsearch.best_params_)
