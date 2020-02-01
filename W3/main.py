import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

pd.options.mode.chained_assignment = None

corpus = pd.read_csv('../corpus.csv')

predictors = corpus[['incomeperperson','employrate','suicideDanger','alcconsumption','femaleemployrate','lifeexpectancy']]

predictors.replace(' ', np.nan, inplace=True)
predictors = predictors.dropna()

predictors['incomeperperson']  = preprocessing.scale(predictors['incomeperperson'].astype('float64'))
predictors['employrate']       = preprocessing.scale(predictors['employrate'].astype('float64'))
predictors['alcconsumption']   = preprocessing.scale(predictors['alcconsumption'].astype('float64'))
predictors['femaleemployrate'] = preprocessing.scale(predictors['femaleemployrate'].astype('float64'))
predictors['lifeexpectancy'] = preprocessing.scale(predictors['lifeexpectancy'].astype('float64'))

target     = predictors[['suicideDanger']]
predictors = predictors[['incomeperperson','employrate','alcconsumption','femaleemployrate','lifeexpectancy']]

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=.15, random_state = 0)

# Implementation whith cv = 3
model=LassoLarsCV(cv=100, precompute=False).fit(pred_train, tar_train.values.ravel())

print(dict(zip(predictors.columns, model.coef_)))

m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.show()

m_log_alphascv = -np.log10(model.cv_alphas_+0.001)
plt.figure()
plt.plot(m_log_alphascv, model.mse_path_, ':')
plt.plot(m_log_alphascv, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.show()

train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print (' Training data MSE ', train_error)
print (' Test data MSE ', test_error)

# R-square from training and test data
rsquared_train = model.score(pred_train,tar_train)
rsquared_test = model.score(pred_test,tar_test)
print (' Training data R-square ', rsquared_train * 100)
print (' Test data R-square ', rsquared_test  * 100)
