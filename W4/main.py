import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

pd.options.mode.chained_assignment = None

corpus = pd.read_csv('../corpus.csv')

predictors = corpus[['incomeperperson','employrate','suicideDanger','alcconsumption','femaleemployrate','lifeexpectancy']]

predictors.replace(' ', np.nan, inplace=True)
predictors = predictors.dropna()

predictors['incomeperperson']  = preprocessing.scale(predictors['incomeperperson'].astype('float64'))
predictors['employrate']       = preprocessing.scale(predictors['employrate'].astype('float64'))
predictors['alcconsumption']   = preprocessing.scale(predictors['alcconsumption'].astype('float64'))
predictors['femaleemployrate'] = preprocessing.scale(predictors['femaleemployrate'].astype('float64'))
predictors['lifeexpectancy']   = preprocessing.scale(predictors['lifeexpectancy'].astype('float64'))

target     = predictors[['suicideDanger']]
predictors = predictors[['incomeperperson','employrate','alcconsumption','femaleemployrate','lifeexpectancy']]

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=.15, random_state = 0)