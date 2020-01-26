import pandas as pd
import numpy as np
import operator
import pydot
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

corpus = pd.read_csv('../corpus.csv')

dic = {}

for i,col in enumerate(corpus.columns): 
  print(" ",i,col)
  dic[i] = col

features = []
feature_sel = 0

print(" Select features (-1 exit)")

while(feature_sel != -1):
  feature_sel = int(input("> "))
  if (feature_sel >= 0):
    features.append(dic[feature_sel])
    print(" ",features)

estimators = int(input(" Select estimators (1-100) > "))

features.append('suicideDanger')
corpus = corpus[features]

corpus.replace(' ', np.nan, inplace=True)
corpus = corpus.dropna()
corpus = corpus.reset_index(drop=True)

features.pop()
X_train = corpus[features]
Y_trian = corpus[['suicideDanger']]

dummy = Y_trian['suicideDanger'].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_train,Y_trian, random_state = 0)

scores_train    = []
scores_test     = []
best_score_test = 0
estimators_list = list(range(1,estimators))

for estimators_in in range(1,estimators):
  clas = RandomForestClassifier(n_estimators=estimators_in, random_state = 0).fit(X_train, y_train.values.ravel())
  pred = clas.predict(X_test)
  y_test_copy = y_test['suicideDanger'].values.tolist()
  pred  = zip(pred,y_test_copy)
  score = clas.score(X_test,y_test_copy)
  scoreT = clas.score(X_train,y_train)
  scores_train.append(scoreT)
  scores_test.append(score)

  if best_score_test < score:
    best_score_test = score

# print()
# print("Test score: ",scores_test)
# print()
# print("Train score: ",scores_train)
# print()
print()
print(" Best score:", best_score_test)
print()

feat_score = list(zip(features, clas.feature_importances_))
feat_score.sort(key = operator.itemgetter(1), reverse = True)

print(" Features: ", feat_score)


plt.figure()
plt.plot(estimators_list, scores_train, label='training data')
plt.plot(estimators_list, scores_test, label='test data')
plt.legend(loc=4)
plt.show()





